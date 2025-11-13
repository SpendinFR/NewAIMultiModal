"""Data source abstractions used across the training loop."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional


COLORS = ["red", "blue", "green", "yellow", "purple", "gray"]
MATERIALS = ["metal", "rubber"]
SHAPES = ["cube", "sphere", "cylinder"]


@dataclass
class ObjectDescription:
    color: str
    material: str
    shape: str
    size: float
    position: List[float]


@dataclass
class DataSample:
    """Simple multimodal sample used by the reference pipeline."""

    text: str
    vision: List[ObjectDescription]
    state: List[float]
    action: str
    intervention: Optional[str]
    target_text: str


class BaseDataSource:
    def __iter__(self) -> Iterator[DataSample]:  # pragma: no cover - interface only
        raise NotImplementedError


class SyntheticCLEVRSampler(BaseDataSource):
    """Generates CLEVR-like scenes with textual descriptions."""

    def __init__(self, num_scenes: int = 1000, seed: int = 0) -> None:
        self.num_scenes = num_scenes
        self.random = random.Random(seed)

    def _sample_object(self) -> ObjectDescription:
        color = self.random.choice(COLORS)
        material = self.random.choice(MATERIALS)
        shape = self.random.choice(SHAPES)
        size = round(self.random.uniform(0.5, 1.5), 2)
        position = [round(self.random.uniform(-3.0, 3.0), 2) for _ in range(3)]
        return ObjectDescription(color=color, material=material, shape=shape, size=size, position=position)

    def _describe_scene(self, objects: List[ObjectDescription]) -> str:
        chunks = [
            f"{obj.material} {obj.shape} {obj.color} at ({obj.position[0]}, {obj.position[1]})"
            for obj in objects
        ]
        return "; ".join(chunks)

    def __iter__(self) -> Iterator[DataSample]:
        for _ in range(self.num_scenes):
            num_objects = self.random.randint(2, 5)
            objects = [self._sample_object() for _ in range(num_objects)]
            text = self._describe_scene(objects)
            action = self.random.choice(["move", "paint", "swap", "describe"])
            intervention = None
            if self.random.random() < 0.3:
                target_idx = self.random.randrange(len(objects))
                new_color = self.random.choice([c for c in COLORS if c != objects[target_idx].color])
                intervention = f"paint object_{target_idx} {new_color}"
            state = [self.random.uniform(-1.0, 1.0) for _ in range(4)]
            yield DataSample(
                text=text,
                vision=objects,
                state=state,
                action=action,
                intervention=intervention,
                target_text=text,
            )


class JSONLinesSource(BaseDataSource):
    """Loads scenes from a JSONL file when available."""

    def __init__(self, path: Path, seed: int = 0) -> None:
        self.path = path
        self.random = random.Random(seed)
        self._data: List[DataSample] = []
        if path.exists():
            with path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    item = json.loads(line)
                    objects = [
                        ObjectDescription(
                            color=obj.get("color", self.random.choice(COLORS)),
                            material=obj.get("material", self.random.choice(MATERIALS)),
                            shape=obj.get("shape", self.random.choice(SHAPES)),
                            size=float(obj.get("size", 1.0)),
                            position=list(obj.get("position", [0.0, 0.0, 0.0])),
                        )
                        for obj in item.get("vision", [])
                    ]
                    self._data.append(
                        DataSample(
                            text=item.get("text", ""),
                            vision=objects,
                            state=item.get("state", [0.0, 0.0, 0.0, 0.0]),
                            action=item.get("action", "describe"),
                            intervention=item.get("intervention"),
                            target_text=item.get("target_text", item.get("text", "")),
                        )
                    )
        else:
            self._data = list(SyntheticCLEVRSampler(num_scenes=256, seed=seed))

    def __iter__(self) -> Iterator[DataSample]:
        for sample in self._data:
            yield sample


class MixedDataLoader:
    """Stitches multiple sources together according to the ratios in the config."""

    def __init__(
        self,
        real_source: Optional[BaseDataSource],
        synthetic_source: Optional[BaseDataSource],
        batch_size: int,
        real_ratio: float,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.real_samples = list(real_source) if real_source is not None else []
        self.synthetic_samples = list(synthetic_source) if synthetic_source is not None else []
        if not self.real_samples and not self.synthetic_samples:
            raise ValueError("At least one data source must be provided")
        self.batch_size = batch_size
        self.real_ratio = real_ratio
        self.rng = rng or random.Random(0)

    def _sample_from(self, pool: List[DataSample]) -> DataSample:
        return self.rng.choice(pool)

    def batches(self) -> Iterator[List[DataSample]]:
        """Yield infinite batches respecting the requested ratios."""

        while True:
            batch: List[DataSample] = []
            for _ in range(self.batch_size):
                if self.rng.random() < self.real_ratio and self.real_samples:
                    batch.append(self._sample_from(self.real_samples))
                else:
                    batch.append(self._sample_from(self.synthetic_samples or self.real_samples))
            yield batch
