"""Simplified multimodal perception encoder."""
from __future__ import annotations

import hashlib
import random
from typing import Dict, Iterable, List

from ..data.sources import DataSample, ObjectDescription
from ..math_utils import Vector, matvec, random_matrix, zeros


class PerceptionEncoder:
    """Turns raw multimodal inputs into a shared feature space."""

    def __init__(self, config: Dict[str, Dict[str, int]], slot_dim: int, seed: int = 0) -> None:
        self.config = config
        self.slot_dim = slot_dim
        rng = random.Random(seed)
        self.text_proj = random_matrix(slot_dim, config["text_transformer"]["hidden_dim"], rng)
        self.vision_proj = random_matrix(slot_dim, config["vision_encoder"]["hidden_dim"], rng)
        self.state_proj = random_matrix(slot_dim, slot_dim, rng)

    def _hash_tokens(self, text: str, dim: int) -> Vector:
        vec = zeros(dim)
        tokens = text.lower().split()
        if not tokens:
            return vec
        for idx, token in enumerate(tokens):
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            bucket = int(digest, 16) % dim
            vec[bucket] += 1.0 + (idx + 1) / len(tokens)
        scale = len(tokens) or 1
        return [value / scale for value in vec]

    def _encode_objects(self, objects: List[ObjectDescription], dim: int) -> Vector:
        vec = zeros(dim)
        if not objects:
            return vec
        for idx, obj in enumerate(objects):
            color_idx = abs(hash(obj.color)) % dim
            shape_idx = abs(hash(obj.shape)) % dim
            material_idx = abs(hash(obj.material)) % dim
            vec[color_idx] += 0.5
            vec[shape_idx] += 0.5
            vec[material_idx] += 0.5
            vec[(color_idx + idx) % dim] += obj.size
            vec[(shape_idx + idx) % dim] += sum(obj.position)
        return [value / len(objects) for value in vec]

    def _encode_state(self, state: Iterable[float], dim: int) -> Vector:
        values = list(state)
        if not values:
            values = [0.0, 0.0, 0.0, 0.0]
        padded = zeros(dim)
        limit = min(dim, len(values))
        for idx in range(limit):
            padded[idx] = float(values[idx])
        return padded

    def encode(self, sample: DataSample) -> List[Vector]:
        """Return a list of slot-sized vectors representing each modality."""

        text_dim = self.config["text_transformer"]["hidden_dim"]
        text_features = matvec(self.text_proj, self._hash_tokens(sample.text, text_dim))

        vision_dim = self.config["vision_encoder"]["hidden_dim"]
        vision_features = matvec(self.vision_proj, self._encode_objects(sample.vision, vision_dim))

        state_features = matvec(self.state_proj, self._encode_state(sample.state, self.slot_dim))

        return [text_features, vision_features, state_features]
