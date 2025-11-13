"""Utility helpers to load and validate YAML configurations."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback path
    yaml = None


@dataclass
class ExperimentConfig:
    """Basic experiment settings."""

    name: str
    seed: int
    output_dir: str


@dataclass
class DataConfig:
    """Configuration of real/synthetic sources and augmentations."""

    sources: Mapping[str, List[Mapping[str, Any]]]
    augmentations: List[Mapping[str, Any]] = field(default_factory=list)
    batch_mix: Mapping[str, float] = field(default_factory=dict)
    trajectory_buffer: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Dictionary-style configuration for the model components."""

    perception: Mapping[str, Any]
    slots: Mapping[str, Any]
    graph: Mapping[str, Any]
    dynamics: Mapping[str, Any]
    reasoner: Mapping[str, Any]
    decoder: Mapping[str, Any]


@dataclass
class LossConfig:
    """Loss coefficients used during training."""

    lambda_rec: float
    lambda_slot: float
    lambda_contr: float
    lambda_dyn: float
    lambda_interv: float
    lambda_text: float
    lambda_kl: float
    lambda_symbolic: float = 0.0


@dataclass
class TrainingStepsConfig:
    pretrain: int
    dynamics: int
    finetune: int


@dataclass
class TrainingConfig:
    optimizer: str
    lr_pretrain: float
    lr_finetune: float
    weight_decay: float
    warmup_steps: int
    batch_size: int
    grad_clip: float
    total_steps: TrainingStepsConfig
    evaluation_interval: int


@dataclass
class EvalConfig:
    metrics: List[str]
    probes: Mapping[str, Any]
    planning: Mapping[str, Any]


@dataclass
class RootConfig:
    """Top-level configuration object used across the codebase."""

    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    phases: Mapping[str, Mapping[str, Any]]
    eval: EvalConfig

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> "RootConfig":
        """Create a :class:`RootConfig` from a Python mapping."""

        missing = {k for k in ["experiment", "data", "model", "loss", "training", "phases", "eval"] if k not in cfg}
        if missing:
            raise ValueError(f"Missing configuration sections: {sorted(missing)}")

        experiment = ExperimentConfig(**cfg["experiment"])
        data = DataConfig(**cfg["data"])
        model = ModelConfig(**cfg["model"])
        loss = LossConfig(**cfg["loss"])
        steps_cfg = TrainingStepsConfig(**cfg["training"]["total_steps"])
        training_cfg = dict(cfg["training"])
        training_cfg["total_steps"] = steps_cfg
        training = TrainingConfig(**training_cfg)
        eval_cfg = EvalConfig(**cfg["eval"])
        return cls(
            experiment=experiment,
            data=data,
            model=model,
            loss=loss,
            training=training,
            phases=cfg["phases"],
            eval=eval_cfg,
        )


def load_config(path: str | Path) -> RootConfig:
    """Load a configuration file into a :class:`RootConfig`.

    The function first tries to rely on :mod:`yaml` when available. When the
    dependency cannot be installed (offline execution), the file is interpreted
    as JSON which is also valid YAML.
    """

    text = Path(path).read_text(encoding="utf-8")
    if yaml is not None:
        cfg_dict = yaml.safe_load(text)
    else:
        cfg_dict = json.loads(text)
    return RootConfig.from_dict(cfg_dict)


def merge_overrides(base: MutableMapping[str, Any], overrides: Optional[Mapping[str, Any]] = None) -> MutableMapping[str, Any]:
    """Recursively merge override values into a configuration mapping."""

    if overrides is None:
        return base
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            merge_overrides(base[key], value)  # type: ignore[index]
        else:
            base[key] = value  # type: ignore[index]
    return base
