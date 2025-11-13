"""High level training loop."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

from ..config import RootConfig
from ..data import MixedDataLoader
from ..data.buffer import BufferItem, TrajectoryBuffer
from ..data.sources import JSONLinesSource, SyntheticCLEVRSampler
from ..math_utils import mean, norm
from ..models import (
    DynamicsModel,
    GraphBuilder,
    InterventionController,
    PerceptionEncoder,
    Reasoner,
    SlotAttentionModule,
    SurfaceRealizer,
)
from .metrics import compute_metrics

LOGGER = logging.getLogger(__name__)


class Trainer:
    """Coordinates data, models and metrics for each training phase."""

    def __init__(self, config: RootConfig) -> None:
        self.config = config
        self.random = random.Random(config.experiment.seed)
        self.output_dir = Path(config.experiment.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._build_components()
        self.buffer = TrajectoryBuffer(
            capacity=int(config.data.trajectory_buffer.get("size", 1024)),
            prioritize_interventions=config.data.trajectory_buffer.get("prioritize_interventions", False),
        )

    def _build_components(self) -> None:
        model_cfg = self.config.model
        slot_dim = model_cfg.slots.get("slot_dim", 128)
        self.perception = PerceptionEncoder(model_cfg.perception, slot_dim=slot_dim, seed=self.config.experiment.seed)
        self.slots = SlotAttentionModule(
            num_slots=model_cfg.slots.get("num_slots", 16),
            slot_dim=slot_dim,
            existence_bias=model_cfg.slots.get("existence_bias", -1.0),
        )
        self.graph_builder = GraphBuilder(hidden_dim=model_cfg.graph.get("hidden_dim", slot_dim))
        self.dynamics = DynamicsModel(
            slot_dim=slot_dim,
            action_dim=model_cfg.dynamics.get("action_dim", 32),
            intervention_heads=model_cfg.dynamics.get("intervention_heads", True),
        )
        self.reasoner = Reasoner(model_cfg.reasoner.get("operator_tokens", []))
        self.decoder = SurfaceRealizer(model_cfg.decoder.get("hidden_dim", 512))
        self.controller = InterventionController(seed=self.config.experiment.seed + 1)

    def _build_dataloader(self) -> MixedDataLoader:
        data_cfg = self.config.data
        real = None
        synth = None
        real_sources = data_cfg.sources.get("real")
        if real_sources:
            real_path = Path(real_sources[0].get("path", ""))
            real = JSONLinesSource(real_path, seed=self.config.experiment.seed)
        synthetic_sources = data_cfg.sources.get("synthetic")
        if synthetic_sources:
            synth_cfg = synthetic_sources[0]
            synth = SyntheticCLEVRSampler(
                num_scenes=int(synth_cfg.get("num_scenes", 512)),
                seed=self.config.experiment.seed + 42,
            )
        return MixedDataLoader(
            real_source=real,
            synthetic_source=synth,
            batch_size=self.config.training.batch_size,
            real_ratio=self.config.data.batch_mix.get("real_ratio", 0.5),
            rng=self.random,
        )

    def _forward(self, sample_batch: List) -> Dict[str, float]:
        losses = {"rec": 0.0, "slot": 0.0, "dyn": 0.0, "text": 0.0}
        metrics_accum: Dict[str, float] = {metric: 0.0 for metric in self.config.eval.metrics}
        for sample in sample_batch:
            features = self.perception.encode(sample)
            slot_set = self.slots.infer(features)
            graph_state = self.graph_builder.build(slot_set)
            predicted = self.dynamics.predict(graph_state, sample.action, sample.intervention)
            reasoner_trace = self.reasoner.run(graph_state)
            decoded = self.decoder.generate(graph_state, reasoner_trace)

            feature_mean = mean(features)
            slot_mean = mean(slot_set.vectors)
            rec_loss = norm([f - s for f, s in zip(feature_mean, slot_mean)])
            slot_loss = 1.0 - (sum(slot_set.existence) / (len(slot_set.existence) or 1))
            dyn_loss = 0.0
            for pred_slot, true_slot in zip(predicted, slot_set.vectors):
                dyn_loss += norm([p - t for p, t in zip(pred_slot, true_slot)])
            dyn_loss /= max(1, len(predicted))
            text_loss = float(abs(len(decoded) - len(sample.target_text))) / (len(sample.target_text) + 1e-6)
            losses["rec"] += rec_loss
            losses["slot"] += slot_loss
            losses["dyn"] += dyn_loss
            losses["text"] += text_loss

            metrics = compute_metrics(graph_state, predicted, sample, self.config.eval.metrics)
            for key, value in metrics.items():
                metrics_accum[key] += value

            self.buffer.add(
                BufferItem(
                    current=sample,
                    action=sample.action,
                    next_state=sample,
                    intervention=sample.intervention,
                )
            )
        batch_size = len(sample_batch)
        for key in losses:
            losses[key] /= batch_size
        for key in metrics_accum:
            metrics_accum[key] /= max(1, batch_size)
        losses.update(metrics_accum)
        return losses

    def run_phase(self, phase: str, max_steps: Optional[int] = None) -> None:
        LOGGER.info("Starting phase %s", phase)
        dataloader = self._build_dataloader()
        steps = max_steps or self.config.training.total_steps.pretrain
        for step, batch in zip(range(steps), dataloader.batches()):
            stats = self._forward(batch)
            if (step + 1) % self.config.training.evaluation_interval == 0 or step == 0:
                msg = ", ".join(f"{k}={v:.3f}" for k, v in stats.items())
                LOGGER.info("[phase=%s][step=%d] %s", phase, step + 1, msg)
        LOGGER.info("Finished phase %s", phase)
