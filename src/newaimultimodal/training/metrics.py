"""Metric helpers consumed by the trainer."""

from __future__ import annotations

from typing import Callable, Dict, List

from ..data.sources import DataSample
from ..math_utils import norm
from ..models.graph import GraphState


def intervention_consistency(graph_state: GraphState, predicted_slots: List[List[float]]) -> float:
    """Measure how close predicted slots remain to the current graph after interventions."""

    baseline = graph_state.slots.vectors
    total_diff = 0.0
    total_norm = 1e-6
    for base, pred in zip(baseline, predicted_slots):
        total_diff += norm([b - p for b, p in zip(base, pred)])
        total_norm += norm(base)
    score = max(0.0, 1.0 - total_diff / total_norm)
    return float(score)


def slot_interpretability(graph_state: GraphState, sample: DataSample) -> float:
    """Proxy score indicating how many slots are confidently used to describe a scene."""

    existence = graph_state.slots.existence
    coverage = sum(existence) / (len(existence) or 1)
    text_len = len(sample.text.split()) + 1e-6
    return float(min(1.0, coverage / text_len * 3.0))


def grounding_accuracy(graph_state: GraphState, sample: DataSample) -> float:
    """Compare slot magnitudes with the positions present in the textual scene."""

    if not sample.vision:
        return 0.0
    positions = [sum(obj.position) for obj in sample.vision]
    slot_norms = [norm(slot) for slot in graph_state.slots.vectors]
    if not slot_norms:
        return 0.0
    diff = abs(sum(slot_norms) / len(slot_norms) - sum(positions) / len(positions))
    return float(1.0 / (1.0 + diff))


METRIC_REGISTRY: Dict[str, Callable[..., float]] = {
    "intervention_consistency": intervention_consistency,
    "slot_interpretability": slot_interpretability,
    "grounding_accuracy": grounding_accuracy,
}


def compute_metrics(
    graph_state: GraphState,
    predicted_slots: List[List[float]],
    sample: DataSample,
    metric_names: List[str],
) -> Dict[str, float]:
    """Evaluate the requested metrics and return them in a dictionary."""

    results: Dict[str, float] = {}
    for name in metric_names:
        fn = METRIC_REGISTRY.get(name)
        if fn is None:
            continue
        if name == "intervention_consistency":
            results[name] = fn(graph_state, predicted_slots)
        else:
            results[name] = fn(graph_state, sample)
    return results
