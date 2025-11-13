"""Surface realization utilities."""
from __future__ import annotations

from typing import Dict, List

from ..math_utils import Vector
from .graph import GraphState


class SurfaceRealizer:
    """Generates lightweight textual descriptions from slot states."""

    def __init__(self, hidden_dim: int, seed: int = 0) -> None:
        self.hidden_dim = hidden_dim

    def _summarize_slot(self, slot_vector: Vector) -> str:
        rounded = tuple(round(value, 2) for value in slot_vector[:4])
        slot_hash = hash(rounded)
        color = ["red", "blue", "green", "yellow"][(slot_hash >> 2) % 4]
        action = ["moving", "stable", "rotating", "highlighted"][(slot_hash >> 5) % 4]
        return f"{color} concept is {action}"

    def generate(self, graph_state: GraphState, reasoner_trace: Dict[str, List[Vector]]) -> str:
        slots = graph_state.slots.vectors
        summaries = [self._summarize_slot(slot) for slot in slots[:3]]
        chain_info = ", ".join(reasoner_trace.keys())
        return f"Concepts: {'; '.join(summaries)}. Reasoning chain: {chain_info}."
