"""Simple neuro-symbolic reasoner operating on graph states."""
from __future__ import annotations

from typing import Dict, List

from ..math_utils import Vector
from .graph import GraphState


class Reasoner:
    """Executes differentiable-style operators on slots."""

    def __init__(self, operator_tokens: List[str]) -> None:
        self.operator_tokens = operator_tokens

    def _select(self, graph_state: GraphState) -> List[Vector]:
        existence = graph_state.slots.existence
        if not existence:
            return []
        threshold = sum(existence) / len(existence)
        selected = [slot for slot, score in zip(graph_state.slots.vectors, existence) if score >= threshold]
        return selected

    def _relate(self, graph_state: GraphState) -> List[Vector]:
        adjacency = graph_state.adjacency
        weighted: List[Vector] = []
        for idx, slot in enumerate(graph_state.slots.vectors):
            score = sum(adjacency[idx]) / len(adjacency[idx]) if adjacency[idx] else 0.0
            weighted.append([value * score for value in slot])
        return weighted

    def _apply_rule(self, vectors: List[Vector]) -> List[Vector]:
        if not vectors:
            return []
        dim = len(vectors[0])
        mean_vec = [0.0 for _ in range(dim)]
        for vec in vectors:
            for idx, value in enumerate(vec):
                mean_vec[idx] += value / len(vectors)
        centered: List[Vector] = []
        for vec in vectors:
            centered.append([value - mean_vec[idx] for idx, value in enumerate(vec)])
        return centered

    def run(self, graph_state: GraphState) -> Dict[str, List[Vector]]:
        """Execute the configured operator chain and return intermediates."""

        trace: Dict[str, List[Vector]] = {}
        current: List[Vector] = graph_state.slots.vectors
        for token in self.operator_tokens:
            if token == "SELECT":
                current = self._select(graph_state)
            elif token == "RELATE":
                current = self._relate(graph_state)
            elif token == "APPLY_RULE":
                current = self._apply_rule(current)
            trace[token] = current
        return trace
