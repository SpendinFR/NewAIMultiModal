"""Graph builder for slot interactions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..math_utils import dot, norm
from .slots import SlotSet


@dataclass
class GraphState:
    """Structured representation of the world graph."""

    slots: SlotSet
    adjacency: List[List[float]]


class GraphBuilder:
    """Converts a slot set into a relational graph."""

    def __init__(self, hidden_dim: int, num_layers: int = 3) -> None:
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def build(self, slots: SlotSet) -> GraphState:
        vectors = slots.vectors
        if not vectors:
            return GraphState(slots=slots, adjacency=[])
        adjacency: List[List[float]] = []
        for a in vectors:
            row: List[float] = []
            for b in vectors:
                denom = (norm(a) * norm(b)) or 1e-6
                row.append(max(0.0, dot(a, b) / denom))
            total = sum(row) or 1.0
            row = [value / total for value in row]
            adjacency.append(row)
        for _ in range(self.num_layers - 1):
            new_adj: List[List[float]] = []
            for i in range(len(adjacency)):
                new_row: List[float] = []
                for j in range(len(adjacency)):
                    value = sum(adjacency[i][k] * adjacency[k][j] for k in range(len(adjacency)))
                    new_row.append(value)
                total = sum(new_row) or 1.0
                new_adj.append([value / total for value in new_row])
            adjacency = new_adj
        return GraphState(slots=slots, adjacency=adjacency)
