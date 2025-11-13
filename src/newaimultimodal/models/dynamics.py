"""Simple dynamics model operating on slot vectors."""
from __future__ import annotations

import hashlib
import random
from typing import List, Optional

from ..math_utils import Vector
from .graph import GraphState


class DynamicsModel:
    """Predicts next slots given an action and optional intervention."""

    def __init__(self, slot_dim: int, action_dim: int, intervention_heads: bool = True, seed: int = 0) -> None:
        self.slot_dim = slot_dim
        self.action_dim = action_dim
        self.intervention_heads = intervention_heads
        rng = random.Random(seed)
        self.transition = [[rng.gauss(0.0, 1.0) for _ in range(slot_dim)] for _ in range(action_dim)]
        self.intervention_matrix = [[rng.gauss(0.0, 1.0) for _ in range(slot_dim)] for _ in range(slot_dim)]

    def _encode_action(self, action: str) -> List[float]:
        vec = [0.0 for _ in range(self.action_dim)]
        digest = hashlib.blake2s(action.encode("utf-8")).hexdigest()
        for idx in range(self.action_dim):
            vec[idx] = int(digest[idx % len(digest)], 16) / 15.0
        return vec

    def _matvec(self, matrix: List[List[float]], vector: List[float]) -> List[float]:
        return [sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix]

    def predict(self, graph_state: GraphState, action: str, intervention: Optional[str] = None) -> List[Vector]:
        action_vec = self._encode_action(action)
        delta = self._matvec(self.transition, action_vec)
        next_slots: List[Vector] = []
        for slot in graph_state.slots.vectors:
            next_slots.append([value + d for value, d in zip(slot, delta)])

        if intervention and self.intervention_heads and graph_state.slots.vectors:
            mean_slot = [sum(values) / len(values) for values in zip(*graph_state.slots.vectors)]
            adjustment = self._matvec(self.intervention_matrix, mean_slot)
            for idx, slot in enumerate(next_slots):
                next_slots[idx] = [value + adjustment[j] for j, value in enumerate(slot)]
        return next_slots
