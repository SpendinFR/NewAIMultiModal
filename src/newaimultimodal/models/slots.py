"""Slot extraction utilities."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List

from ..math_utils import Vector, add, norm, scale, softmax


@dataclass
class SlotSet:
    """Container that stores discovered concepts."""

    vectors: List[Vector]
    existence: List[float]


class SlotAttentionModule:
    """A lightweight and differentiable-inspired slot attention layer."""

    def __init__(self, num_slots: int, slot_dim: int, num_iters: int = 3, existence_bias: float = -1.0, seed: int = 0) -> None:
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iters = num_iters
        self.existence_bias = existence_bias
        rng = random.Random(seed)
        self.slots = [
            [rng.gauss(0.0, 1.0) for _ in range(slot_dim)]
            for _ in range(num_slots)
        ]
        self.scale = 1.0 / math.sqrt(slot_dim)

    def infer(self, features: List[Vector]) -> SlotSet:
        """Apply iterative attention to produce slots."""

        if not features:
            return SlotSet(vectors=[[0.0] * self.slot_dim for _ in range(self.num_slots)], existence=[0.0] * self.num_slots)

        slots = [vec[:] for vec in self.slots]
        for _ in range(self.num_iters):
            attn_weights: List[List[float]] = []
            for feature in features:
                logits = [self.scale * sum(f * s for f, s in zip(feature, slot)) for slot in slots]
                attn_weights.append(softmax(logits))
            new_slots: List[Vector] = []
            for slot_idx in range(self.num_slots):
                numerator = [0.0] * self.slot_dim
                denom = 1e-6
                for feature, weights in zip(features, attn_weights):
                    weight = weights[slot_idx]
                    denom += weight
                    numerator = add(numerator, [weight * value for value in feature])
                slot_update = [value / denom for value in numerator]
                updated = [0.5 * old + 0.5 * new for old, new in zip(slots[slot_idx], slot_update)]
                new_slots.append(updated)
            slots = new_slots

        existence = [1.0 / (1.0 + math.exp(-(norm(slot) + self.existence_bias))) for slot in slots]
        return SlotSet(vectors=slots, existence=existence)
