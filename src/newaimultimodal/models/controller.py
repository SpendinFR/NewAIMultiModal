"""Controller coordinating interventions."""
from __future__ import annotations

import random
from typing import Optional

from .graph import GraphState


class InterventionController:
    """Selects slots and actions for causal interventions."""

    def __init__(self, seed: int = 0) -> None:
        self.random = random.Random(seed)

    def propose(self, graph_state: GraphState) -> Optional[str]:
        """Return a textual description of a candidate intervention."""

        existence = list(graph_state.slots.existence)
        if not existence:
            return None
        best_idx = max(range(len(existence)), key=lambda idx: existence[idx])
        if existence[best_idx] < 0.2:
            return None
        action = self.random.choice(["boost", "suppress", "shuffle"])
        return f"do(slot_{best_idx}={action})"
