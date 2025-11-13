"""Simple replay buffer used during training phases."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Iterator, List, Optional

from .sources import DataSample


@dataclass
class BufferItem:
    """Represents a transition stored in the replay buffer."""

    current: DataSample
    action: str
    next_state: DataSample
    intervention: Optional[str]


class TrajectoryBuffer:
    """A small replay buffer prioritising intervention-rich samples."""

    def __init__(self, capacity: int, prioritize_interventions: bool = False) -> None:
        self.capacity = capacity
        self.prioritize_interventions = prioritize_interventions
        self._buffer: Deque[BufferItem] = deque(maxlen=capacity)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._buffer)

    def add(self, item: BufferItem) -> None:
        """Insert a new transition into the buffer."""

        self._buffer.append(item)

    def extend(self, items: Iterable[BufferItem]) -> None:
        for item in items:
            self.add(item)

    def sample(self, batch_size: int) -> List[BufferItem]:
        """Return a list of samples, optionally emphasising interventions."""

        if not self._buffer:
            raise RuntimeError("Buffer is empty")

        if not self.prioritize_interventions:
            indices = list(range(len(self._buffer)))
        else:
            indices = sorted(
                range(len(self._buffer)),
                key=lambda idx: 0 if self._buffer[idx].intervention is None else -1,
            )
        selected = indices[:batch_size]
        return [self._buffer[i] for i in selected]

    def __iter__(self) -> Iterator[BufferItem]:  # pragma: no cover - trivial
        return iter(self._buffer)
