from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

Clock = Callable[[], float]


class DeadlineExpired(TimeoutError):
    """The caller-owned monotonic budget has no time remaining."""


class TimeoutBudget(Protocol):
    def remaining(self) -> float:
        raise NotImplementedError

    def phase_timeout(self, limit: float | None = None) -> float:
        raise NotImplementedError


def _positive_finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeout must be a positive finite number")
    normalized = float(value)
    if normalized <= 0 or not math.isfinite(normalized):
        raise ValueError("timeout must be a positive finite number")
    return normalized


@dataclass(frozen=True)
class DeadlineBudget:
    deadline: float
    clock: Clock = field(default=time.monotonic, compare=False, repr=False)

    @classmethod
    def from_timeout(
        cls,
        timeout: float,
        *,
        started_at: float | None = None,
        clock: Clock = time.monotonic,
    ) -> "DeadlineBudget":
        normalized = _positive_finite(timeout)
        start = clock() if started_at is None else started_at
        return cls(deadline=start + normalized, clock=clock)

    def remaining(self) -> float:
        return max(0.0, self.deadline - self.clock())

    def phase_timeout(self, limit: float | None = None) -> float:
        remaining = self.remaining()
        if remaining <= 0:
            raise DeadlineExpired
        return remaining if limit is None else min(remaining, limit)
