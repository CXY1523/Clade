import math
import time

import pytest

from app.security.deadline import DeadlineBudget, DeadlineExpired


class ManualClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_deadline_budget_uses_original_start_and_clamps_each_phase() -> None:
    clock = ManualClock()
    budget = DeadlineBudget.from_timeout(60, started_at=90.0, clock=clock)
    assert budget.remaining() == pytest.approx(50.0)
    assert budget.phase_timeout(10.0) == pytest.approx(10.0)
    clock.now = 145.0
    assert budget.phase_timeout(10.0) == pytest.approx(5.0)
    clock.now = 150.0
    with pytest.raises(DeadlineExpired):
        budget.phase_timeout()


@pytest.mark.parametrize("value", [True, 0, -1, math.inf, math.nan, "60"])
def test_deadline_budget_rejects_invalid_timeout(value: object) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        DeadlineBudget.from_timeout(value)  # type: ignore[arg-type]


def test_wall_clock_changes_do_not_affect_monotonic_budget(monkeypatch) -> None:
    clock = ManualClock()
    budget = DeadlineBudget.from_timeout(10.0, clock=clock)
    monkeypatch.setattr(time, "time", lambda: -9_999_999.0)
    clock.now = 104.0
    assert budget.remaining() == pytest.approx(6.0)
