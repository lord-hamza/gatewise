from unittest.mock import Mock

import pytest

from gatewise.budget import TokenBudget
from gatewise.errors import BudgetExceededError


def test_unlimited_by_default() -> None:
    budget = TokenBudget()
    budget.charge("acme", 10**9)
    budget.check("acme", 10**9)
    assert budget.limit("acme") is None
    assert budget.remaining("acme") is None


def test_explicit_and_default_limits() -> None:
    budget = TokenBudget({"acme": 100}, default=10)
    assert budget.limit("acme") == 100
    assert budget.limit("globex") == 10
    budget.set_limit("globex", 50)
    assert budget.limit("globex") == 50


def test_charge_tracks_used_and_remaining() -> None:
    budget = TokenBudget(default=100)
    budget.charge("acme", 30)
    budget.charge("acme", 20)
    assert budget.used("acme") == 50
    assert budget.remaining("acme") == 50
    assert budget.used("globex") == 0
    budget.charge("acme", 80)
    assert budget.remaining("acme") == 0


def test_check_raises_when_over_budget() -> None:
    budget = TokenBudget({"acme": 100})
    budget.charge("acme", 90)
    budget.check("acme", 10)
    with pytest.raises(BudgetExceededError) as info:
        budget.check("acme", 11)
    error = info.value
    assert (error.tenant, error.used, error.requested, error.limit) == ("acme", 90, 11, 100)


def test_window_resets_usage() -> None:
    clock = Mock(return_value=0.0)
    budget = TokenBudget(default=100, window=60, clock=clock)
    budget.charge("acme", 60)
    clock.return_value = 30.0
    budget.charge("acme", 40)
    clock.return_value = 59.0
    with pytest.raises(BudgetExceededError):
        budget.check("acme", 1)
    clock.return_value = 60.0
    budget.check("acme", 100)
    assert budget.used("acme") == 0


def test_reset_one_tenant_or_all() -> None:
    budget = TokenBudget()
    budget.charge("acme", 10)
    budget.charge("globex", 20)
    budget.reset("acme")
    assert (budget.used("acme"), budget.used("globex")) == (0, 20)
    budget.reset()
    assert budget.used("globex") == 0
