import math
import time
from collections.abc import Callable, Mapping

from llm_tollgate.errors import BudgetExceededError


class TokenBudget:
    """Per-tenant token limits, optionally reset every `window` seconds."""

    def __init__(
        self,
        limits: Mapping[str, int] | None = None,
        *,
        default: int | None = None,
        window: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.default = default
        self.window = window
        self._limits = dict(limits or {})
        self._clock = clock
        self._used: dict[str, int] = {}
        self._resets_at: dict[str, float] = {}

    def limit(self, tenant: str) -> int | None:
        return self._limits.get(tenant, self.default)

    def set_limit(self, tenant: str, tokens: int) -> None:
        self._limits[tenant] = tokens

    def used(self, tenant: str) -> int:
        if self._clock() >= self._resets_at.get(tenant, math.inf):
            self.reset(tenant)
        return self._used.get(tenant, 0)

    def remaining(self, tenant: str) -> int | None:
        limit = self.limit(tenant)
        return None if limit is None else max(limit - self.used(tenant), 0)

    def check(self, tenant: str, tokens: int = 0) -> None:
        limit = self.limit(tenant)
        used = self.used(tenant)
        if limit is not None and used + tokens > limit:
            raise BudgetExceededError(tenant, used=used, requested=tokens, limit=limit)

    def charge(self, tenant: str, tokens: int) -> None:
        self._used[tenant] = self.used(tenant) + tokens
        if self.window is not None:
            self._resets_at.setdefault(tenant, self._clock() + self.window)

    def reset(self, tenant: str | None = None) -> None:
        if tenant is None:
            self._used.clear()
            self._resets_at.clear()
        else:
            self._used.pop(tenant, None)
            self._resets_at.pop(tenant, None)
