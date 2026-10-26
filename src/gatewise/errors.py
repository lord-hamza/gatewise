class GatewayError(Exception):
    pass


class ProviderError(GatewayError):
    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after = retry_after


class ProvidersExhaustedError(GatewayError):
    def __init__(self, errors: list[ProviderError]) -> None:
        details = "; ".join(str(error) for error in errors)
        super().__init__(f"all providers failed ({details})")
        self.errors = errors


class BudgetExceededError(GatewayError):
    def __init__(self, tenant: str, *, used: int, requested: int, limit: int) -> None:
        super().__init__(
            f"token budget exceeded for tenant {tenant!r}: "
            f"{used} used + {requested} requested > {limit}"
        )
        self.tenant = tenant
        self.used = used
        self.requested = requested
        self.limit = limit


class PromptInjectionError(GatewayError):
    def __init__(self, score: float, signals: list[str]) -> None:
        super().__init__(f"possible prompt injection (score {score:.2f}: {', '.join(signals)})")
        self.score = score
        self.signals = signals


class StructuredOutputError(GatewayError):
    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw
