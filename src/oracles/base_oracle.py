from abc import ABC, abstractmethod

from src.models import RunResult


class BaseOracle(ABC):
    @abstractmethod
    def check(self, result: RunResult) -> tuple[bool, str | None]:
        """Returns (passed, failure_evidence_or_none)."""
        ...
