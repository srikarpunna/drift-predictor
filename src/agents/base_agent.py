from abc import ABC, abstractmethod

from src.models import MigrationResult


class BaseAgent(ABC):
    @abstractmethod
    def act(self, migration_result: MigrationResult) -> str:
        """Returns diagnosis or repair action."""
        ...
