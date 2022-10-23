from abc import ABC, abstractmethod
from typing import Optional

from domain.models.test_case.test_case_execution_history import TestCaseExecutionHistory


class TestCaseExecutionService(ABC):

    @abstractmethod
    def get_last_execution(self, test_case_id: str) -> Optional[TestCaseExecutionHistory]:
        pass

    @abstractmethod
    def save(self, result: TestCaseExecutionHistory) -> TestCaseExecutionHistory:
        pass
