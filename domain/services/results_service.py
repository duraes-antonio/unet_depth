from abc import ABC, abstractmethod
from typing import List

from domain.models.train_result import TrainResult, FinalTrainResult


class ResultService(ABC):

    @abstractmethod
    def get_by_test(self, test_case_id: str) -> List[TrainResult]:
        pass

    @abstractmethod
    def get_and_unify(self, test_case_id: str) -> FinalTrainResult:
        pass
