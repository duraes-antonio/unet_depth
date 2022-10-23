from datetime import datetime
from typing import Union

from bson import ObjectId

from domain.models.named_entity import Entity
from domain.models.train_result import TrainResult


class TestCaseExecutionHistory(Entity):
    epoch: int
    test_case_id: Union[ObjectId, str]
    created_at: datetime
    model_name: str
    model_id: str
    gpu_description: str
    cpu_description: str
    result: TrainResult
