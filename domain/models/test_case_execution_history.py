from datetime import datetime

from bson import ObjectId

from domain.models.named_entity import Entity


class TestCaseExecutionHistory(Entity):
    last_epoch: int
    test_case_id: ObjectId
    created_at: datetime
    model_name: str
    model_id: str
    gpu_description: str
    cpu_description: str
