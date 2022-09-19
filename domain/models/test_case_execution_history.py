from datetime import datetime

from bson import ObjectId

from domain.models.named_entity import Entity


# name: 'attention-unet_epoch-15_adam_resnet-101_imagenet-0'
class TestCaseExecutionHistory(Entity):
    start_epoch: int
    last_epoch: int
    test_case_id: ObjectId
    created_at: datetime
    model_name: str
    model_id: str
    gpu_description: str
    cpu_description: str
