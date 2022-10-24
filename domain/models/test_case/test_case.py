from datetime import datetime
from enum import Enum
from typing import Optional

from domain.models.named_entity import Entity
from domain.models.network import Networks, KerasBackbone, Optimizers


class TestCaseState(Enum):
    Available = 'available'
    Busy = 'busy'
    Done = 'done'


class TestCase(Entity):
    network: Networks
    backbone: KerasBackbone
    optimizer: Optimizers
    use_imagenet_weights: bool
    state: TestCaseState
    created_at: datetime
    last_modified: Optional[datetime]
