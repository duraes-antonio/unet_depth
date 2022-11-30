from datetime import datetime
from enum import Enum
from typing import Optional

from typing_extensions import TypedDict

from domain.models.named_entity import Entity
from domain.models.network import Networks, KerasBackbone, Optimizers


class TestCaseState(Enum):
    Available = 'available'
    Busy = 'busy'
    Done = 'done'


class InputReadMode(Enum):
    BGR2GRAY = 'bgr2gray'
    ANY_DEPTH = 'anydepth'


class TestCaseConfig(TypedDict):
    network: Networks
    backbone: KerasBackbone
    optimizer: Optimizers
    use_imagenet_weights: bool
    size: int
    filter_min: int
    filter_max: int
    read_mode: InputReadMode


class TestCase(Entity):
    config: TestCaseConfig
    state: TestCaseState
    created_at: datetime
    last_modified: Optional[datetime]
