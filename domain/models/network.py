from enum import Enum
from typing_extensions import TypedDict


class Networks(Enum):
    AttentionUNet = 'attention-unet'
    SwinUNet = 'swin-unet'
    TransUNet = 'trans-unet'
    UNet = 'unet'


class KerasBackbone(Enum):
    """
    See all supported: https://github.com/yingkaisha/keras-unet-collection/blob/main/keras_unet_collection/_model_unet_2d.py#L262
    """
    VGG16 = 'VGG16'
    VGG19 = 'VGG19'
    ResNet50 = 'ResNet50'
    ResNet101 = 'ResNet101'


class Optimizers(Enum):
    Adam = 'adam'
    RMSProp = 'rmsprop'
    SGD = 'sgd'


class NetworkConfig(TypedDict):
    size: int
    filter_min: int
    filter_max: int
    pool: bool
    unpool: bool
