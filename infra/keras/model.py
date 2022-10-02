from typing import Optional, Tuple

from keras_unet_collection import models
from tensorflow import keras

from domain.models.network import Networks
from domain.models.test_case import TestCase


# Instanciar modelo com base no caso de teste
def build_model(test_case: TestCase, _input_shape: Tuple[int, int, int]) -> Optional[keras.Model]:
    backbone = (test_case['backbone']).value
    network = test_case['network']
    use_imagenet_weights = test_case['use_imagenet_weights']

    n_labels = 1
    filter_num = [64, 128, 256, 512, 1024]
    activation = 'ReLU'
    out_activation = 'Sigmoid'
    weights = 'imagenet' if use_imagenet_weights else None
    pool = False
    unpool = True
    batch_norm = True

    if network == Networks.UNet:
        return models.unet_2d(
            _input_shape, filter_num=filter_num, n_labels=n_labels, stack_num_down=2,
            stack_num_up=2, activation=activation, output_activation=out_activation,
            batch_norm=batch_norm, pool=pool, unpool=unpool, backbone=backbone,
            weights=weights, freeze_backbone=True, freeze_batch_norm=True,
        )

    if network == Networks.AttentionUNet:
        return models.att_unet_2d(
            _input_shape, filter_num=filter_num, n_labels=n_labels, stack_num_down=2,
            stack_num_up=2, activation=activation, atten_activation='ReLU', attention='add',
            output_activation=out_activation, batch_norm=batch_norm, pool=pool,
            unpool=unpool, backbone=backbone, weights=weights, freeze_backbone=True,
            freeze_batch_norm=True, name='attunet'
        )
    #
    # if network == Networks.SwinUNet:
    #     return models.swin_unet_2d(
    #         input_size=_input_shape, filter_num_begin, n_labels=n_labels, depth=4,
    #         stack_num_down=2, stack_num_up=2, patch_size, num_heads,
    #         window_size, num_mlp, output_activation=out_activation, shift_window=True,
    #         name='swin_unet'
    #     )

    if network == Networks.TransUNet:
        return models.transunet_2d(
            input_size=_input_shape, filter_num=filter_num, n_labels=n_labels,
            stack_num_down=2, stack_num_up=2, embed_dim=768, num_mlp=3072,
            num_heads=12, num_transformer=12, activation=activation, mlp_activation='GELU',
            output_activation=out_activation, batch_norm=batch_norm, pool=pool,
            unpool=unpool, backbone=backbone, weights=weights, freeze_backbone=True,
            freeze_batch_norm=True, name='transunet'
        )

    raise ValueError(f"Invalid network '{network}'")
