from typing import Union, List

from typing_extensions import Literal

VALIDATION_METRICS_NAME = Literal[
    'val_loss',
    'val_abs_rel',
    'val_sq_rel',
    'val_rmse',
    'val_rmse_log',
    'val_log_10',
    'val_threshold_1',
    'val_threshold_2',
    'val_threshold_3',
]
METRIC_NAME = Literal[
    'loss',
    'abs_rel',
    'sq_rel',
    'rmse',
    'rmse_log',
    'log_10',
    'threshold_1',
    'threshold_2',
    'threshold_3',
]
METRICS_NAMES: List[METRIC_NAME] = [
    'loss',
    'abs_rel',
    'sq_rel',
    'rmse',
    'rmse_log',
    'log_10',
    'threshold_1',
    'threshold_2',
    'threshold_3',
]
TRAIN_VAL_METRICS_NAME = Union[METRIC_NAME, VALIDATION_METRICS_NAME]
