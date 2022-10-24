from typing import List

from typing_extensions import TypedDict


class TrainResult(TypedDict):
    loss: float
    abs_rel: float
    sq_rel: float
    rmse: float
    rmse_log: float
    log_10: float
    threshold_1: float
    threshold_2: float
    threshold_3: float

    val_loss: float
    val_abs_rel: float
    val_sq_rel: float
    val_rmse: float
    val_rmse_log: float
    val_log_10: float
    val_threshold_1: float
    val_threshold_2: float
    val_threshold_3: float


class FinalTrainResult(TypedDict):
    loss: List[float]
    abs_rel: List[float]
    sq_rel: List[float]
    rmse: List[float]
    rmse_log: List[float]
    log_10: List[float]
    threshold_1: List[float]
    threshold_2: List[float]
    threshold_3: List[float]

    val_loss: List[float]
    val_abs_rel: List[float]
    val_sq_rel: List[float]
    val_rmse: List[float]
    val_rmse_log: List[float]
    val_log_10: List[float]
    val_threshold_1: List[float]
    val_threshold_2: List[float]
    val_threshold_3: List[float]
