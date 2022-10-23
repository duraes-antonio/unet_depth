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
