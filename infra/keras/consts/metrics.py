import infra.keras.metrics as tf_metrics
from infra.keras.loss import depth_loss

all_metrics = [
    tf_metrics.abs_rel,
    tf_metrics.sq_rel,
    tf_metrics.rmse,
    tf_metrics.rmse_log,
    tf_metrics.log_10,
    tf_metrics.threshold_1,
    tf_metrics.threshold_2,
    tf_metrics.threshold_3,
]

metrics_custom_object = {
    'depth_loss': depth_loss,
    'abs_rel': tf_metrics.abs_rel,
    'sq_rel': tf_metrics.sq_rel,
    'rmse': tf_metrics.rmse,
    'rmse_log': tf_metrics.rmse_log,
    'log_10': tf_metrics.log_10,
    'threshold_1': tf_metrics.threshold_1,
    'threshold_2': tf_metrics.threshold_2,
    'threshold_3': tf_metrics.threshold_3,
}
