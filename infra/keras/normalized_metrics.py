import tensorflow as tf
import tensorflow.keras.backend as k_backend

from tensorflow import Tensor
from tensorflow import math as tf_math

MIN_DEPTH = tf.constant(1e-3)
MAX_DEPTH = tf.constant(80.00)


def normalize_min(value: Tensor) -> Tensor:
    return tf.where(tf.less(value, tf.constant(0.0)), MIN_DEPTH, value)


def normalize_max(value: Tensor) -> Tensor:
    return tf.where(tf.greater(value, MAX_DEPTH), MAX_DEPTH, value)


def normalize_tensor(value: Tensor) -> Tensor:
    return normalize_max(normalize_min(value))


def log10(value: Tensor) -> Tensor:
    log_numerator = tf_math.log(value)
    log_denominator = tf_math.log(tf.constant(10, dtype=log_numerator.dtype))
    return log_numerator / log_denominator


def build_threshold(delta: int = 1):
    def threshold(ground_truth: Tensor, predicted: Tensor):
        normalized_prediction = normalize_tensor(predicted)
        thresh = tf_math.maximum(
            (ground_truth / normalized_prediction),
            (normalized_prediction / ground_truth)
        )
        return k_backend.mean(thresh < 1.25 ** delta)

    return threshold


def threshold_1(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    return build_threshold(1)(ground_truth, predicted)


def threshold_2(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    return build_threshold(2)(ground_truth, predicted)


def threshold_3(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    return build_threshold(3)(ground_truth, predicted)


def abs_rel(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    normalized_prediction = normalize_tensor(predicted)
    abs_diff = tf_math.abs(ground_truth - normalized_prediction)
    return tf_math.reduce_mean(abs_diff / ground_truth)


def sq_rel(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    normalized_prediction = normalize_tensor(predicted)
    squared_diff = tf_math.squared_difference(ground_truth, normalized_prediction)
    return tf_math.reduce_mean(squared_diff / ground_truth)


def rmse(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    normalized_prediction = normalize_tensor(predicted)
    squared_error = tf_math.squared_difference(ground_truth, normalized_prediction)
    return tf_math.sqrt(tf_math.reduce_mean(squared_error))


def rmse_log(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    normalized_prediction = normalize_tensor(predicted)

    gt_log = tf_math.log(ground_truth)
    predicted_log = tf_math.log(normalized_prediction)

    squared_log_error = tf_math.squared_difference(gt_log, predicted_log)
    return tf_math.sqrt(tf_math.reduce_mean(squared_log_error))


def log_10(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    normalized_prediction = normalize_tensor(predicted)
    diff_log_10 = log10(ground_truth) - log10(normalized_prediction)
    return tf_math.reduce_mean(tf_math.abs(diff_log_10))
