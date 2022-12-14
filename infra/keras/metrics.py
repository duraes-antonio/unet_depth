from typing import Tuple

import tensorflow as tf
import tensorflow.keras.backend as k_backend
from tensorflow import Tensor
from tensorflow import math as tf_math

MIN_DEPTH = tf.constant(1e-3)
MAX_DEPTH = tf.constant(80.00)


def build_poly_decay(n_epochs: int, init_learning_rate: float):
    def poly_decay(epoch: int):
        max_epochs = n_epochs
        base_lr = init_learning_rate
        power = 1.0
        return base_lr * (1 - (epoch / float(max_epochs))) ** power

    return poly_decay


def normalize_min(value: Tensor) -> Tensor:
    return tf.where(tf.less(value, MIN_DEPTH), MIN_DEPTH, value)


def normalize_max(value: Tensor) -> Tensor:
    return tf.where(tf.greater(value, MAX_DEPTH), MAX_DEPTH, value)


def normalize_tensor(value: Tensor) -> Tensor:
    return normalize_max(normalize_min(value))


def normalize_tensors(ground_truth: Tensor, prediction: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Convert ground truth and prediction tensors to ndarray pair
    :param ground_truth: GT tensor
    :param prediction: Prediction tensor
    :return: Tuple with ground truth, prediction ndarrays
    """
    gt_mask = tf.greater(ground_truth, 0.0)
    prediction_masked = tf.boolean_mask(prediction, gt_mask)
    gt_masked = tf.boolean_mask(ground_truth, gt_mask)
    return gt_masked, normalize_tensor(prediction_masked)


def log10(value: Tensor) -> Tensor:
    log_numerator = tf_math.log(value)
    log_denominator = tf_math.log(tf.constant(10, dtype=log_numerator.dtype))
    return log_numerator / log_denominator


def build_threshold(delta: int = 1):
    def threshold(ground_truth: Tensor, predicted: Tensor):
        gt, predict = normalize_tensors(ground_truth, predicted)
        thresh = tf_math.maximum(
            (gt / predict),
            (predict / gt)
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
    gt, predict = normalize_tensors(ground_truth, predicted)
    abs_diff = tf_math.abs(gt - predict)
    return tf_math.reduce_mean(abs_diff / gt)


def sq_rel(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    gt, predict = normalize_tensors(ground_truth, predicted)
    squared_diff = tf_math.squared_difference(gt, predict)
    return tf_math.reduce_mean(squared_diff / gt)


def rmse(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    gt, predict = normalize_tensors(ground_truth, predicted)
    squared_error = tf_math.squared_difference(gt, predict)
    return tf_math.sqrt(tf_math.reduce_mean(squared_error))


def rmse_log(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    gt, predict = normalize_tensors(ground_truth, predicted)

    gt_log = tf_math.log(gt)
    predicted_log = tf_math.log(predict)

    squared_log_error = tf_math.squared_difference(gt_log, predicted_log)
    return tf_math.sqrt(tf_math.reduce_mean(squared_log_error))


def log_10(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    gt, predict = normalize_tensors(ground_truth, predicted)
    diff_log_10 = log10(gt) - log10(predict)
    return tf_math.reduce_mean(tf_math.abs(diff_log_10))
