from typing import Tuple

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


def remove_zero(value: Tensor) -> Tensor:
    return tf.boolean_mask(value, tf.greater(value, 0.0))


def normalize_tensor(value: Tensor) -> Tensor:
    return normalize_max(normalize_min(value))


def normalize_tensors(ground_truth: Tensor, prediction: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Convert ground truth and prediction tensors to ndarray pair
    :param ground_truth: GT tensor
    :param prediction: Prediction tensor
    :return: Tuple with ground truth, prediction ndarrays
    """
    return normalize_tensor(ground_truth), normalize_tensor(prediction)


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
    tf.print(gt.shape)
    print('GT\n\n\nPREDICT')
    tf.print(predict.shape)

    print(f'\n\n\nGT: {type(gt)}\nPREDICT: {type(predict)}\n\n\n')
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
