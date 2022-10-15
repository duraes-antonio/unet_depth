from typing import Tuple

import numpy as np
import tensorflow as tf
from numpy import ndarray
from tensorflow import Tensor

MIN_DEPTH = 1e-3
MAX_DEPTH = 80


def normalize_tensor(value: ndarray) -> ndarray:
    clone = value.copy()
    clone[clone < MIN_DEPTH] = MIN_DEPTH
    clone[clone > MAX_DEPTH] = MAX_DEPTH
    return clone


def tensors_to_array(
        ground_truth: Tensor, prediction: Tensor
) -> Tuple[ndarray, ndarray]:
    """
    Convert ground truth and prediction tensors to ndarray pair
    :param ground_truth: GT tensor
    :param prediction: Prediction tensor
    :return: Tuple with ground truth, prediction ndarrays
    """
    prediction_np = tf.get_static_value(prediction)
    gt_np = tf.get_static_value(ground_truth)
    return gt_np, prediction_np


def build_threshold(delta: int = 1):
    def threshold(ground_truth: Tensor, predicted: Tensor) -> ndarray:
        gt, prediction = tensors_to_array(ground_truth, predicted)
        prediction = normalize_tensor(prediction)
        thresh: ndarray = np.maximum((gt / prediction), (prediction / gt))
        return (thresh < 1.25 ** delta).mean()

    return threshold


def threshold_1(ground_truth: Tensor, predicted: Tensor) -> ndarray:
    return build_threshold(1)(ground_truth, predicted)


def threshold_2(ground_truth: Tensor, predicted: Tensor) -> ndarray:
    return build_threshold(2)(ground_truth, predicted)


def threshold_3(ground_truth: Tensor, predicted: Tensor) -> ndarray:
    return build_threshold(3)(ground_truth, predicted)


def rmse(ground_truth: Tensor, predicted: Tensor) -> ndarray:
    gt, prediction = tensors_to_array(ground_truth, predicted)
    prediction = normalize_tensor(prediction)
    squared_diff: ndarray = (gt - prediction) ** 2
    return np.sqrt(squared_diff.mean())


def rmse_log(ground_truth: Tensor, predicted: Tensor) -> Tensor:
    gt, prediction = tensors_to_array(ground_truth, predicted)
    prediction = normalize_tensor(prediction)
    squared_diff_log: ndarray = (np.log(gt) - np.log(prediction)) ** 2
    return np.sqrt(squared_diff_log.mean())


def abs_rel(ground_truth: Tensor, predicted: Tensor) -> ndarray:
    gt, prediction = tensors_to_array(ground_truth, predicted)
    prediction = normalize_tensor(prediction)
    return np.mean(np.abs(gt - prediction) / gt)


def sq_rel(ground_truth: Tensor, predicted: Tensor) -> ndarray:
    gt, prediction = tensors_to_array(ground_truth, predicted)
    prediction = normalize_tensor(prediction)
    return np.mean(((gt - prediction) ** 2) / gt)


def log_10(ground_truth: Tensor, predicted: Tensor) -> ndarray:
    gt, prediction = tensors_to_array(ground_truth, predicted)
    prediction = normalize_tensor(prediction)
    return (np.abs(np.log10(gt) - np.log10(prediction))).mean()
