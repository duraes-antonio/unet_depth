import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as k_backend
from numpy import inf
from tensorflow import Tensor
from tensorflow import math as tf_math


def log10(value: Tensor):
    log_numerator = tf_math.log(value)
    log_denominator = tf_math.log(tf.constant(10, dtype=log_numerator.dtype))
    return log_numerator / log_denominator


def build_poly_decay(n_epochs: int, init_learning_rate: float):
    def poly_decay(epoch: int):
        max_epochs = n_epochs
        base_lr = init_learning_rate
        power = 1.0
        return base_lr * (1 - (epoch / float(max_epochs))) ** power

    return poly_decay


def depth_acc(ground_truth: Tensor, predicted: Tensor):
    return k_backend.mean(k_backend.equal(k_backend.round(ground_truth), k_backend.round(predicted)))


def build_threshold(delta: int = 1):
    def threshold(ground_truth: Tensor, predicted: Tensor):
        thresh = tf_math.maximum((ground_truth / predicted), (predicted / ground_truth))
        return k_backend.mean(thresh < 1.25 ** delta)

    return threshold


def threshold_1(ground_truth: Tensor, predicted: Tensor):
    return build_threshold(1)(ground_truth, predicted)


def threshold_2(ground_truth: Tensor, predicted: Tensor):
    return build_threshold(2)(ground_truth, predicted)


def threshold_3(ground_truth: Tensor, predicted: Tensor):
    return build_threshold(3)(ground_truth, predicted)


def abs_rel(ground_truth: Tensor, predicted: Tensor):
    gt_np = np.array(tf.get_static_value(ground_truth))
    pred_np = np.array(tf.get_static_value(ground_truth))

    if inf in gt_np:
        print('\nground_truth:', gt_np)

    if inf in pred_np:
        print('\npred_inf:', pred_np)

    abs_diff = tf_math.abs(ground_truth - predicted)

    div_np = np.array(tf.get_static_value(abs_diff / ground_truth))

    if inf in div_np:
        print('\nabs_diff / ground_truth:', abs_diff / ground_truth)

    return tf_math.reduce_mean(abs_diff / ground_truth)


def sq_rel(ground_truth: Tensor, predicted: Tensor):
    gt_np = np.array(tf.get_static_value(ground_truth))
    pred_np = np.array(tf.get_static_value(ground_truth))
    print()

    if inf in gt_np:
        print('\nground_truth:', gt_np)

    if inf in pred_np:
        print('\npred_inf:', pred_np)

    squared_diff = tf_math.squared_difference(ground_truth, predicted)

    sq_np = np.array(tf.get_static_value(squared_diff))

    if inf in sq_np:
        print('\nsq_np:', sq_np)

    return tf_math.reduce_mean(squared_diff / ground_truth)


def rmse(ground_truth: Tensor, predicted: Tensor):
    squared_error: Tensor = tf_math.squared_difference(ground_truth, predicted)
    return tf_math.sqrt(tf_math.reduce_mean(squared_error))


def rmse_log(ground_truth: Tensor, predicted: Tensor):
    gt_normalized = tf.clip_by_value(ground_truth, clip_value_min=1e-10, clip_value_max=1e10)
    predicted_normalized = tf.clip_by_value(predicted, clip_value_min=1e-10, clip_value_max=1e10)

    gt_log = tf_math.log(gt_normalized)
    predicted_log = tf_math.log(predicted_normalized)

    squared_log_error = tf_math.squared_difference(gt_log, predicted_log)
    return tf_math.sqrt(tf_math.reduce_mean(squared_log_error))


def log_10(ground_truth: Tensor, predicted: Tensor):
    gt_normalized = tf.clip_by_value(ground_truth, clip_value_min=1e-10, clip_value_max=1e10)
    predicted_normalized = tf.clip_by_value(predicted, clip_value_min=1e-10, clip_value_max=1e10)

    diff_log_10 = log10(gt_normalized) - log10(predicted_normalized)
    return tf_math.reduce_mean(tf_math.abs(diff_log_10))
