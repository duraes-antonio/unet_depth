import tensorflow
from tensorflow import Tensor
from tensorflow import math as tf_math


def log10(value: Tensor):
    log_numerator = tf_math.log(value)
    log_denominator = tf_math.log(tensorflow.constant(10, dtype=log_numerator.dtype))
    return log_numerator / log_denominator


def build_poly_decay(n_epochs: int, init_learning_rate: float):
    def poly_decay(epoch: int):
        max_epochs = n_epochs
        base_lr = init_learning_rate
        power = 1.0
        return base_lr * (1 - (epoch / float(max_epochs))) ** power

    return poly_decay


def depth_acc(ground_truth: Tensor, predicted: Tensor):
    return tf_math.reduce_mean(tf_math.equal(tf_math.round(ground_truth), tf_math.round(predicted)))


def build_threshold(delta: int = 1):
    def threshold(ground_truth: Tensor, predicted: Tensor):
        thresh = tf_math.maximum((ground_truth / predicted), (predicted / ground_truth))
        return tf_math.reduce_mean(thresh < 1.25 ** delta)

    return threshold


def threshold_1(ground_truth: Tensor, predicted: Tensor):
    return build_threshold(1)(ground_truth, predicted)


def threshold_2(ground_truth: Tensor, predicted: Tensor):
    return build_threshold(2)(ground_truth, predicted)


def threshold_3(ground_truth: Tensor, predicted: Tensor):
    return build_threshold(3)(ground_truth, predicted)


def abs_rel(ground_truth: Tensor, predicted: Tensor):
    gt_inf: Tensor[bool] = tf_math.is_inf(ground_truth)
    pred_inf: Tensor[bool] = tf_math.is_inf(predicted)

    if True in gt_inf:
        print('\nground_truth:', ground_truth)

    if True in pred_inf:
        print('\npred_inf:', pred_inf)

    abs_diff = tf_math.abs(ground_truth - predicted)

    div_inf: Tensor[bool] = tf_math.is_inf(abs_diff / ground_truth)

    if True in div_inf:
        print('\nabs_diff / ground_truth:', abs_diff / ground_truth)

    return tf_math.reduce_mean(abs_diff / ground_truth)


def sq_rel(ground_truth: Tensor, predicted: Tensor):
    squared_diff = tf_math.squared_difference(ground_truth, predicted)
    return tf_math.reduce_mean(squared_diff / ground_truth)


def rmse(ground_truth: Tensor, predicted: Tensor):
    squared_error: Tensor = tf_math.squared_difference(ground_truth, predicted)
    return tf_math.sqrt(tf_math.reduce_mean(squared_error))


def rmse_log(ground_truth: Tensor, predicted: Tensor):
    gt_log = tf_math.log(ground_truth)
    predicted_log = tf_math.log(predicted)
    squared_log_error = tf_math.squared_difference(gt_log, predicted_log)
    return tf_math.sqrt(tf_math.reduce_mean(squared_log_error))


def log_10(ground_truth: Tensor, predicted: Tensor):
    diff_log_10 = log10(ground_truth) - log10(predicted)
    return tf_math.reduce_mean(tf_math.abs(diff_log_10))
