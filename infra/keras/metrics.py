import tensorflow
import tensorflow.python.keras.backend as k_backend
from tensorflow import Tensor


def log10(value: Tensor):
    numerator = k_backend.log(value)
    denominator = k_backend.log(tensorflow.constant(10, dtype=numerator.dtype))
    return numerator / denominator


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
        thresh = k_backend.maximum((ground_truth / predicted), (predicted / ground_truth))
        return k_backend.mean(thresh < 1.25 ** delta)

    return threshold


def threshold_1():
    return build_threshold(1)


def threshold_2():
    return build_threshold(2)


def threshold_3():
    return build_threshold(3)


def abs_rel(ground_truth: Tensor, predicted: Tensor):
    return k_backend.mean(k_backend.abs(ground_truth - predicted) / ground_truth)


def sq_rel(ground_truth: Tensor, predicted: Tensor):
    return k_backend.mean(((ground_truth - predicted) ** 2) / ground_truth)


def rmse(ground_truth: Tensor, predicted: Tensor):
    square_error: Tensor = (ground_truth - predicted) ** 2
    return k_backend.sqrt(k_backend.mean(square_error))


def rmse_log(ground_truth: Tensor, predicted: Tensor):
    square_log_error = (k_backend.log(ground_truth) - k_backend.log(predicted)) ** 2
    return k_backend.sqrt(k_backend.mean(square_log_error))


def log_10(ground_truth: Tensor, predicted: Tensor):
    diff_log_10 = log10(ground_truth) - log10(predicted)
    diff_log_10_tensor = tensorflow.convert_to_tensor(diff_log_10)
    return k_backend.mean(k_backend.abs(diff_log_10_tensor))
