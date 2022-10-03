import numpy
import tensorflow.keras.backend as K
from numpy import ndarray


def build_poly_decay(n_epochs: int, init_learning_rate: float):
    def poly_decay(epoch: int):
        max_epochs = n_epochs
        base_lr = init_learning_rate
        power = 1.0
        return base_lr * (1 - (epoch / float(max_epochs))) ** power

    return poly_decay


def depth_acc(y_true, y_predicted):
    return K.mean(K.equal(K.round(y_true), K.round(y_predicted)))


def build_threshold(delta: int = 1):
    def threshold(ground_truth: ndarray, predicted: ndarray):
        thresh = numpy.maximum((ground_truth / predicted), (predicted / ground_truth))
        return (thresh < 1.25 ** delta).mean()

    return threshold


def threshold_1():
    return build_threshold(1)


def threshold_2():
    return build_threshold(2)


def threshold_3():
    return build_threshold(3)


def abs_rel(ground_truth: ndarray, predicted: ndarray):
    return numpy.mean(numpy.abs(ground_truth - predicted) / ground_truth)


def sq_rel(ground_truth: ndarray, predicted: ndarray):
    return numpy.mean(((ground_truth - predicted) ** 2) / ground_truth)


def rmse(ground_truth: ndarray, predicted: ndarray):
    square_error = (ground_truth - predicted) ** 2
    return numpy.sqrt(square_error.mean())


def rmse_log(ground_truth: ndarray, predicted: ndarray):
    square_log_error = (numpy.log(ground_truth) - numpy.log(predicted)) ** 2
    return numpy.sqrt(square_log_error.mean())


def log_10(ground_truth: ndarray, predicted: ndarray):
    diff_log_10 = numpy.log10(ground_truth) - numpy.log10(predicted)
    return (numpy.abs(diff_log_10)).mean()
