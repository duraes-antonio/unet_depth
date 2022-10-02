import tensorflow.keras.backend as K


def build_poly_decay(n_epochs: int, init_learning_rate: float):
    def poly_decay(epoch: int):
        max_epochs = n_epochs
        base_lr = init_learning_rate
        power = 1.0
        return base_lr * (1 - (epoch / float(max_epochs))) ** power

    return poly_decay


def depth_acc(y_true, y_predicted):
    return K.mean(K.equal(K.round(y_true), K.round(y_predicted)))
