import tensorflow as tf
import tensorflow.keras.backend as k_backend


def depth_loss(y_true, y_predicted, theta=0.1, max_depth_val=1000.0 / 10.0):
    w1, w2, w3 = 1.0, 1.0, theta

    # Point-wise depth
    l_depth = k_backend.mean(k_backend.abs(y_predicted - y_true), axis=-1)

    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_prediction, dx_prediction = tf.image.image_gradients(y_predicted)
    l_edges = k_backend.mean(k_backend.abs(dy_prediction - dy_true) + k_backend.abs(dx_prediction - dx_true), axis=-1)

    # Structural similarity (SSIM) index
    l_ssim = k_backend.clip((1 - tf.image.ssim(y_true, y_predicted, max_depth_val)) * 0.5, 0, 1)

    return (w1 * l_ssim) + (w2 * k_backend.mean(l_edges)) + (w3 * k_backend.mean(l_depth))


def build_l_depth(ground_truth_n_pixels: int):
    def l_depth(y_true, y_predicted):
        """
        Formalizado em: https://arxiv.org/pdf/1411.4734.pdf
        """
        y_true = k_backend.cast(y_true, dtype='float32')
        y_predicted = k_backend.cast(y_predicted, dtype='float32')

        # Replace the ground truth infinite values with the value 1
        log_y_true = k_backend.tf.where(
            k_backend.tf.math.is_inf(y_true),
            k_backend.tf.ones_like(y_true),
            k_backend.log(y_true)
        )

        # Replace the predicted infinite values with the value 1
        log_y_predicted = k_backend.tf.where(
            k_backend.tf.math.is_inf(y_predicted),
            k_backend.tf.ones_like(y_predicted),
            k_backend.log(y_predicted)
        )

        diff_log_y = k_backend.cast(log_y_true - log_y_predicted, dtype='float32')

        # First summation of loss equation
        log_diff = k_backend.cast(
            k_backend.sum(k_backend.square(diff_log_y)) / ground_truth_n_pixels,
            dtype='float32'
        )

        square_n_pixels = k_backend.cast(k_backend.square(ground_truth_n_pixels), dtype='float32')

        # First summation of loss equation
        penalty = k_backend.square(k_backend.sum(diff_log_y)) / square_n_pixels
        return log_diff + penalty

    return l_depth
