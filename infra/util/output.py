from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plot
import numpy
from numpy import ndarray
from tensorflow import keras

from domain.models.test_case.test_case import TestCase
from infra.util.dataset import PathPairs
from infra.util.preprocessing import preprocess_image, preprocess_depth_map


def print_test_case(test_case: TestCase):
    title = '-' * 10 + 'PROCESSANDO CASO DE TESTE' + '-' * 10
    output = f"""
{title}
ID:             {test_case['id']}
Network:        {test_case['network']}
Backbone:       {test_case['backbone']}
Otimizador:     {test_case['optimizer']}
Pesos imagenet: {test_case['use_imagenet_weights']}
{'-' * len(title)}
    """
    print(output)


def plot_image_comparison(
        model: keras.Model,
        xy_path_pairs: PathPairs,
        image_size: Tuple[int, int] = (256, 256)
):
    """
    Plot input image, prediction, ground truth side by side
    :param model: Keras model instance
    :param xy_path_pairs: List of path pairs (input path, ground truth path)
    """
    images_name = [Path(x_path).name for x_path, y_path in xy_path_pairs]
    color_map = plot.get_cmap('inferno_r')
    width, height = image_size

    def plot_depth_map(depth_map: ndarray, title: str, col_number: int):
        _depth_map = numpy.squeeze(depth_map, axis=-1)
        plot_axis = plot.subplot(1, 3, col_number)
        plot_axis.set_title(title)
        plot.imshow(_depth_map, cmap=color_map)

    for index, (x_path, y_path) in enumerate(xy_path_pairs):
        plot.figure(figsize=(16, 16), dpi=72)

        x = numpy.empty((1, width, height, 3))
        x[0] = preprocess_image(x_path)

        # Imagem original
        image_name = images_name[index]
        input_axis = plot.subplot(1, 3, 1)
        input_axis.set_title(f'Input ({image_name})')
        plot.imshow(x[0])

        # Predição
        predicted = model.predict(x)
        plot_depth_map(predicted[0], 'Prediction', 2)

        # Ground truth
        y = preprocess_depth_map(y_path)
        plot_depth_map(y, 'Ground Truth', 3)

        plot.show()

    return None
