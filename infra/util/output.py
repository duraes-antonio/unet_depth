from pathlib import Path

import matplotlib.pyplot as plot
import numpy
from keras import Model
from numpy import ndarray

from domain.models.data.data_generator import NyuV2Generator
from domain.models.test_case import TestCase
from infra.util.dataset import PathPairs


def print_test_case(test_case: TestCase):
    title = '-' * 10 + 'CASO DE TESTE' + '-' * 10
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
        model: Model,
        xy_path_pairs: PathPairs,
):
    """
    Plot input image, prediction, ground truth side by side
    :param model: Keras model instance
    :param xy_path_pairs: List of path pairs (input path, ground truth path)
    """
    images_name = [Path(x_path).name for x_path, y_path in xy_path_pairs]
    test_generator = NyuV2Generator(xy_path_pairs, shuffle=False)
    color_map = plot.get_cmap('inferno_r')

    def plot_depth_map(depth_map: ndarray, title: str, col_number: int):
        _depth_map = numpy.squeeze(depth_map, axis=-1)
        plot_axis = plot.subplot(1, 3, col_number)
        plot.imshow(_depth_map, cmap=color_map)
        plot_axis.set_title(title)

    for image_name, (x, y) in zip(images_name, test_generator):
        plot.figure(figsize=(16, 16), dpi=72)
        predicted = model.predict(x)

        # Imagem original
        input_axis = plot.subplot(1, 3, 1)
        plot.imshow(x)
        input_axis.set_title('Input')

        # Predição
        plot_depth_map(predicted, 'Prediction', 2)

        # Ground truth
        plot_depth_map(y, image_name, 3)

        plot.show()

    return None
