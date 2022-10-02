from pathlib import Path
from typing import Union, List

import matplotlib.pyplot as plot
import numpy
from keras import Model
from numpy import array, ndarray

from domain.models.test_case import TestCase
from infra.util.preprocessing import preprocess_depth_map


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
        images: ndarray,
        ground_truth_paths: Union[List[str], array],
        n: int
):
    """
    Plot input image, prediction, ground truth side by side
    :param model: Keras model instance
    :param images: ndarray with all input images
    :param ground_truth_paths: Ground truth path list (must be in same order as input images array)
    :param n: Number of images to be displayed
    """
    predicted = model.predict(images[:n, ])
    n_max = min(n, len(images))
    color_map = plot.get_cmap('inferno_r')

    def plot_depth_map(depth_map: ndarray, title: str, col_number: int):
        _depth_map = numpy.squeeze(depth_map, axis=-1)
        plot_axis = plot.subplot(1, 3, col_number)
        plot.imshow(_depth_map, cmap=color_map)
        plot_axis.set_title(title)

    for index in range(n_max):
        plot.figure(figsize=(16, 16), dpi=72)

        # Imagem original
        input_axis = plot.subplot(1, 3, 1)
        input_image = images[index]
        plot.imshow(input_image)
        input_axis.set_title('Input')

        # Predição
        prediction = predicted[index]
        plot_depth_map(prediction, 'Prediction', 2)

        # Ground truth
        label_path = ground_truth_paths[index]
        target_depth_map = preprocess_depth_map(label_path)
        plot_depth_map(target_depth_map, Path(label_path).name, 3)

        plot.show()

    return None
