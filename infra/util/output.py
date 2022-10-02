from typing import Union, List

import matplotlib.pyplot as plot
import numpy
from keras import Model
from numpy import array

from domain.models.test_case import TestCase
from infra.util.preprocessing import preprocess_depth_map, preprocess_image


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
        images_path: Union[List[str], array],
        images_gt: Union[List[str], array],
        n: int
):
    predicted = model.predict(images_path)
    n_max = min(n, len(images_path))

    for index in range(n_max):
        # Predição
        prediction = predicted[index]
        prediction = numpy.squeeze(prediction, axis=-1)
        plot.subplot(1, 3, 1)
        plot.axis('off')
        plot.imshow(prediction, cmap=plot.get_cmap('viridis_r'))

        # Ground truth
        path = images_path[index]
        label_path = images_gt[index]
        plot.subplot(1, 3, 2)
        plot.axis('off')
        target_depth_map = preprocess_depth_map(label_path)
        target_depth_map = numpy.squeeze(target_depth_map, axis=-1)
        plot.imshow(target_depth_map, cmap=plot.get_cmap('inferno_r'))

        # Imagem original
        plot.subplot(1, 3, 3)
        plot.axis('off')
        original_image = preprocess_image(path)
        plot.imshow(original_image)
        plot.show()

    return None
