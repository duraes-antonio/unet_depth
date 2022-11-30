from typing import Tuple

import numpy
from numpy import ndarray, arange, floor
from tensorflow import keras

from domain.models.test_case.test_case import InputReadMode
from infra.util.dataset import PathPairs
from infra.util.preprocessing import preprocess_image, preprocess_depth_map


class NyuV2Generator(keras.utils.Sequence):
    batch_size: int
    n_channels: int
    indexes: ndarray
    image_size: Tuple[int, int]

    def __init__(
            self, path_list: PathPairs,
            batch_size=8, shuffle=False, seed=42,
            image_size: Tuple[int, int] = (256, 256),
            n_channels: int = 3,
            read_mode: InputReadMode = InputReadMode.BGR2GRAY
    ):
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.paths = path_list
        self.indexes = arange(len(path_list))
        self.length = int(floor(len(path_list) / batch_size))
        self.image_size = image_size
        self.read_mode = read_mode

        if shuffle:
            self.__shuffle_indexes__(path_list, seed)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Tuple[ndarray, ndarray]:
        start_index_batch = index * self.batch_size
        end_index_batch = (index + 1) * self.batch_size
        indexes = self.indexes[start_index_batch: end_index_batch]
        list_ids_temp = [self.paths[i] for i in indexes]
        return self.__data_generation__(list_ids_temp)

    @staticmethod
    def __shuffle_indexes__(path_pairs: PathPairs, seed: int) -> ndarray:
        numpy.random.seed(seed)
        indexes = arange(len(path_pairs))
        numpy.random.shuffle(indexes)
        return indexes

    def __data_generation__(self, paths_pairs: PathPairs) -> Tuple[ndarray, ndarray]:
        """
        Constrói uma tupla com os pares de ndarray de entrada e de ground truth
        :param paths_pairs: Pares de paths (imagem entrada, ground truth)
        :return: Par (ndarray imagens de entrada, ndarray ground truth)
        """
        width, height = self.image_size
        empty_batch = numpy.empty((self.batch_size, height, width, self.n_channels))
        target_depth_map = numpy.empty((self.batch_size, height, width, 1))

        for index, (x_path, y_path) in enumerate(paths_pairs):
            empty_batch[index,] = preprocess_image(x_path, self.image_size)
            target_depth_map[index,] = preprocess_depth_map(y_path, self.read_mode, self.image_size)
        return empty_batch, target_depth_map
