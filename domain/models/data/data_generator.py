from typing import Tuple, List, Iterable, Union

import numpy
from numpy import ndarray, arange, floor
from tensorflow.keras.utils import Sequence

from infra.util.preprocessing import preprocess_image, preprocess_depth_map


class NyuV2Generator(Sequence):
    shuffle: bool
    is_depth_map: bool
    batch_size: int
    n_channels: int
    image_size: Tuple[int, int] = (640, 480)
    indexes: ndarray

    def __init__(self, path_list: List[str], labels, batch_size=4, n_channels=3, shuffle=True, is_depth_map=False, seed=42):
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.is_depth_map = is_depth_map
        self.labels = labels
        self.paths = path_list
        self.indexes = arange(len(path_list))
        self.length = int(floor(len(path_list) / batch_size))

        self.shuffle = shuffle
        self.seed = seed
        self.on_epoch_end()

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> Union[ndarray, Tuple[ndarray, ndarray]]:
        start_index_batch = index * self.batch_size
        end_index_batch = (index + 1) * self.batch_size
        indexes = self.indexes[start_index_batch: end_index_batch]
        list_ids_temp = [self.paths[i] for i in indexes]
        return self.__data_generation__(list_ids_temp)

    def on_epoch_end(self):
        self.indexes = arange(len(self.paths))

        if self.shuffle:
            numpy.random.shuffle(self.indexes)

    def __data_generation__(self, paths: Iterable[str]) -> Union[ndarray, Tuple[ndarray, ndarray]]:
        width, height = self.image_size
        empty_batch = numpy.empty((self.batch_size, width, height, self.n_channels))

        if self.is_depth_map:
            for index, target_path in enumerate(paths):
                empty_batch[index,] = preprocess_image(target_path)
            return empty_batch

        target_depth_map = numpy.empty((self.batch_size, width, height, 1))

        for index, target_path in enumerate(paths):
            empty_batch[index,] = preprocess_image(target_path)
            target_depth_map[index,] = preprocess_depth_map(self.labels[target_path])
        return empty_batch, target_depth_map
