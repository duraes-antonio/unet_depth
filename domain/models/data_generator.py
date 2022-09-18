import random
from typing import Tuple, List

import numpy as np
from keras.utils import Sequence
from numpy import ndarray

from domain.util.preprocessing import preprocess_image, preprocess_depth_map


# from tensorflow.keras.utils import Sequence


class NyuV2Generator(Sequence):
    shuffle: bool
    is_depth_map: bool
    batch_size: int
    n_channels: int
    image_size: Tuple[int, int] = (640, 480)
    indexes: ndarray

    def __init__(self, list_ids: List[str], labels, batch_size=4, n_channels=3, shuffle=True, is_depth_map=False):
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.is_depth_map = is_depth_map
        self.shuffle = shuffle
        self.labels = labels
        self.list_IDs = list_ids
        self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_ids_temp = [self.list_IDs[k] for k in indexes]

        if self.is_depth_map:
            return self.__data_generation__(list_ids_temp)

        x, y = self.__data_generation__(list_ids_temp)
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation__(self, list_ids_temp: List[str]):
        width, height = self.image_size
        x = np.empty((self.batch_size, width, height, self.n_channels))

        if self.is_depth_map:
            # TODO: desabilitar flip
            for index, _id in enumerate(list_ids_temp):
                res = random.choice([True, False])
                x[index,] = preprocess_image(_id, res)
            return x

        y = np.empty((self.batch_size, width, height, 1))

        for index, _id in enumerate(list_ids_temp):
            # TODO: desabilitar flip
            res = random.choice([True, False])
            x[index,] = preprocess_image(_id, res)
            y[index,] = preprocess_depth_map(self.labels[_id], res)
        return x, y
