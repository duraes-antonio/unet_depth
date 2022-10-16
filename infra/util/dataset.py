import csv
from random import shuffle
from typing import List, Tuple

from typing_extensions import TypedDict

PathPairs = List[Tuple[str, str]]


class PartitionedDatasetPaths(TypedDict):
    train: PathPairs
    validation: PathPairs


def read_nyu_csv(csv_file_path) -> List[Tuple[str, str]]:
    """
    Lê CSV que relacionada x e y e retona uma lista de pares de paths (x, y)
    :param csv_file_path: Path do arquivo CSV com o nome de x e y
    :return: Lista de pares (path input, path ground truth)
    """
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        return [('./' + row[0], './' + row[1]) for row in csv_reader if len(row) > 0]


def split_train_validation(
        paths_to_split: PathPairs,
        val_percent: float,
        seed: int
) -> Tuple[PathPairs, PathPairs]:
    import random
    random.seed(seed)
    random.shuffle(paths_to_split)

    n_all_paths = len(paths_to_split)
    n_train_paths = int(n_all_paths * (1.0 - val_percent))

    train_paths = paths_to_split[:n_train_paths]
    validation_paths = paths_to_split[n_train_paths:]
    return train_paths, validation_paths


def load_nyu_train_paths(
        train_csv_path: str,
        val_percent: 0.3,
        seed: int,
        dataset_usage_percent: float = 1
) -> PartitionedDatasetPaths:
    xy_paths_pairs = read_nyu_csv(train_csv_path)

    if dataset_usage_percent < 1:
        shuffle(xy_paths_pairs)
        last_index = int(len(xy_paths_pairs) * dataset_usage_percent)
        xy_paths_pairs = xy_paths_pairs[:last_index]

    train_path_pairs, val_path_pairs = split_train_validation(xy_paths_pairs, val_percent, seed)
    return PartitionedDatasetPaths(train=train_path_pairs, validation=val_path_pairs)
