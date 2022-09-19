import csv
from typing import List, Tuple, Dict, MutableSequence, TypedDict


class PartitionedDataset(TypedDict):
    train: MutableSequence[str]
    validation: MutableSequence[str]


def read_nyu_csv(csv_file_path) -> List[Tuple[str, str]]:
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        return [('./' + row[0], './' + row[1]) for row in csv_reader if len(row) > 0]


def split_train_validation(
        paths_to_split: MutableSequence[str],
        val_percent: float,
        seed: int
) -> Tuple[MutableSequence[str], MutableSequence[str]]:
    import random
    random.seed(seed)
    random.shuffle(paths_to_split)

    count_all_paths = len(paths_to_split)
    count_train_paths = int(count_all_paths * (1.0 - val_percent))

    train_paths = paths_to_split[:count_train_paths]
    validation_paths = paths_to_split[count_train_paths:]
    return train_paths, validation_paths


def load_nyu_train_paths(
        train_csv_path: str,
        val_percent: 0.3,
        seed: int
) -> Tuple[PartitionedDataset, Dict[str, str]]:
    xy_paths_pairs = read_nyu_csv(train_csv_path)
    y_path_by_x_path: Dict[str, str] = {x_path: y_path for x_path, y_path in xy_paths_pairs}

    x_paths = [x_path for x_path, y_path in xy_paths_pairs]
    x_train_paths, x_val_paths = split_train_validation(x_paths, val_percent, seed)

    partition = PartitionedDataset(train=x_train_paths, validation=x_val_paths)
    return partition, y_path_by_x_path
