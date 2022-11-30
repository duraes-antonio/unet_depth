from typing import Tuple, Dict, Any

import cv2
from numpy import array, ndarray, reshape

from domain.models.test_case.test_case import InputReadMode

DEFAULT_HEIGHT = 256
DEFAULT_SIZE = (DEFAULT_HEIGHT, DEFAULT_HEIGHT)


def normalize_img(img: array) -> ndarray:
    return (img - img.min()) / (img.max() - img.min())


def preprocess_image(img_path: str, size: Tuple[int, int] = DEFAULT_SIZE) -> ndarray:
    image: ndarray = cv2.imread(img_path)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    image = image.astype("float")
    image = normalize_img(image)
    return image


opencv_mode_from_enum: Dict[InputReadMode, Any] = {
    InputReadMode.BGR2GRAY: cv2.COLOR_BGR2GRAY,
    InputReadMode.ANY_DEPTH: cv2.IMREAD_ANYDEPTH,
}


def preprocess_depth_map(
        depth_map_path: str,
        read_mode: InputReadMode,
        size: Tuple[int, int] = DEFAULT_SIZE
) -> ndarray:
    depth_map: ndarray = cv2.imread(depth_map_path, opencv_mode_from_enum[read_mode])
    depth_map = cv2.resize(depth_map, size, interpolation=cv2.INTER_AREA)
    width, height = size

    depth_map = depth_map.astype("float")
    depth_map = normalize_img(depth_map)
    depth_map = reshape(depth_map, (width, height, 1))
    return depth_map
