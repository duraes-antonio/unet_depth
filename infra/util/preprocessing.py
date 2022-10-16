import cv2
from numpy import ndarray, reshape

DEFAULT_HEIGHT = 256
DEFAULT_SIZE = (DEFAULT_HEIGHT, DEFAULT_HEIGHT)


def normalize_img(img: ndarray) -> ndarray:
    return (img - img.min()) / (img.max() - img.min())


def preprocess_image(img_path: str) -> ndarray:
    image: ndarray = cv2.imread(img_path)
    image = cv2.resize(image, DEFAULT_SIZE, interpolation=cv2.INTER_AREA)
    image = image.astype("float")
    image = normalize_img(image)
    return image


def preprocess_depth_map(depth_map_path: str) -> ndarray:
    depth_map: ndarray = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH)
    depth_map = cv2.resize(depth_map, DEFAULT_SIZE, interpolation=cv2.INTER_AREA)

    depth_map = depth_map.astype("float")
    depth_map = normalize_img(depth_map)
    depth_map = reshape(depth_map, (DEFAULT_HEIGHT, DEFAULT_HEIGHT, 1))
    return depth_map
