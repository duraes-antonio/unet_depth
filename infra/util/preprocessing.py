import cv2
from numpy import ndarray, reshape


def normalize_img(img: ndarray) -> ndarray:
    return (img - img.min()) / (img.max() - img.min())


def preprocess_image(img_path: str) -> ndarray:
    image: ndarray = cv2.imread(img_path)
    image = image.astype("float")
    image = normalize_img(image)
    return image


def preprocess_depth_map(depth_map_path: str, image_shape=(640, 480)) -> ndarray:
    depth_map: ndarray = cv2.imread(depth_map_path)
    depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
    width, height = image_shape

    depth_map = depth_map.astype("float")
    depth_map = normalize_img(depth_map)
    depth_map = reshape(depth_map, (height, width, 1))
    return depth_map
