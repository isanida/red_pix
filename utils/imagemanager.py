import cv2
import numpy as np
import os


def get_filename(image_id: str, data_path: str) -> str:
    """
    Method to get image file path from its id and type
    """
    ext = 'jpg'
    return os.path.join(data_path, "{}.{}".format(image_id, ext))


def read_image(img_path: str=None, data_path: str=None, img_id: str=None) -> np.ndarray:
    """
    Method to get image data as np.array specifying image id and type
    """
    if img_id and data_path:
        img_path = get_filename(image_id=img_id, data_path=data_path)

    if img_path:
        img = cv2.imread(img_path)
        assert img is not None, "Failed to read image : %s, %s" % (img_id, data_path)
        print("Read image: {}".format(img_path))
        return img

    return np.array([])





