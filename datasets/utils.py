import numpy as np
import cv2

def calculate_new_dimensions(height, width, max_size: int=512):
    aspect_ratio = width / height
    if height > width:
        height = max_size
        width  = int(height * aspect_ratio)
    else:
        width  = max_size
        height = int(width / aspect_ratio)

    return height, width

def resize(
    image: np.ndarray,
    landmarks: np.ndarray=None,
    dimensions: tuple=(int, int)
):
    image_height, image_width = image.shape[0:2]
    new_height, new_width = dimensions
    ratio_height, ratio_width = (image_height / new_height), (image_width / new_width)
    
    image = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )
    
    if landmarks is not None:
        landmarks = np.stack([
            landmarks[:, 0] / ratio_width,
            landmarks[:, 1] / ratio_height
        ], axis=-1)

        return image, landmarks

    return image

def normalize_landmarks(
    landmarks: np.ndarray, 
    height: int, 
    width: int, 
    num_landmarks: int=19
):
    original_shape = landmarks.shape
    landmarks = np.reshape(landmarks, (-1, num_landmarks, 2))

    landmarks = np.stack([
        landmarks[:, :, 0] / width,
        landmarks[:, :, 1] / height
    ], axis=-1, dtype=np.float32)

    if len(original_shape) == 2:
        landmarks = np.reshape(landmarks, original_shape)

    return landmarks

def denormalize_landmarks(
    landmarks: np.ndarray, 
    height: int, 
    width: int, 
    num_landmarks: int=19
):
    original_shape = landmarks.shape
    landmarks = np.reshape(landmarks, (-1, num_landmarks, 2))

    landmarks = np.stack([
        landmarks[:, :, 0] * width,
        landmarks[:, :, 1] * height
    ], axis=-1, dtype=np.float32)

    if len(original_shape) == 2:
        landmarks = np.reshape(landmarks, original_shape)

    return landmarks.round()

def rescale(image, scale: float, offset: int = 0, dtype: str="float32"):
    scale = np.array(scale, dtype)
    offset = np.array(offset, dtype)
    image = image * scale + offset
    return image