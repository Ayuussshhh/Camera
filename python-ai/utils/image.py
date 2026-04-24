import base64

import cv2
import numpy as np


def decode_base64_image(image_data: str) -> np.ndarray:
    encoded = image_data.split(",", 1)[1] if "," in image_data else image_data
    image_bytes = base64.b64decode(encoded)
    numpy_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(numpy_buffer, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image payload.")

    return image


def encode_jpeg(image: np.ndarray) -> bytes:
    success, buffer = cv2.imencode(".jpg", image)

    if not success:
        raise ValueError("Unable to encode frame as JPEG.")

    return buffer.tobytes()
