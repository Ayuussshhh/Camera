from typing import Generator

import cv2

from utils.config import settings
from utils.image import encode_jpeg


class CameraManager:
    def __init__(self) -> None:
        self.capture = None

    def open(self):
        if self.capture is not None and self.capture.isOpened():
            return self.capture

        self.capture = cv2.VideoCapture(settings.camera_index)

        if not self.capture.isOpened():
            raise RuntimeError("Unable to open the configured camera device.")

        return self.capture

    def read(self):
        capture = self.open()
        success, frame = capture.read()

        if not success:
            raise RuntimeError("Unable to read a frame from camera.")

        return frame

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


camera_manager = CameraManager()


def generate_mjpeg_stream() -> Generator[bytes, None, None]:
    while True:
        frame = camera_manager.read()
        payload = encode_jpeg(frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
        )
