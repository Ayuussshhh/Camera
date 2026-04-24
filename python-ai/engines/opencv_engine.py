import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import cv2
import numpy as np
from mediapipe import Image as MPImage
from mediapipe import ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from engines.base_engine import BaseRecognitionEngine
from utils.config import settings


LOGGER = logging.getLogger(__name__)
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
LANDMARK_SAMPLE_INDICES = [
    33,
    133,
    362,
    263,
    1,
    61,
    291,
    199,
    10,
    152,
    4,
    168,
    234,
    454,
    78,
    308,
    13,
]


class MediaPipeEngine(BaseRecognitionEngine):
    name = "mediapipe"
    minimum_enrollment_quality = 0.34

    def __init__(self) -> None:
        model_path = self._ensure_face_landmarker_model()
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=5,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def extract_embeddings(self, image: np.ndarray) -> List[Dict[str, object]]:
        if image is None or image.size == 0:
            return []

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb_image)
        landmarker_result = self.face_landmarker.detect(mp_image)

        if not landmarker_result.face_landmarks:
            return []

        detections: List[Dict[str, object]] = []
        image_height, image_width = image.shape[:2]

        for face_landmarks in landmarker_result.face_landmarks:
            coordinates = np.asarray(
                [
                    (landmark.x * image_width, landmark.y * image_height)
                    for landmark in face_landmarks
                ],
                dtype=np.float32,
            )
            bbox = self._landmark_bbox(coordinates, image.shape)

            if not bbox:
                continue

            x, y, w, h = bbox
            face_crop = image[y : y + h, x : x + w]
            local_coordinates = coordinates.copy()
            local_coordinates[:, 0] -= x
            local_coordinates[:, 1] -= y

            if face_crop.size == 0 or min(face_crop.shape[:2]) < 60:
                continue

            aligned_face, landmark_vector, frontal_score = self._prepare_face(
                face_crop,
                local_coordinates,
            )

            if aligned_face is None:
                LOGGER.debug("Skipping empty aligned face crop.")
                continue

            embedding = self._build_embedding(aligned_face, landmark_vector)
            quality = self._quality_score(
                aligned_face,
                image.shape,
                w,
                h,
                0.75,
                frontal_score,
            )

            detections.append(
                {
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "embedding": embedding.astype(np.float32).tolist(),
                    "quality": round(quality, 4),
                    "detector_confidence": 0.75,
                }
            )

        return sorted(
            detections,
            key=lambda item: float(item.get("quality", 0.0)),
            reverse=True,
        )

    def _ensure_face_landmarker_model(self) -> Path:
        settings.models_dir.mkdir(parents=True, exist_ok=True)

        if settings.face_landmarker_model_path.exists():
            return settings.face_landmarker_model_path

        LOGGER.info(
            "Downloading MediaPipe face landmarker model from %s",
            FACE_LANDMARKER_MODEL_URL,
        )

        try:
            urlretrieve(
                FACE_LANDMARKER_MODEL_URL,
                settings.face_landmarker_model_path,
            )
        except Exception as error:  # pragma: no cover - network dependent
            raise RuntimeError(
                "Unable to download the MediaPipe face landmarker model. "
                f"Download it manually to '{settings.face_landmarker_model_path.as_posix()}'."
            ) from error

        return settings.face_landmarker_model_path

    def _landmark_bbox(
        self,
        coordinates: np.ndarray,
        image_shape: Tuple[int, int, int],
    ) -> Optional[Tuple[int, int, int, int]]:
        image_height, image_width = image_shape[:2]
        min_x = float(np.min(coordinates[:, 0]))
        min_y = float(np.min(coordinates[:, 1]))
        max_x = float(np.max(coordinates[:, 0]))
        max_y = float(np.max(coordinates[:, 1]))
        pad_x = max(int((max_x - min_x) * 0.18), 12)
        pad_y = max(int((max_y - min_y) * 0.22), 12)

        x1 = max(int(min_x) - pad_x, 0)
        y1 = max(int(min_y) - pad_y, 0)
        x2 = min(int(max_x) + pad_x, image_width)
        y2 = min(int(max_y) + pad_y, image_height)
        width = x2 - x1
        height = y2 - y1

        if width <= 1 or height <= 1:
            return None

        return x1, y1, width, height

    def _prepare_face(
        self,
        face_crop: np.ndarray,
        coordinates: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], np.ndarray, float]:
        crop_height, crop_width = face_crop.shape[:2]
        left_eye = coordinates[[33, 133, 159, 145]].mean(axis=0)
        right_eye = coordinates[[362, 263, 386, 374]].mean(axis=0)
        angle = np.degrees(
            np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        )
        center = tuple(((left_eye + right_eye) / 2.0).tolist())
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_face = cv2.warpAffine(
            face_crop,
            rotation_matrix,
            (crop_width, crop_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )
        rotated_coordinates = self._apply_affine(coordinates, rotation_matrix)
        x1, y1, x2, y2 = self._landmark_crop_box(rotated_coordinates, rotated_face.shape)
        focused_face = rotated_face[y1:y2, x1:x2]

        if focused_face.size == 0:
            focused_face = rotated_face

        resized_face = cv2.resize(focused_face, (128, 128), interpolation=cv2.INTER_CUBIC)
        landmark_vector = self._landmark_vector(rotated_coordinates, (x1, y1, x2, y2))
        frontal_score = self._frontality_score(rotated_coordinates)

        return resized_face, landmark_vector, frontal_score

    def _apply_affine(
        self,
        coordinates: np.ndarray,
        matrix: np.ndarray,
    ) -> np.ndarray:
        homogeneous = np.concatenate(
            [coordinates, np.ones((coordinates.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
        return homogeneous @ matrix.T

    def _landmark_crop_box(
        self,
        coordinates: np.ndarray,
        image_shape: Tuple[int, int, int],
    ) -> Tuple[int, int, int, int]:
        image_height, image_width = image_shape[:2]
        min_x = float(np.min(coordinates[:, 0]))
        min_y = float(np.min(coordinates[:, 1]))
        max_x = float(np.max(coordinates[:, 0]))
        max_y = float(np.max(coordinates[:, 1]))
        pad_x = max(int((max_x - min_x) * 0.18), 12)
        pad_y = max(int((max_y - min_y) * 0.22), 12)

        x1 = max(int(min_x) - pad_x, 0)
        y1 = max(int(min_y) - pad_y, 0)
        x2 = min(int(max_x) + pad_x, image_width)
        y2 = min(int(max_y) + pad_y, image_height)

        return x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)

    def _landmark_vector(
        self,
        coordinates: np.ndarray,
        face_box: Tuple[int, int, int, int],
    ) -> np.ndarray:
        x1, y1, x2, y2 = face_box
        width = max(float(x2 - x1), 1.0)
        height = max(float(y2 - y1), 1.0)
        selected = coordinates[LANDMARK_SAMPLE_INDICES].copy()
        selected[:, 0] = np.clip((selected[:, 0] - x1) / width, 0.0, 1.0)
        selected[:, 1] = np.clip((selected[:, 1] - y1) / height, 0.0, 1.0)

        eye_distance = float(np.linalg.norm(coordinates[263] - coordinates[33])) + 1e-6
        mouth_width = float(np.linalg.norm(coordinates[291] - coordinates[61]) / eye_distance)
        nose_to_chin = float(np.linalg.norm(coordinates[152] - coordinates[1]) / eye_distance)
        jaw_width = float(np.linalg.norm(coordinates[454] - coordinates[234]) / eye_distance)
        face_height = float(np.linalg.norm(coordinates[152] - coordinates[10]) / eye_distance)
        lip_open = float(np.linalg.norm(coordinates[13] - coordinates[14]) / eye_distance)
        nose_to_forehead = float(np.linalg.norm(coordinates[10] - coordinates[1]) / eye_distance)
        ratios = np.asarray(
            [
                mouth_width,
                nose_to_chin,
                jaw_width,
                face_height,
                lip_open,
                nose_to_forehead,
            ],
            dtype=np.float32,
        )

        return np.concatenate([selected.flatten(), ratios]).astype(np.float32)

    def _frontality_score(self, coordinates: np.ndarray) -> float:
        eye_distance = float(np.linalg.norm(coordinates[263] - coordinates[33])) + 1e-6
        eye_mid_x = float((coordinates[33][0] + coordinates[263][0]) / 2.0)
        nose_x = float(coordinates[1][0])
        mouth_mid_x = float((coordinates[61][0] + coordinates[291][0]) / 2.0)
        nose_offset = abs(nose_x - eye_mid_x) / eye_distance
        mouth_offset = abs(nose_x - mouth_mid_x) / eye_distance
        vertical_eye_delta = abs(coordinates[33][1] - coordinates[263][1]) / eye_distance

        return float(
            np.clip(1.0 - min(nose_offset + mouth_offset + vertical_eye_delta, 1.0), 0.0, 1.0)
        )

    def _build_embedding(
        self,
        aligned_face: np.ndarray,
        landmark_vector: np.ndarray,
    ) -> np.ndarray:
        grayscale = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        grayscale = self.clahe.apply(grayscale)
        grayscale = cv2.GaussianBlur(grayscale, (3, 3), 0)
        grayscale_float = grayscale.astype(np.float32) / 255.0

        histogram = cv2.calcHist([grayscale], [0], None, [32], [0, 256]).flatten().astype(np.float32)
        histogram = histogram / (np.linalg.norm(histogram) + 1e-8)

        hog_features = self._hog_descriptor(grayscale)
        lbp_features = self._lbp_descriptor(grayscale)
        dct_features = cv2.dct(grayscale_float)[:16, :16].flatten()[1:].astype(np.float32)
        dct_features = dct_features / (np.linalg.norm(dct_features) + 1e-8)
        normalized_landmarks = landmark_vector / (np.linalg.norm(landmark_vector) + 1e-8)

        embedding = np.concatenate(
            [
                0.28 * hog_features,
                0.2 * lbp_features,
                0.24 * dct_features,
                0.08 * histogram,
                0.2 * normalized_landmarks,
            ],
        ).astype(np.float32)

        return embedding / (np.linalg.norm(embedding) + 1e-8)

    def _hog_descriptor(self, grayscale: np.ndarray) -> np.ndarray:
        resized = cv2.resize(grayscale, (128, 128), interpolation=cv2.INTER_CUBIC)
        gradient_x = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=1)
        gradient_y = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=1)
        magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)
        angle = angle % 180.0
        cell_size = 32
        bins = 9
        cell_histograms = []

        for start_y in range(0, 128, cell_size):
            for start_x in range(0, 128, cell_size):
                cell_magnitude = magnitude[start_y : start_y + cell_size, start_x : start_x + cell_size].flatten()
                cell_angle = angle[start_y : start_y + cell_size, start_x : start_x + cell_size].flatten()
                histogram, _ = np.histogram(
                    cell_angle,
                    bins=bins,
                    range=(0, 180),
                    weights=cell_magnitude,
                )
                cell_histograms.append(histogram.astype(np.float32))

        histogram_grid = np.asarray(cell_histograms, dtype=np.float32).reshape(4, 4, bins)
        blocks = []

        for row in range(3):
            for column in range(3):
                block = histogram_grid[row : row + 2, column : column + 2].flatten()
                block = block / (np.linalg.norm(block) + 1e-8)
                blocks.append(block)

        return np.concatenate(blocks).astype(np.float32)

    def _lbp_descriptor(self, grayscale: np.ndarray) -> np.ndarray:
        resized = cv2.resize(grayscale, (130, 130), interpolation=cv2.INTER_CUBIC)
        center = resized[1:-1, 1:-1]
        lbp = np.zeros_like(center, dtype=np.uint8)
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
        ]

        for bit, (offset_y, offset_x) in enumerate(neighbors):
            comparison = resized[
                1 + offset_y : 129 + offset_y,
                1 + offset_x : 129 + offset_x,
            ] >= center
            lbp |= comparison.astype(np.uint8) << bit

        cell_size = 32
        cell_histograms = []

        for start_y in range(0, 128, cell_size):
            for start_x in range(0, 128, cell_size):
                cell = lbp[start_y : start_y + cell_size, start_x : start_x + cell_size].flatten()
                histogram, _ = np.histogram(cell, bins=16, range=(0, 256))
                histogram = histogram.astype(np.float32)
                histogram = histogram / (np.linalg.norm(histogram) + 1e-8)
                cell_histograms.append(histogram)

        return np.concatenate(cell_histograms).astype(np.float32)

    def _quality_score(
        self,
        aligned_face: np.ndarray,
        frame_shape: Tuple[int, int, int],
        bbox_width: int,
        bbox_height: int,
        detector_score: float,
        frontal_score: float,
    ) -> float:
        grayscale = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
        brightness = float(np.mean(grayscale))
        frame_area = max(float(frame_shape[0] * frame_shape[1]), 1.0)
        face_area_ratio = float((bbox_width * bbox_height) / frame_area)
        sharpness_score = min(sharpness / 180.0, 1.0)
        exposure_score = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
        size_score = min(face_area_ratio / 0.14, 1.0)

        return float(
            np.clip(
                (0.35 * detector_score)
                + (0.25 * sharpness_score)
                + (0.2 * size_score)
                + (0.1 * exposure_score)
                + (0.1 * frontal_score),
                0.0,
                1.0,
            )
        )


OpenCVEngine = MediaPipeEngine
