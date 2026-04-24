from typing import Dict, Optional

import cv2
import numpy as np


class AntiSpoofService:
    def evaluate(self, frame: np.ndarray, bbox: Optional[list] = None) -> Dict[str, object]:
        if bbox:
            x, y, w, h = bbox
            frame = frame[max(y, 0) : max(y, 0) + h, max(x, 0) : max(x, 0) + w]

        if frame.size == 0:
            return {
                "passed": False,
                "score": 0.0,
                "mode": "heuristic-ready",
                "reason": "Empty frame crop",
            }

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(grayscale))
        sharpness = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())

        brightness_ok = 40.0 < brightness < 220.0
        sharpness_ok = sharpness > 60.0
        score = min(max((sharpness / 200.0), 0.0), 1.0)

        return {
            "passed": brightness_ok and sharpness_ok,
            "score": round(score, 4),
            "mode": "heuristic-ready",
            "reason": "Passed baseline frame quality checks"
            if brightness_ok and sharpness_ok
            else "Frame failed baseline liveness quality checks",
        }


anti_spoof_service = AntiSpoofService()
