# repositories/segmentation_repository.py
import cv2
import numpy as np
from models.segmentation_engine import SegmentationEngine

class SegmentationRepository:
    """
    One‑image inference + mask cleanup.

    • Calls MediaPipe engine.
    • Post‑processes raw mask to sharpen torso / hair edges.
    """

    def __init__(self) -> None:
        self.engine = SegmentationEngine()

    # ---------- private helpers ----------
    @staticmethod
    def _clean_mask(mask_u8: np.ndarray) -> np.ndarray:
        """
        1) Close small holes
        2) Erode 1 px fringe
        3) Feather edge → smooth alpha ramp
        """
        kernel = np.ones((5, 5), np.uint8)

        closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
        eroded = cv2.erode(closed, np.ones((2, 2), np.uint8), iterations=1)

        blur = cv2.GaussianBlur(eroded, (0, 0), sigmaX=3, sigmaY=3)
        return (blur > 20).astype("uint8") * 255   # keep soft tail

    # ---------- public API ----------
    def retrieve_mask(self, rgb: np.ndarray, thr: float = 0.35) -> np.ndarray:
        """
        Returns uint8 mask (H, W) with 0‑255 values.
        thr : soft‑mask threshold in [0,1]
        """
        soft = self.engine.predict(rgb)               # float32 0‑1
        raw  = (soft > thr).astype("uint8") * 255
        return self._clean_mask(raw)

