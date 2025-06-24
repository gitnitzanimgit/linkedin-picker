# # repositories/segmentation_repository.py
# import cv2, numpy as np
# from models.segmentation_engine import SegmentationEngine
# from pathlib import Path
#
#
# class SegmentationRepository:
#     _TARGET_SIZE = 512                       # keep hi‑res input
#
#     def __init__(self):
#         self.engine = SegmentationEngine()
#
#     # ---------- helpers ----------
#     def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
#         return cv2.dnn.blobFromImage(
#             rgb, 1 / 255.0,
#             size=(self._TARGET_SIZE, self._TARGET_SIZE),
#             swapRB=True, crop=False,
#         )
#
#     @staticmethod
#     def _clean_mask(mask: np.ndarray) -> np.ndarray:
#         """
#         1. Close pin‑holes       (dilate→erode)
#         2. Light erode 1 px      (shave fringe)
#         3. Feather edge w/ σ=3   (smooth alpha ramp)
#         Returns uint8 0–255.
#         """
#         kernel = np.ones((5, 5), np.uint8)
#
#         closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
#         eroded = cv2.erode(closed, np.ones((2, 2), np.uint8), iterations=1)
#
#         blur   = cv2.GaussianBlur(eroded, (0, 0), sigmaX=30, sigmaY=3)
#         return (blur > 20).astype("uint8") * 255         # keep soft alpha tail
#
#     # ---------- public API ----------
#     def retrieve_mask(self, rgb: np.ndarray, thr: float = 0.35) -> np.ndarray:
#         """
#         Lower thr (0.35) keeps more pixels, then _clean_mask shaves edges.
#         """
#         h, w = rgb.shape[:2]
#         blob = self._preprocess(rgb)
#         self.engine.net.setInput(blob)
#         prob = self.engine.net.forward()[0, 0]
#         prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
#
#         raw = (prob > thr).astype("uint8") * 255
#         return self._clean_mask(raw)

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

