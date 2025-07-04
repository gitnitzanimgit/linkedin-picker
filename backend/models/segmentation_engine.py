# models/segmentation_engine.py
"""
Singleton wrapper around MediaPipe Selfie Segmentation.

• Loads the TFLite graph once per Python process.
• Exposes .predict(rgb)  →  float mask (H, W) in [0, 1].
"""
from __future__ import annotations
import mediapipe as mp
import numpy as np

class SegmentationEngine:
    _instance: "SegmentationEngine" | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_runtime()
        return cls._instance

    # --------------------------------------------------
    def _init_runtime(self) -> None:
        self._mp_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )

    # --------------------------------------------------
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """
        Args
        ----
        rgb : np.ndarray  (H, W, 3)  uint8  RGB order

        Returns
        -------
        mask : np.ndarray  (H, W)  float32  [0, 1]
        """
        results = self._mp_seg.process(rgb)
        return results.segmentation_mask.astype("float32")

