# services/segmentation_service.py
from typing import Dict
import numpy as np
from models.image import Image
from repositories.segmentation_repository import SegmentationRepository

class SegmentationService:
    """
    Caches masks at the business‑logic layer.
    """

    def __init__(self) -> None:
        self.repo = SegmentationRepository()
        self._mask_cache: Dict[int, np.ndarray] = {}

    def mask_person(self, img: Image, thr: float = 0.5) -> np.ndarray:
        key = id(img)
        if key not in self._mask_cache:
            self._mask_cache[key] = self.repo.retrieve_mask(img.pixels, thr)
        return self._mask_cache[key]
