from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Image:
    """
    Simple data object: RGB pixels (+ optional source path for bookkeeping).
    No OpenCV logic outside this file.
    """
    pixels: np.ndarray # Shape (H, W, 3), dtype uint8, RGB order.
    path: Path | None = None # Source of the image.
    original_pixels: np.ndarray | None = None # Original unmodified pixels for comparison
