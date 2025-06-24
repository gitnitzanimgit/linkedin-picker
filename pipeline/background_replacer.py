# pipeline/background_replacer.py
from pathlib import Path
import os
import uuid
from typing import List, Tuple

from dotenv import load_dotenv

from models.image import Image
from services.background_service import BackgroundService
from services.image_service import ImageService

# ------------------------------------------------------------------
# env‑vars
load_dotenv()
PROCESSED_DIR = os.getenv("PROCESSED_DIR_PATH", "data/processed_gallery")
OUTPUT_EXT    = os.getenv("OUTPUT_IMG_EXT", ".jpg")          # e.g. ".jpg"
DEFAULT_BG    = (230, 230, 230)                              # LinkedIn light‑gray

# ------------------------------------------------------------------
def replace_background(
    gallery: List[Image],
    *,
    background_service: BackgroundService = BackgroundService(),
    image_service: ImageService          = ImageService(),
    color: Tuple[int, int, int]          = DEFAULT_BG,
    processed_dir: str | Path            = PROCESSED_DIR,
    ext: str                             = OUTPUT_EXT,
) -> List[Image]:
    """
    For every Image in *gallery*:
        • generate person mask
        • composite onto a solid background (default LinkedIn gray)
        • update pixels in-memory (preserving original)
    Returns the same Image objects with updated pixels.
    """
    for img in gallery:
        # 1. replace BG → new pixels
        new_pixels = background_service.replace_with_blur(img)

        # 2. update pixels in-memory while preserving original
        image_service.apply_pipeline_modification(img, new_pixels)

    return gallery
