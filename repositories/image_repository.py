from pathlib import Path
from typing import Union, Iterable, List, Iterator
import numpy as np
import cv2
from PIL import Image as PILImage
from dotenv import load_dotenv
import os
import signal
from models.image import Image  # adjust if path differs

# Load environment variables
load_dotenv()

class ImageRepository:
    """
    Handles file I/O and pixel updates for Image entities.
    """
    def __init__(self):
        # Load as dictionary
        self.VALID_EXTS = {ext for ext in os.getenv("VALID_IMAGE_EXTENSIONS").split(",")}

    @staticmethod
    def create_image(pixels: np.ndarray, path: Union[str, Path] = None) -> Image:
        if path is None:
            return Image(pixels)
        return Image(pixels=pixels, path=Path(path))

    def retrieve_image_dimensions(self, img: Image):
        return img.pixels.shape[:2]

    @staticmethod
    def load(path: Union[str, Path], rgb: bool = True, timeout: int = 5) -> Image:
        path = Path(path)

        # ─── timeout wrapper (5 s default) ────────────────────────────────
        def _handler(signum, frame):
            raise TimeoutError(f"cv2.imread timed-out after {timeout}s: {path}")

        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout)
        try:
            arr_bgr = cv2.imread(str(path))  # ← original call
        finally:
            signal.alarm(0)  # always disarm
        # ──────────────────────────────────────────────────────────────────

        if arr_bgr is None:
            raise FileNotFoundError(f"Image not found or unreadable: {path}")

        arr = arr_bgr[:, :, ::-1] if rgb else arr_bgr
        return Image(pixels=arr, path=path)
    @staticmethod
    def save(image: Image) -> None:
        PILImage.fromarray(image.pixels).save(image.path)

    @staticmethod
    def set_pixels(image: Image, new_pixels: np.ndarray) -> None:
        image.pixels = new_pixels
    
    @staticmethod
    def save_original_pixels(image: Image) -> None:
        """Save current pixels as original for before/after comparison"""
        if image.original_pixels is None:
            image.original_pixels = image.pixels.copy()
    
    @staticmethod
    def update_pixels_preserve_original(image: Image, new_pixels: np.ndarray) -> None:
        """Update pixels while preserving original for comparison"""
        if image.original_pixels is None:
            image.original_pixels = image.pixels.copy()
        image.pixels = new_pixels

    def iter_dir(
        self,
        folder: Union[str, Path],
        *,
        recursive: bool = False,
        exts: Iterable[str] | None = None,
    ) -> Iterator[Image]:
        """
        Yield Image objects one at a time.  Nothing accumulates in memory.
        """
        folder = Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(folder)

        allowed = {e.lower() for e in (exts or self.VALID_EXTS)}
        pattern = "**/*" if recursive else "*"

        for p in folder.glob(pattern):
            print(f"FILE: {p}")
            if p.suffix.lower() not in allowed:
                print(f"Skipping due to extension: {p}")
                continue
            if not p.is_file():
                print(f"Skipping because not file: {p}")
                continue
            try:
                print(f"Loading: {p}")
                yield self.load(p)
                print(f"Loaded: {p}")
            except Exception as err:
                print(f"Skipping {p.name}: {err}")

    def load_dir(
        self, folder: Union[str, Path], *, recursive=False, exts=None
    ) -> List[Image]:
        """
        Legacy helper that still returns a list, but internally streams.
        """
        return list(self.iter_dir(folder, recursive=recursive, exts=exts))