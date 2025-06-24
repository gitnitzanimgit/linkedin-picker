from pathlib import Path
from typing import Iterable, List, Union, Iterator
import cv2
import os
import numpy as np
from models.image import Image
from models.scored_image import ScoredImage
from repositories.image_repository import ImageRepository
from dotenv import load_dotenv
import torch
import torchvision.transforms as T
from PIL import Image as PILImage
from models.image_adjustments import BrightnessContrastGamma

# Load environment variables
load_dotenv()


class ImageService:
    """I/O helpers.  No CV logic, no InsightFace imports."""
    def __init__(self):
        self.SHARPNESS_THR = float(os.getenv("SHARPNESS_THR"))
        self.BRIGHT_MIN = float(os.getenv("BRIGHT_MIN"))
        self.BRIGHT_MAX = float(os.getenv("BRIGHT_MAX"))
        self.image_repository = ImageRepository()

    def create_image(self, pixels: np.ndarray, path: Union[str, Path] = None) -> Image:
        return self.image_repository.create_image(pixels, path)

    def load(self, path: str | Path) -> Image:
        """Load a single image from disk into an Image object."""
        return self.image_repository.load(path)

    def stream_gallery(
        self,
        folder: Union[str, Path],
        *,
        recursive: bool = False,
        exts: Iterable[str] | None = None,
    ) -> Iterator[Image]:
        """
        Yield images lazily instead of returning a gigantic list.
        """
        return self.image_repository.iter_dir(folder,
                                              recursive=recursive,
                                              exts=exts)

    # save_gallery can now accept *any* iterable
    def save_gallery(self, gallery: Iterable[Image]):
        for img in gallery:
            self.save(img)

    @staticmethod
    def _to_grayscale(img_pixels: np.ndarray):
        grayscale_pixels = cv2.cvtColor(img_pixels, cv2.COLOR_RGB2GRAY)
        return grayscale_pixels

    @staticmethod
    def _get_sharpness_score(grayscale_pixels: np.ndarray) -> float:
        """
        Args:
            grayscale_pixels (np.ndarray): A grayscale pixels representation of an image.

        Returns:
            (float): The Laplacian variance of that image.
        """
        return cv2.Laplacian(grayscale_pixels, cv2.CV_64F).var()

    def check_sharpness(self, img: Image):
        """
        Checks the sharpness of an image.

        Args:
            img (Image): An image object
        Returns:
            True if the image is sharper than a given threshold, False otherwise.
        """
        grayscale_pixels = self._to_grayscale(img.pixels)
        sharpness_score = self._get_sharpness_score(grayscale_pixels)
        is_sharp = sharpness_score > self.SHARPNESS_THR
        return is_sharp, sharpness_score

    def check_brightness(self, img: Image):
        grayscale_pixels = self._to_grayscale(img.pixels)
        mean_grayscale_img = grayscale_pixels.mean()
        is_bright = self.BRIGHT_MIN < mean_grayscale_img < self.BRIGHT_MAX
        return is_bright, mean_grayscale_img

    def update_pixels(self, image: Image, new_pixels: np.ndarray) -> None:
        """
        Business-level method to replace the current image pixels.
        """
        self.image_repository.set_pixels(image, new_pixels)
    
    def preserve_original_state(self, image: Image) -> None:
        """
        Preserve the current image state before processing pipeline.
        """
        self.image_repository.save_original_pixels(image)
    
    def apply_pipeline_modification(self, image: Image, new_pixels: np.ndarray) -> None:
        """
        Apply a pipeline modification while preserving original for comparison.
        """
        self.image_repository.update_pixels_preserve_original(image, new_pixels)

    def save(self, image: Image) -> None:
        """
        Business-level method to save the image to a specific path.
        """
        self.image_repository.save(image)

    def save_gallery(self, gallery: List[Image]):
        for img in gallery:
            self.save(img)

    def crop_pixels(self, img: Image, bound_r, bound_l, bound_t, bound_b):
        # Debug logging for crop bounds
        width = bound_r - bound_l
        height = bound_b - bound_t
        print(f"🔍 CROP DEBUG: bounds=({bound_l},{bound_t},{bound_r},{bound_b}) → width={width}, height={height}")
        
        if bound_l >= bound_r or bound_t >= bound_b:
            img_h, img_w = img.pixels.shape[:2]
            print(f"❌ INVALID CROP BOUNDS:")
            print(f"   Original image: {img_w}x{img_h}")
            print(f"   Crop bounds: left={bound_l}, right={bound_r}, top={bound_t}, bottom={bound_b}")
            print(f"   Resulting size: {width}x{height}")
            raise ValueError(f"Invalid crop bounds would create {width}x{height} image")
            
        return img.pixels[bound_t:bound_b, bound_l:bound_r].copy()

    def get_image_dimensions(self, img: Image):
        return self.image_repository.retrieve_image_dimensions(img)

    # ─── ML-friendly utilities ────────────────────────────────────────
    # Static transforms reused by several models
    _TO_TENSOR = T.ToTensor()
    _CENTER_224 = T.Compose([
        T.Resize(224, antialias=True),
        T.CenterCrop(224),
        T.ToTensor()
    ])

    def to_tensor(self, img: Image, device: str = "cpu") -> torch.Tensor:
        """
        Convert Image.pixels → (1,3,H,W) float tensor in [0,1].
        Ensures the NumPy array is C-contiguous so torch.from_numpy() accepts it.
        """
        np_img = img.pixels
        if not np_img.flags['C_CONTIGUOUS']:  # ① guard
            np_img = np.ascontiguousarray(np_img)  # ② fix

        return self._TO_TENSOR(np_img).unsqueeze(0).to(device)

    def center_crop_224(self, img: Image, device: str = "cpu") -> torch.Tensor:
        """
        Return a CLIP-ready crop tensor (1,3,224,224).
        """
        pil_obj = PILImage.fromarray(img.pixels)
        crop = self._CENTER_224(pil_obj).unsqueeze(0).to(device)
        return crop

    def apply_bcg(
            self,
            img: Image,
            bcg: BrightnessContrastGamma,
            device: str = "cpu",
    ) -> Image:
        """
        Apply a Brightness/Contrast/Gamma adjustment and return a *new* Image.
        """
        tensor = self.to_tensor(img, device)
        edited = bcg.apply_to_tensor(tensor).clamp(0, 1)

        np_img = (
            (edited.squeeze(0).cpu().numpy() * 255)
            .astype("uint8")
            .transpose(1, 2, 0)  # CHW → HWC
        )

        new_path = (
            img.path.with_stem(img.path.stem + "_edited") if img.path else None
        )
        return self.create_image(np_img, new_path)

    def to_pil_image(self, img: Image) -> PILImage.Image:
        """
        Convert Image.pixels → PIL Image object.
        Ensures the NumPy array is C-contiguous.
        """
        np_img = img.pixels
        if not np_img.flags['C_CONTIGUOUS']:
            np_img = np.ascontiguousarray(np_img)
        
        return PILImage.fromarray(np_img)

    def save_scored_gallery(self, scored_gallery: List[ScoredImage]):
        """
        Save a gallery of ScoredImage objects to disk.
        """
        for scored_img in scored_gallery:
            self.save(scored_img.image)


