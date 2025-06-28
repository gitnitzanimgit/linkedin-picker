from typing import Tuple
import numpy as np
import cv2

from ..models.image import Image
from .segmentation_service import SegmentationService
from .face_analysis_service import FaceAnalysisService


class BackgroundService:
    """
    Business‑level helper for background replacement.

    • Uses SegmentationService to get the person mask.
    • Returns a **new** Image object (no path yet) containing the
      composited result, so your path semantics stay honest.
    """

    _DEFAULT_COLOR: Tuple[int, int, int] = (230, 230, 230)  # studio light‑gray

    def __init__(self):
        self.seg_service = SegmentationService()
        self.face_analysis_service = FaceAnalysisService()


    @staticmethod
    def _compose(
            fg: np.ndarray,
            alpha_u8: np.ndarray,
            bg: np.ndarray,
            radius: int = 6  # ←  edge‑softness control
    ) -> np.ndarray:
        """
        Alpha‑blend with adjustable feather radius.

        • radius = 4‑6  ‑‑ typical studio fade
        • radius = 8‑10 ‑‑ very soft DSLR‑like edge
        """
        if radius > 0:
            alpha_u8 = cv2.GaussianBlur(alpha_u8, (0, 0),
                                        sigmaX=radius, sigmaY=radius)

        alpha = alpha_u8.astype("float32") / 255.0
        alpha = cv2.merge([alpha, alpha, alpha])  # (H,W,3)

        return (fg.astype("float32") * alpha +
                bg.astype("float32") * (1.0 - alpha)).astype("uint8")

    # --------------------------------------------------------------
    def replace_with_blur(
            self,
            img: Image,
            blur_ratio: float = 0.2,  # 20 % of original size → *strong* blur
            thr: float = 0.5,
    ) -> np.ndarray:
        """
        Heavy portrait‑mode blur.

        • blur_ratio 0.2  →  downscale to 20 %, blur, upscale → creamy BG
        • Increase to 0.3‑0.4 for milder blur, 0.1 for extreme blur.
        """
        mask = self.seg_service.mask_person(img, thr=thr)
        mask = self._force_face_roi(img, mask, margin=0.12)

        # ---------- build blurred background ----------
        h, w = img.pixels.shape[:2]
        small = cv2.resize(img.pixels, None,
                           fx=blur_ratio, fy=blur_ratio,
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigmaX=5, sigmaY=5)
        blurred_bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

        out = self._compose(img.pixels, mask, blurred_bg, radius=20)
        return Image(pixels=out).pixels

    def _force_face_roi(
            self,
            img: Image,
            mask_u8: np.ndarray,
            margin: float = 0.12,  # 12 % padding around bbox
    ) -> np.ndarray:
        """
        Set mask to 255 inside the detected face bounding‑box (+padding).

        margin: fraction of bbox size to pad on all sides.
        """
        # we assume exactly one face (your earlier validation)
        face = self.face_analysis_service.get_faces(img)[0]
        x1, y1, x2, y2 = map(int, face.bbox)

        # pad bbox a bit
        w, h = x2 - x1, y2 - y1
        pad_x = int(w * margin)
        pad_y = int(h * margin)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(mask_u8.shape[1], x2 + pad_x)
        y2 = min(mask_u8.shape[0], y2 + pad_y)

        mask_u8[y1:y2, x1:x2] = 255
        return mask_u8
