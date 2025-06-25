from __future__ import annotations

from pathlib import Path
from typing import List, Union
import os
import logging

import torch
from tqdm import trange
from dotenv import load_dotenv

from models.image import Image
from models.image_adjustments import BrightnessContrastGamma
from models.clip_model import ClipModel
from services.image_service import ImageService

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ImageEnhancementService:
    """
    Finds brightness/contrast/gamma that maximise CLIP similarity
    to a "well-lit headshot" prompt for each input image.
    *   No I/O here—works only with Image objects (RGB numpy arrays).
    *   Uses environment variables for configuration.
    """

    def __init__(self, 
                 clip_arch: str = None,
                 clip_pretrained: str = None,
                 prompt: str = None):
        """
        Initialize the ImageEnhancementService with configurable parameters.
        
        Args:
            clip_arch: CLIP model architecture (defaults to env var)
            clip_pretrained: CLIP pretrained weights (defaults to env var)  
            prompt: Enhancement prompt (defaults to env var)
        """
        # Load configuration from environment variables
        self.prompt = prompt or os.getenv("ENHANCEMENT_PROMPT", 
            "Subject studio-style lit, face clear and crisp, Overall image brightness is high")
        
        # Optimization hyperparameters
        self.lr_start = float(os.getenv("ENHANCEMENT_LR_START", "0.005"))
        self.lr_factor = float(os.getenv("ENHANCEMENT_LR_FACTOR", "0.10"))
        self.lr_patience = int(os.getenv("ENHANCEMENT_LR_PATIENCE", "50"))
        self.stop_patience = int(os.getenv("ENHANCEMENT_STOP_PATIENCE", "100"))
        self.eps = float(os.getenv("ENHANCEMENT_EPS", "1e-4"))
        self.max_steps = int(os.getenv("ENHANCEMENT_MAX_STEPS", "200"))
        
        # Initialize models (use defaults to maintain caching)
        if clip_arch is None and clip_pretrained is None:
            # Use default ClipModel for caching
            self.clip = ClipModel()
        else:
            # Only override if explicitly requested
            clip_arch = clip_arch or os.getenv("CLIP_MODEL_ARCH", "ViT-B-32")
            clip_pretrained = clip_pretrained or os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")
            self.clip = ClipModel(arch=clip_arch, ckpt=clip_pretrained)
        self.txt_feat = self.clip.encode_text(self.prompt)
        self.img_svc = ImageService()
        
        logger.info(f"ImageEnhancementService initialized with prompt: {self.prompt[:50]}...")

    # ─── Public API ────────────────────────────────────────────────
    def optimise_file(self, path: Union[str, Path]) -> Path:
        img = self.img_svc.load(path)
        edited = self._optimise(img)
        self.img_svc.save(edited)
        return edited.path

    def optimise_files(self, paths: List[Union[str, Path]]):
        return [self.optimise_file(p) for p in paths]

    # ─── Internal helpers ──────────────────────────────────────────
    def _sim(self, tensor):
        img_feat = self.clip.encode_image(tensor)
        return torch.cosine_similarity(img_feat, self.txt_feat, dim=1)[0]

    def _optimise(self, img: Image) -> Image:
        device = self.clip.device
        crop = self.img_svc.center_crop_224(img, device)

        bcg = BrightnessContrastGamma()
        p_b, p_c, p_g = bcg.as_parameters(device)
        params = [p_b, p_c, p_g]
        opt = torch.optim.SGD(params, lr=self.lr_start, momentum=0.9)

        best = self._sim(crop).item()
        lr_now = self.lr_start
        no_imp = no_imp_lr = 0
        trace = [best]

        for _ in trange(self.max_steps, desc="optim", ncols=70):
            opt.zero_grad()
            sim = self._sim(BrightnessContrastGamma._apply_raw(crop, p_b, p_c, p_g))
            (-sim).backward()
            opt.step()

            score = sim.item()
            trace.append(score)

            if score > best + self.eps:
                best, no_imp, no_imp_lr = score, 0, 0
            else:
                no_imp += 1
                no_imp_lr += 1

            if no_imp_lr >= self.lr_patience:
                lr_now *= self.lr_factor
                for g in opt.param_groups:
                    g["lr"] = lr_now
                no_imp_lr = 0

            if no_imp >= self.stop_patience:
                break

        # Apply changes
        bcg = BrightnessContrastGamma.from_raw(p_b, p_c, p_g)
        edited_img = self.img_svc.apply_bcg(img, bcg, device=device)

        # Debug plotting removed to prevent threading issues

        return edited_img
    
    def enhance_top_images(self, images: List[Image]) -> List[Image]:
        """
        Enhance a list of top-scoring images for LinkedIn use.
        
        Args:
            images: List of Image objects to enhance
            
        Returns:
            List[Image]: Enhanced images optimized for LinkedIn
        """
        enhanced_images = []
        
        for i, img in enumerate(images, 1):
            print(f"   Enhancing image {i}/{len(images)}")
            enhanced_img = self._optimise(img)
            enhanced_images.append(enhanced_img)
        
        return enhanced_images