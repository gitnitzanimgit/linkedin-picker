from __future__ import annotations

from pathlib import Path
from typing import List, Union

import torch
from tqdm import trange

from models.image import Image
from models.image_adjustments import BrightnessContrastGamma
from models.clip_model import ClipModel
from services.image_service import ImageService


class ImageEnhancementService:
    """
    Finds brightness/contrast/gamma that maximise CLIP similarity
    to a “well-lit headshot” prompt for each input image.
    """

    PROMPT = ("Subject studio-style lit, face clear and crisp, "
              "Overall image brightness is high")

    # hyper-params
    LR_START = 0.005
    LR_FACTOR = 0.10
    LR_PATIENCE = 50
    STOP_PAT = 100
    EPS = 1e-4
    MAX_STEPS = 200

    def __init__(self):
        self.clip = ClipModel()
        self.txt_feat = self.clip.encode_text(self.PROMPT)
        self.img_svc = ImageService()

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
        opt = torch.optim.SGD(params, lr=self.LR_START, momentum=0.9)

        best = self._sim(crop).item()
        lr_now = self.LR_START
        no_imp = no_imp_lr = 0
        trace = [best]

        for _ in trange(self.MAX_STEPS, desc="optim", ncols=70):
            opt.zero_grad()
            sim = self._sim(BrightnessContrastGamma._apply_raw(crop, p_b, p_c, p_g))
            (-sim).backward()
            opt.step()

            score = sim.item()
            trace.append(score)

            if score > best + self.EPS:
                best, no_imp, no_imp_lr = score, 0, 0
            else:
                no_imp += 1
                no_imp_lr += 1

            if no_imp_lr >= self.LR_PATIENCE:
                lr_now *= self.LR_FACTOR
                for g in opt.param_groups:
                    g["lr"] = lr_now
                no_imp_lr = 0

            if no_imp >= self.STOP_PAT:
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
            print(f"   🎨 Enhancing image {i}/{len(images)}")
            enhanced_img = self._optimise(img)
            enhanced_images.append(enhanced_img)
        
        return enhanced_images
