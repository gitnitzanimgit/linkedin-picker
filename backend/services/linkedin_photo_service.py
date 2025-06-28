from __future__ import annotations
import torch
from pathlib import Path
from torchvision import transforms, models
from ..models.image import Image
from ..models.clip_model import ClipModel
from .image_service import ImageService
from .cropping_service import CroppingService
from .face_analysis_service import FaceAnalysisService
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


class LinkedInPhotoService:
    """
    Service for evaluating LinkedIn photo quality using a pre-trained ResNet18 model.
    *   No I/O here—works only with Image objects (RGB numpy arrays).
    *   Loads model once and caches it for efficient batch processing.
    """
    
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path or os.getenv("LINKEDIN_MODEL_PATH", "backend/models/linkedin_resnet18_cost_min.pth"))
        self.model = None
        self.preprocess = None
        self.threshold = None
        self.image_service = ImageService()
        self.cropping_service = CroppingService()
        self.face_analysis_service = FaceAnalysisService()
        self.clip_model = ClipModel()
        self._load_model()
    
    def _load_model(self):
        """
        Load the pre-trained LinkedIn photo quality model and preprocessing pipeline.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"LinkedIn model not found at {self.model_path}")
        
        # Load checkpoint
        ckpt = torch.load(self.model_path, map_location="cpu")
        
        # Initialize model
        self.model = models.resnet18()
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)
        self.model.load_state_dict(ckpt["state_dict"], strict=False)
        self.model.eval()
        
        # Set preprocessing and threshold
        self.threshold = ckpt["threshold"]
        img_mean = ckpt["mean"]
        img_std = ckpt["std"]
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(img_mean, img_std)
        ])
    
    
    def score_image(self, img: Image) -> float:
        """
        Score a single image for LinkedIn photo quality.
        Uses proper face cropping based on FFHQ preprocessing.
        
        Args:
            img (Image): An image object to score
            
        Returns:
            float: Quality score between 0 and 100 (higher is better)
        """
        # Get faces in the image
        faces = self.face_analysis_service.get_faces(img)
        
        if not faces:
            print("No faces detected for LinkedIn scoring")
            return 0.0
        
        # Use the largest face for scoring
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        # Default PAD_FACTOR_FACE_MODEL if not set in env
        pad_factor_face_model = float(os.getenv("PAD_FACTOR_FACE_MODEL", "0.3"))
        
        face_pixels = self.cropping_service.crop_image_around_face(
            img, largest_face,
            pad_top_factor=pad_factor_face_model + 0.15,
            pad_bottom_factor=pad_factor_face_model - 0.05,
            pad_side_factor=pad_factor_face_model
        )
        face_img = self.image_service.create_image(face_pixels)
        
        # Use ImageService's tensor conversion for consistency
        x = self.preprocess(self.image_service.to_pil_image(face_img)).unsqueeze(0)
        
        # Preprocess and predict
        with torch.no_grad():
            raw_score = torch.sigmoid(self.model(x)).item()
        
        # Convert to 0-100 scale
        scaled_score = raw_score * 100
        
        # Debug: Print raw and scaled scores
        print(f"LinkedIn Model - Raw: {raw_score:.4f}, Scaled: {scaled_score:.2f}")
        
        return scaled_score
    
    def score_face_neutrality(self, img: Image) -> float:
        """
        Soft-max CLIP head.
        Returns probability [%] that the face shows a *non-neutral* expression.
        Higher = less neutral  (worse for LinkedIn-style portraits).
        """
        prompts = [
            # index 0 – neutral (positive for LinkedIn)
            "A completely neutral facial expression",
            # index 1 – non-neutral (smile, frown, surprise …)
            "An even slightly non-neutral facial expression",
        ]

        # Convert image to CLIP-ready tensor (224x224)
        img_tensor = self.image_service.center_crop_224(img, self.clip_model.device)

        with torch.no_grad():
            img_feat = self.clip_model.encode_image(img_tensor)
            txt_feat_neutral = self.clip_model.encode_text(prompts[0])  
            txt_feat_non_neutral = self.clip_model.encode_text(prompts[1])
            txt_feat = torch.cat([txt_feat_neutral, txt_feat_non_neutral], dim=0)
            
            sims = (img_feat @ txt_feat.T) * 100          # (1×2) logits
            probs = sims.softmax(dim=-1)                  # → probabilities

        raw_non_neutral = probs[0, 1].item()              # probability the face is non-neutral
        scaled_score    = raw_non_neutral * 100           # 0‥100

        return scaled_score
    
    def is_good_quality(self, img: Image, threshold: float | None = None) -> bool:
        """
        Check if an image meets the LinkedIn photo quality threshold.
        
        Args:
            img (Image): An image object to check
            threshold (float, optional): Custom threshold (0-100). Uses model default if None.
            
        Returns:
            bool: True if image meets quality threshold, False otherwise
        """
        if threshold is None:
            threshold = self.threshold * 100
        
        score = self.score_image(img)
        return score >= threshold

    def check_linkedin_quality(self, img: Image, threshold: float | None = None):
        """
        Check LinkedIn photo quality and return both score and pass/fail result.
        Follows the pattern of other services like check_sharpness().
        
        Args:
            img (Image): An image object to check
            threshold (float, optional): Custom threshold (0-100). Uses model default if None.
            
        Returns:
            tuple: (is_good_quality, score)
        """
        if threshold is None:
            threshold = self.threshold * 100
        
        score = self.score_image(img)
        is_good = score >= threshold
        return is_good, score 