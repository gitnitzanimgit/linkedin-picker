from __future__ import annotations
import torch
from pathlib import Path
from torchvision import transforms, models
from models.image import Image
from services.image_service import ImageService
from services.cropping_service import CroppingService
from services.face_analysis_service import FaceAnalysisService
import open_clip
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
        self.model_path = Path(model_path or os.getenv("LINKEDIN_MODEL_PATH", "linkedin_resnet18_cost_min.pth"))
        self.model = None
        self.preprocess = None
        self.threshold = None
        self.image_service = ImageService()
        self.cropping_service = CroppingService()
        self.face_analysis_service = FaceAnalysisService()
        self._load_model()
        self._load_clip_model()
    
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
    
    def _load_clip_model(self):
        """
        Load CLIP model for face neutrality detection.
        """
        # Use MPS on Apple Silicon, CUDA on NVIDIA, CPU otherwise
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", 
            pretrained="laion2b_s34b_b79k"
        )
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.clip_model.to(self.device).eval()
    
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
        
        # Use FFHQ-style face cropping (proper face model padding)
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
        pil_img = self.image_service.to_pil_image(img).convert("RGB")

        prompts = [
            # index 0 – neutral (positive for LinkedIn)
            "A completely neutral facial expression",
            # index 1 – non-neutral (smile, frown, surprise …)
            "An even slightly non-neutral facial expression",
        ]

        tokens = self.clip_tokenizer(prompts).to(self.device)
        image_tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_feat = self.clip_model.encode_image(image_tensor)
            txt_feat = self.clip_model.encode_text(tokens)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
            sims = (img_feat @ txt_feat.T) * 100          # (1×2) logits
            probs = sims.softmax(dim=-1)                  # → probabilities

        raw_non_neutral = probs[0, 1].item()              # probability the face is non-neutral
        scaled_score    = raw_non_neutral * 100           # 0‥100

        print(
            f"Non-neutrality score: {scaled_score:.1f}%"
        )
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