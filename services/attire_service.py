from __future__ import annotations
import torch
import open_clip
from models.image import Image
from services.image_service import ImageService
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


class AttireService:
    """
    Service for evaluating attire appropriateness using OpenCLIP model.
    *   No I/O here—works only with Image objects (RGB numpy arrays).
    *   Loads model once and caches it for efficient batch processing.
    """
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        # Use MPS on Apple Silicon, CUDA on NVIDIA, CPU otherwise
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.image_service = ImageService()
        self._load_model()
    
    def _load_model(self):
        """
        Load the OpenCLIP model and tokenizer for attire evaluation.
        """
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, 
            pretrained=self.pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.to(self.device).eval()
    
    def score_image(self, img: Image) -> float:
        """
        Score a single image for attire appropriateness.
        
        Args:
            img (Image): An image object to score
            
        Returns:
            float: Appropriateness score between 0 and 100 (higher is better)
        """
        # Use ImageService's PIL conversion for consistency
        pil_img = self.image_service.to_pil_image(img).convert("RGB")
        
        # Define prompts for appropriate vs inappropriate attire
        prompts = [
            "a person wearing normal clothes for a profile photo, like an unwrinkled and suitable t-shirt or a jacket",
            "a person wearing strange, revealing, or inappropriate clothes for a profile photo, like a costume, tank top, or large v-neck"
        ]
        
        # Tokenize prompts
        tokens = self.tokenizer(prompts).to(self.device)
        
        # Preprocess image
        image_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            img_feat = self.model.encode_image(image_tensor)
            txt_feat = self.model.encode_text(tokens)
            
            # Normalize features
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
            
            # Calculate similarities
            sims = (img_feat @ txt_feat.T) * 100
            probs = sims.softmax(dim=-1)
        
        # Get raw appropriateness score (probability of appropriate attire)
        raw_score = probs[0][0].item()
        scaled_score = raw_score * 100
        
        # Debug: Print raw and scaled scores
        print(f"CLIP Attire - Raw: {raw_score:.4f}, Scaled: {scaled_score:.2f}")
        
        return scaled_score
    
    def is_appropriate_attire(self, img: Image, threshold: float = 60.0) -> bool:
        """
        Check if an image has appropriate attire for a LinkedIn profile.
        
        Args:
            img (Image): An image object to check
            threshold (float): Minimum score threshold (0-100). Default 60.
            
        Returns:
            bool: True if attire is appropriate, False otherwise
        """
        score = self.score_image(img)
        return score >= threshold

    def check_attire_appropriateness(self, img: Image, threshold: float = 60.0):
        """
        Check attire appropriateness and return both score and pass/fail result.
        Follows the pattern of other services like check_sharpness().
        
        Args:
            img (Image): An image object to check
            threshold (float): Minimum score threshold (0-100). Default 60.
            
        Returns:
            tuple: (is_appropriate, score)
        """
        score = self.score_image(img)
        is_appropriate = score >= threshold
        return is_appropriate, score 