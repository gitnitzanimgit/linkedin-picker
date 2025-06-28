# pipeline/final_scorer.py
from pathlib import Path
import os
import uuid
from typing import List

from dotenv import load_dotenv

from ..models.image import Image
from ..models.scored_image import ScoredImage
from ..services.linkedin_photo_service import LinkedInPhotoService
from ..services.attire_service import AttireService
from ..services.image_service import ImageService
from ..services.scored_image_service import ScoredImageService

# env‑vars
load_dotenv()
SCORED_DIR = os.getenv("SCORED_DIR_PATH", "data/scored_gallery")
OUTPUT_EXT = os.getenv("OUTPUT_IMG_EXT", ".jpg")

# Default thresholds
LINKEDIN_THRESHOLD = float(os.getenv("LINKEDIN_THRESHOLD", "60.0"))
ATTIRE_THRESHOLD = float(os.getenv("ATTIRE_THRESHOLD", "60.0"))

def score_final_photos(
    gallery: List[Image],
    *,
    linkedin_photo_service: LinkedInPhotoService = LinkedInPhotoService(),
    attire_service: AttireService = AttireService(),
    image_service: ImageService = ImageService(),
    scored_dir: str | Path = SCORED_DIR,
    ext: str = OUTPUT_EXT,
    linkedin_threshold: float = LINKEDIN_THRESHOLD,
    attire_threshold: float = ATTIRE_THRESHOLD,
) -> List[ScoredImage]:
    """
    For every Image in *gallery*:
        • score LinkedIn photo quality (0-100)
        • score attire appropriateness (0-100)
        • calculate final score as (linkedin_score × attire_score) / 100
        • create ScoredImage objects with unique paths
        • log top 5 scoring images
    Returns the list of ScoredImage objects (no threshold filtering for now).
    """
    scored_dir = Path(scored_dir)
    scored_dir.mkdir(parents=True, exist_ok=True)

    final_scored_gallery: List[ScoredImage] = []

    for img in gallery:
        # 1. Score LinkedIn photo quality
        linkedin_score = linkedin_photo_service.score_image(img)
        
        # 2. Score attire appropriateness
        attire_score = attire_service.score_image(img)
        
        # 3. Score face neutrality (NEW - not included in final score yet)
        face_neutrality_score = linkedin_photo_service.score_face_neutrality(img)
        
        # 4. Calculate final score (multiplicative) - NO THRESHOLD FILTERING
        final_score = (linkedin_score * attire_score) / 100
        
        # 5. Construct unique filename using ImageService pattern
        filename = f"{uuid.uuid1().hex}{ext}"
        out_path = scored_dir / filename
        
        # 6. Create new Image with unique path using ImageService
        new_img = image_service.create_image(img.pixels, out_path)
        new_img.original_pixels = img.original_pixels  # Preserve original pixels through pipeline
        
        # 7. Create ScoredImage object
        scored_img = ScoredImage(
            image=new_img,
            linkedin_score=linkedin_score,
            attire_score=attire_score,
            face_neutrality_score=face_neutrality_score,
            final_score=final_score
        )
        
        final_scored_gallery.append(scored_img)

    # Apply professional photo filters and keep only top 5
    scored_image_service = ScoredImageService()
    filtered_gallery = scored_image_service.apply_quality_filters(final_scored_gallery)
    top_5_gallery = scored_image_service.keep_only_top_n(filtered_gallery, n=5)
    scored_image_service.log_scoring_results(top_5_gallery)

    return top_5_gallery


def _log_top_scoring_images(scored_gallery: List[ScoredImage], top_n: int = 5):
    """
    Log the paths and scores of all images.
    
    Args:
        scored_gallery: List of ScoredImage objects
        top_n: Number of top images to highlight (default 5)
    """
    if not scored_gallery:
        print("No images were scored.")
        return
    
    # Sort by final score in descending order
    sorted_gallery = sorted(scored_gallery, key=lambda x: x.final_score, reverse=True)

    # Print all images with their scores
    for i, scored_img in enumerate(sorted_gallery, 1):
        path = scored_img.image.path or "No path"
        filename = Path(path).name if path else "Unknown"

        # Highlight top N images
        if i <= top_n:
            print(f"{i:2d}.  Score: {scored_img.final_score:6.2f} | "
                  f"LinkedIn: {scored_img.linkedin_score:5.1f} | "
                  f"Attire: {scored_img.attire_score:5.1f} | "
                  f"Non-neutrality: {scored_img.face_neutrality_score:5.1f} | "
                  f"File: {filename}")
        else:
            print(f"{i:2d}.    Score: {scored_img.final_score:6.2f} | "
                  f"LinkedIn: {scored_img.linkedin_score:5.1f} | "
                  f"Attire: {scored_img.attire_score:5.1f} | "
                  f"Non-neutrality: {scored_img.face_neutrality_score:5.1f} | "
                  f"File: {filename}")
    
    # Calculate ranges for raw outputs (divide by 100 to get raw values)
    linkedin_raw_scores = [score.linkedin_score / 100 for score in scored_gallery]
    attire_raw_scores = [score.attire_score / 100 for score in scored_gallery]
    face_neutrality_raw_scores = [score.face_neutrality_score / 100 for score in scored_gallery]
    
    print(f"{'='*80}")
    print(f"MODEL OUTPUT RANGES:")
    print(f"LinkedIn Model Raw Range: {min(linkedin_raw_scores):.4f} - {max(linkedin_raw_scores):.4f}")
    print(f"CLIP Attire Raw Range: {min(attire_raw_scores):.4f} - {max(attire_raw_scores):.4f}")
    print(f"CLIP Non-neutrality Raw Range: {min(face_neutrality_raw_scores):.4f} - {max(face_neutrality_raw_scores):.4f}")
    print(f"{'='*80}")
    print(f"Total images scored: {len(scored_gallery)}")
    print(f"Top {top_n} images are highlighted with ")
    print(f"{'='*80}\n") 