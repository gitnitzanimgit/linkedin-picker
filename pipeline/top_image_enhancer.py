"""
Top Image Enhancer Pipeline
Performs final optimization on the top 5 highest-scoring LinkedIn photos.
This is the final step before saving the best images.
"""

import os
import uuid
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from models.scored_image import ScoredImage
from services.image_service import ImageService
from services.image_enhancement_service import ImageEnhancementService

# Load environment variables
load_dotenv()

# Enhanced images output directory
ENHANCED_DIR = os.getenv("ENHANCED_DIR_PATH", "data/enhanced_gallery")
OUTPUT_EXT = os.getenv("OUTPUT_IMG_EXT", ".jpg")


def enhance_top_images(
    top_scored_gallery: List[ScoredImage],
    *,
    image_service: ImageService = ImageService(),
    enhancement_service: ImageEnhancementService = ImageEnhancementService(),
    enhanced_dir: str | Path = ENHANCED_DIR,
    ext: str = OUTPUT_EXT
) -> List[ScoredImage]:
    """
    Apply final image enhancements to the top-scoring LinkedIn photos.
    
    This is the final pipeline step that:
    1. Takes the top 5 scored images
    2. Applies professional image enhancements (CLIP-guided lighting optimization)
    3. Updates pixels in-memory and sets final paths for saving
    4. Returns enhanced ScoredImage objects ready for final save
    
    Args:
        top_scored_gallery: List of top-scoring ScoredImage objects (max 5)
        image_service: Service for image operations
        enhancement_service: Service for image enhancement
        enhanced_dir: Directory to save enhanced images
        ext: File extension for enhanced images
        
    Returns:
        List[ScoredImage]: Enhanced images with updated paths
    """
    enhanced_dir = Path(enhanced_dir)
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    
    if not top_scored_gallery:
        print("No top images to enhance.")
        return []
    
    print(f"\n{'='*60}")
    print(f"🎨 ENHANCING TOP {len(top_scored_gallery)} IMAGES - FINAL STEP!")
    print(f"{'='*60}")
    
    for i, scored_img in enumerate(top_scored_gallery, 1):
        print(f"   🎨 Enhancing image {i}/{len(top_scored_gallery)}")
        
        # Apply enhancement to get new pixels
        enhanced_img = enhancement_service._optimise(scored_img.image)
        
        # Update pixels in-memory while preserving original
        image_service.apply_pipeline_modification(scored_img.image, enhanced_img.pixels)
        
        # Set final path for saving
        filename = f"enhanced_{uuid.uuid1().hex}{ext}"
        enhanced_path = enhanced_dir / filename
        scored_img.image.path = enhanced_path
        
        print(f"   ✨ Enhanced image {i} ready: {filename}")
    
    print(f"\n🏆 FINAL ENHANCEMENT COMPLETE!")
    print(f"   📁 Enhanced images will be saved to: {enhanced_dir}")
    print(f"   🎯 {len(top_scored_gallery)} images ready for LinkedIn!")
    print(f"{'='*60}\n")
    
    return top_scored_gallery


def log_final_results(enhanced_gallery: List[ScoredImage]) -> None:
    """
    Log the final enhanced image results.
    
    Args:
        enhanced_gallery: List of final enhanced ScoredImage objects
    """
    if not enhanced_gallery:
        print("No enhanced images to display.")
        return
    
    print(f"{'='*80}")
    print(f"🎯 FINAL LINKEDIN PHOTOS - READY TO USE!")
    print(f"{'='*80}")
    
    for i, scored_img in enumerate(enhanced_gallery, 1):
        path = scored_img.image.path
        filename = Path(path).name if path else "Unknown"
        
        print(f"{i}. 🏆 FINAL Score: {scored_img.final_score:6.2f} | "
              f"LinkedIn: {scored_img.linkedin_score:5.1f} | "
              f"Attire: {scored_img.attire_score:5.1f} | "
              f"Non-neutrality: {scored_img.face_neutrality_score:5.1f} | "
              f"File: {filename}")
    
    print(f"{'='*80}")
    print(f"🎉 SUCCESS! Your {len(enhanced_gallery)} best LinkedIn photos are ready!")
    print(f"📁 Find them in the enhanced_gallery directory")
    print(f"{'='*80}\n")