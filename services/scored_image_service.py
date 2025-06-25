import os
from typing import List
from pathlib import Path
from dotenv import load_dotenv

from models.scored_image import ScoredImage
from repositories.scored_image_repository import ScoredImageRepository

# Load environment variables
load_dotenv()


class ScoredImageService:
    """
    Service for business logic operations on ScoredImage collections.
    Handles professional photo filtering, quality assessment, and result logging.
    """
    
    def __init__(self):
        self.repository = ScoredImageRepository()
        
        # Load thresholds from environment variables
        self.min_non_neutrality = float(os.getenv("MIN_NON_NEUTRALITY_THRESHOLD", "50.0"))
        self.min_linkedin_score = float(os.getenv("MIN_LINKEDIN_THRESHOLD", "1.0"))
        self.min_attire_score = float(os.getenv("MIN_ATTIRE_THRESHOLD", "60.0"))
        self.min_final_score = float(os.getenv("MIN_FINAL_THRESHOLD", "1.0"))
    
    def filter_professional_images(
        self, 
        scored_images: List[ScoredImage],
        apply_non_neutrality_filter: bool = True,
        apply_linkedin_filter: bool = False,
        apply_attire_filter: bool = False,
        apply_final_filter: bool = False
    ) -> List[ScoredImage]:
        """
        Apply professional photo quality filters based on business rules.
        
        Args:
            scored_images: List of ScoredImage objects
            apply_non_neutrality_filter: Filter out overly neutral/robotic faces
            apply_linkedin_filter: Filter by minimum LinkedIn score
            apply_attire_filter: Filter by minimum attire score
            apply_final_filter: Filter by minimum final score
            
        Returns:
            List[ScoredImage]: Filtered list meeting professional standards
        """
        filtered_images = scored_images.copy()
        original_count = len(filtered_images)
        
        # Apply non-neutrality filter (remove too robotic faces)
        if apply_non_neutrality_filter:
            before_count = len(filtered_images)
            filtered_images = self.repository.filter_by_non_neutrality(
                filtered_images, self.min_non_neutrality
            )
            removed_count = before_count - len(filtered_images)
            if removed_count > 0:
                print(f" Filtered {removed_count} images with non-neutrality < {self.min_non_neutrality}% (too neutral/robotic)")
        
        # Apply LinkedIn quality filter
        if apply_linkedin_filter:
            before_count = len(filtered_images)
            filtered_images = self.repository.filter_by_linkedin_score(
                filtered_images, self.min_linkedin_score
            )
            removed_count = before_count - len(filtered_images)
            if removed_count > 0:
                print(f" Filtered {removed_count} images with LinkedIn score < {self.min_linkedin_score}")
        
        # Apply attire filter
        if apply_attire_filter:
            before_count = len(filtered_images)
            filtered_images = self.repository.filter_by_attire_score(
                filtered_images, self.min_attire_score
            )
            removed_count = before_count - len(filtered_images)
            if removed_count > 0:
                print(f" Filtered {removed_count} images with attire score < {self.min_attire_score}")
        
        # Apply final score filter
        if apply_final_filter:
            before_count = len(filtered_images)
            filtered_images = self.repository.filter_by_final_score(
                filtered_images, self.min_final_score
            )
            removed_count = before_count - len(filtered_images)
            if removed_count > 0:
                print(f" Filtered {removed_count} images with final score < {self.min_final_score}")
        
        total_removed = original_count - len(filtered_images)
        if total_removed > 0:
            print(f" Total: {total_removed} images filtered, {len(filtered_images)} remaining")
        
        return filtered_images
    
    def get_best_linkedin_photos(
        self, 
        scored_images: List[ScoredImage], 
        count: int = 5,
        apply_filters: bool = True
    ) -> List[ScoredImage]:
        """
        Get the best LinkedIn photos after applying quality filters.
        
        Args:
            scored_images: List of ScoredImage objects
            count: Number of top photos to return
            apply_filters: Whether to apply professional filters first
            
        Returns:
            List[ScoredImage]: Top photos for LinkedIn use
        """
        # Apply filters if requested
        if apply_filters:
            filtered_images = self.filter_professional_images(
                scored_images, 
                apply_non_neutrality_filter=True
            )
        else:
            filtered_images = scored_images
        
        # Get top N by final score
        top_photos = self.repository.get_top_n(filtered_images, count, "final_score")
        
        return top_photos
    
    def keep_only_top_n(
        self, 
        scored_images: List[ScoredImage], 
        n: int = 5
    ) -> List[ScoredImage]:
        """
        Keep only the top N highest-scoring images.
        Used for maintaining a rolling "best of" collection.
        
        Args:
            scored_images: List of ScoredImage objects
            n: Number of top images to keep (default 5)
            
        Returns:
            List[ScoredImage]: Top N images only
        """
        if len(scored_images) <= n:
            return scored_images
        
        # Sort by final score and keep only top N
        top_images = self.repository.get_top_n(scored_images, n, "final_score")
        
        # Log which images were kept vs discarded
        discarded_count = len(scored_images) - len(top_images)
        if discarded_count > 0:
            print(f" Keeping top {n} images, discarded {discarded_count} lower-scoring images")
        
        return top_images
    
    def apply_quality_filters(
        self, 
        scored_images: List[ScoredImage]
    ) -> List[ScoredImage]:
        """
        Apply all quality filters with current environment settings.
        
        Args:
            scored_images: List of ScoredImage objects
            
        Returns:
            List[ScoredImage]: Filtered images meeting all quality standards
        """
        return self.filter_professional_images(
            scored_images,
            apply_non_neutrality_filter=True,
            apply_linkedin_filter=False,  # Usually too restrictive
            apply_attire_filter=False,    # Usually too restrictive  
            apply_final_filter=False      # Usually too restrictive
        )
    
    def log_scoring_results(
        self, 
        scored_images: List[ScoredImage], 
        top_n: int = 5,
        title: str = "SCORING RESULTS"
    ) -> None:
        """
        Log detailed scoring results for all images.
        
        Args:
            scored_images: List of ScoredImage objects
            top_n: Number of top images to highlight
            title: Title for the results section
        """
        if not scored_images:
            print("No images were scored.")
            return
        
        # Sort by final score
        sorted_images = self.repository.sort_by_final_score(scored_images)
        
        print(f"\n{'='*80}")
        print(f"{title} - ALL {len(scored_images)} IMAGES:")
        print(f"{'='*80}")
        
        # Print all images with their scores
        for i, scored_img in enumerate(sorted_images, 1):
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
        
        # Get statistics
        stats = self.repository.get_score_statistics(scored_images)
        
        print(f"{'='*80}")
        print(f"MODEL OUTPUT RANGES:")
        print(f"LinkedIn Model Raw Range: {stats['linkedin']['min']/100:.4f} - {stats['linkedin']['max']/100:.4f}")
        print(f"CLIP Attire Raw Range: {stats['attire']['min']/100:.4f} - {stats['attire']['max']/100:.4f}")
        print(f"CLIP Non-neutrality Raw Range: {stats['non_neutrality']['min']/100:.4f} - {stats['non_neutrality']['max']/100:.4f}")
        print(f"{'='*80}")
        print(f"Total images scored: {len(scored_images)}")
        print(f"Top {top_n} images are highlighted with ")
        print(f"{'='*80}\n")
    
    def get_filter_settings(self) -> dict:
        """
        Get current filter threshold settings.
        
        Returns:
            dict: Current threshold values
        """
        return {
            "min_non_neutrality": self.min_non_neutrality,
            "min_linkedin_score": self.min_linkedin_score,
            "min_attire_score": self.min_attire_score,
            "min_final_score": self.min_final_score
        }