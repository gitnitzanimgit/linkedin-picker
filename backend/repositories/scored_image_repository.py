from typing import List
from ..models.scored_image import ScoredImage


class ScoredImageRepository:
    """
    Repository for data operations on ScoredImage collections.
    Handles filtering, sorting, and selection operations.
    """
    
    def filter_by_non_neutrality(
        self, 
        scored_images: List[ScoredImage], 
        min_threshold: float
    ) -> List[ScoredImage]:
        """
        Filter images by minimum non-neutrality score.
        
        Args:
            scored_images: List of ScoredImage objects
            min_threshold: Minimum non-neutrality score (images below this are too neutral/robotic)
            
        Returns:
            List[ScoredImage]: Filtered list with appropriately expressive images
        """
        return [
            scored_img for scored_img in scored_images 
            if scored_img.face_neutrality_score >= min_threshold
        ]
    
    def filter_by_linkedin_score(
        self, 
        scored_images: List[ScoredImage], 
        min_threshold: float
    ) -> List[ScoredImage]:
        """
        Filter images by minimum LinkedIn photo quality score.
        
        Args:
            scored_images: List of ScoredImage objects
            min_threshold: Minimum LinkedIn score threshold
            
        Returns:
            List[ScoredImage]: Filtered list with good LinkedIn quality
        """
        return [
            scored_img for scored_img in scored_images 
            if scored_img.linkedin_score >= min_threshold
        ]
    
    def filter_by_attire_score(
        self, 
        scored_images: List[ScoredImage], 
        min_threshold: float
    ) -> List[ScoredImage]:
        """
        Filter images by minimum attire appropriateness score.
        
        Args:
            scored_images: List of ScoredImage objects
            min_threshold: Minimum attire score threshold
            
        Returns:
            List[ScoredImage]: Filtered list with appropriate attire
        """
        return [
            scored_img for scored_img in scored_images 
            if scored_img.attire_score >= min_threshold
        ]
    
    def filter_by_final_score(
        self, 
        scored_images: List[ScoredImage], 
        min_threshold: float
    ) -> List[ScoredImage]:
        """
        Filter images by minimum final (combined) score.
        
        Args:
            scored_images: List of ScoredImage objects
            min_threshold: Minimum final score threshold
            
        Returns:
            List[ScoredImage]: Filtered list with good overall scores
        """
        return [
            scored_img for scored_img in scored_images 
            if scored_img.final_score >= min_threshold
        ]
    
    def sort_by_final_score(
        self, 
        scored_images: List[ScoredImage], 
        descending: bool = True
    ) -> List[ScoredImage]:
        """
        Sort images by final score.
        
        Args:
            scored_images: List of ScoredImage objects
            descending: If True, sort highest to lowest (default)
            
        Returns:
            List[ScoredImage]: Sorted list
        """
        return sorted(scored_images, key=lambda x: x.final_score, reverse=descending)
    
    def sort_by_linkedin_score(
        self, 
        scored_images: List[ScoredImage], 
        descending: bool = True
    ) -> List[ScoredImage]:
        """
        Sort images by LinkedIn score.
        
        Args:
            scored_images: List of ScoredImage objects
            descending: If True, sort highest to lowest (default)
            
        Returns:
            List[ScoredImage]: Sorted list
        """
        return sorted(scored_images, key=lambda x: x.linkedin_score, reverse=descending)
    
    def get_top_n(
        self, 
        scored_images: List[ScoredImage], 
        n: int,
        sort_by: str = "final_score"
    ) -> List[ScoredImage]:
        """
        Get top N images by specified score.
        
        Args:
            scored_images: List of ScoredImage objects
            n: Number of top images to return
            sort_by: Score to sort by ("final_score", "linkedin_score", "attire_score")
            
        Returns:
            List[ScoredImage]: Top N images
        """
        if sort_by == "final_score":
            sorted_images = self.sort_by_final_score(scored_images)
        elif sort_by == "linkedin_score":
            sorted_images = self.sort_by_linkedin_score(scored_images)
        elif sort_by == "attire_score":
            sorted_images = sorted(scored_images, key=lambda x: x.attire_score, reverse=True)
        else:
            raise ValueError(f"Invalid sort_by parameter: {sort_by}")
        
        return sorted_images[:n]
    
    def get_score_statistics(self, scored_images: List[ScoredImage]) -> dict:
        """
        Calculate statistics for all score types.
        
        Args:
            scored_images: List of ScoredImage objects
            
        Returns:
            dict: Statistics including min, max, mean for each score type
        """
        if not scored_images:
            return {}
        
        linkedin_scores = [img.linkedin_score for img in scored_images]
        attire_scores = [img.attire_score for img in scored_images]
        neutrality_scores = [img.face_neutrality_score for img in scored_images]
        final_scores = [img.final_score for img in scored_images]
        
        return {
            "count": len(scored_images),
            "linkedin": {
                "min": min(linkedin_scores),
                "max": max(linkedin_scores),
                "mean": sum(linkedin_scores) / len(linkedin_scores)
            },
            "attire": {
                "min": min(attire_scores),
                "max": max(attire_scores),
                "mean": sum(attire_scores) / len(attire_scores)
            },
            "non_neutrality": {
                "min": min(neutrality_scores),
                "max": max(neutrality_scores),
                "mean": sum(neutrality_scores) / len(neutrality_scores)
            },
            "final": {
                "min": min(final_scores),
                "max": max(final_scores),
                "mean": sum(final_scores) / len(final_scores)
            }
        }