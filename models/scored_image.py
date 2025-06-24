from __future__ import annotations
from dataclasses import dataclass
from models.image import Image


@dataclass
class ScoredImage:
    """
    Data object containing an Image and its quality scores.
    Used for storing LinkedIn photo and attire scoring results.
    """
    image: Image
    linkedin_score: float  # LinkedIn photo quality score (0-100)
    attire_score: float    # Attire appropriateness score (0-100)
    face_neutrality_score: float  # Face neutrality score (0-100) - higher = more neutral
    final_score: float     # Combined final score (0-100) 