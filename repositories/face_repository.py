from models.face import Face
import numpy as np
from typing import Tuple


class FaceRepository:
    """
    Simple access layer for Face model attributes.
    """

    @staticmethod
    def retrieve_bbox(face: Face) -> Tuple[int, int, int, int]:
        return face.bbox

    @staticmethod
    def retrieve_pose(face: Face) -> Tuple[float, float, float]:
        return face.pose

    @staticmethod
    def retrieve_det_score(face: Face) -> float:
        return face.det_score

    @staticmethod
    def retrieve_embedding(face: Face) -> np.ndarray:
        return face.embedding
