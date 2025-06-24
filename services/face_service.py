from models.face import Face
from repositories.face_repository import FaceRepository
import numpy as np
from typing import Tuple


class FaceService:
    """
    Business logic layer for accessing face attributes.
    Delegates low-level retrieval to FaceRepository.
    """

    def __init__(self):
        self.repository = FaceRepository()

    def get_bbox_coordinates(self, face: Face) -> Tuple[int, int, int, int]:
        return self.repository.retrieve_bbox(face)

    def get_pose(self, face: Face) -> Tuple[float, float, float]:
        return self.repository.retrieve_pose(face)

    def get_detection_score(self, face: Face) -> float:
        return self.repository.retrieve_det_score(face)

    def get_embedding(self, face: Face) -> np.ndarray:
        return self.repository.retrieve_embedding(face)

    def get_bbox_dimensions(self, face: Face) -> [float, float]:
        """
        Gets a face, returns the bbox height and width
        :param face:
        :return:
        """
        left, top, right, bottom = self.get_bbox_coordinates(face)
        height = bottom - top
        width = right - left
        return height, width