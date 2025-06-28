from __future__ import annotations
import numpy as np
from ..models.image import Image
from ..models.face_engine import FaceEngine
from ..models.face import Face
from .face_engine_service import FaceEngineService
from ..repositories.face_repository import FaceRepository
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)

# Debug: Check if logger is working
print(f"Logger name: {logger.name}")
print(f"Logger level: {logger.level}")
print(f"Logger effective level: {logger.getEffectiveLevel()}")
print(f"Logger handlers: {logger.handlers}")
logger.info("Test log message from face_analysis_service")

# Load environment variables
load_dotenv()


class FaceAnalysisService:
    """
    High-level face logic on top of a heavy `FaceEngine`.
    *   No I/O here—works only with Image objects (RGB numpy arrays).
    *   Caches detection results per Image instance to avoid repeats.
    """
    def __init__(self, engine: FaceEngine | None = None):
        self.face_engine_service = FaceEngineService()        # singleton model
        self.face_repository = FaceRepository()
        self._cache: dict[int, list] = {}             # id(img) → faces
        self.DET_CONF_THR = float(os.getenv("DET_CONF_THR"))
        self.AREA_MIN = float(os.getenv("AREA_MIN"))
        self.AREA_MAX = float(os.getenv("AREA_MAX"))
        self.POSE_YAW_PITCH = float(os.getenv("POSE_YAW_PITCH"))
        self.POSE_ROLL = float(os.getenv("POSE_ROLL"))

    def _faces(self, img: Image):
        """
        Receives an Image object and returns the face embeddings of that image.
        """
        k = id(img)
        if k not in self._cache:
            self._cache[k] = self.face_engine_service.detect_faces(img)
        return self._cache[k]

    def num_faces(self, img: Image) -> int:
        """
        Receives an Image object and returns the number of faces in that image.
        """
        return len(self._faces(img))

    def has_single_face(self, img: Image) -> bool:
        """
        Checks if an image contains a single face.
        Args:
            img (Image): An image object

        Returns:
            True if img contains a single face, False otherwise.
        """
        return self.num_faces(img) == 1

    def get_faces(self, img: Image):
        """
        Returns list of detected faces in an image.
        """
        return self._faces(img)

    # ------------------------- embeddings / ID -------------------------
    def _get_face_embedding(self, face: Face):
        return self.face_repository.retrieve_embedding(face)

    @staticmethod
    def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
        logger.debug(f"Cosine similarity: {float((v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))}")
        return float((v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def _similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        return self._cosine(embedding_a, embedding_b)

    def is_same_person(self, face_a: Face, face_b: Face, thr: float = None) -> bool:
        if thr is None:
            thr = float(os.getenv("FACE_SIMILARITY_THRESHOLD", "0.3"))
        face_a_embedding = self._get_face_embedding(face_a)
        face_b_embedding = self._get_face_embedding(face_b)
        return self._similarity(face_a_embedding, face_b_embedding) >= thr

    # -- individual checks (private) -----------------------------------
    def check_size_for_ref(self, face: Face, img: Image):
        """
        Receives an image and a face embedding, checks if the size is good for a reference image.

        Args:
            face (Face): A face embedding
            img (Image): An image object

        Returns:
            True if the face to image ratio good enough for a reference image, False otherwise.
        """
        img_height, img_width = img.pixels.shape[:2]
        face_height, face_width = face.bbox[3] - face.bbox[1], face.bbox[2] - face.bbox[0]
        area_ratio = (face_width * face_height) / (img_height * img_width)

        is_good_size = self.AREA_MIN < area_ratio < self.AREA_MAX
        return is_good_size, area_ratio

    def check_pose(self, face: Face):
        yaw, pitch, roll = face.pose
        is_good_pose = (
            abs(yaw) < self.POSE_YAW_PITCH
            and abs(pitch) < self.POSE_YAW_PITCH
            and abs(roll) < self.POSE_ROLL
        )
        return is_good_pose, (yaw, pitch, roll)

    @staticmethod
    def get_det_conf_score(face: Face):
        return face.det_score

    def is_face(self, face: Face):
        det_conf_score = self.get_det_conf_score(face)
        return det_conf_score >= self.DET_CONF_THR
