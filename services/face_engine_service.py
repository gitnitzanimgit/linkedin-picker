from repositories.face_engine_repository import FaceEngineRepository
from models.image import Image


class FaceEngineService:
    """
    Business logic on top of raw face engine repository.
    """

    def __init__(self):
        self.face_engine_repository = FaceEngineRepository()

    def detect_faces(self, img: Image):
        return self.face_engine_repository.infer_faces(img.pixels)
