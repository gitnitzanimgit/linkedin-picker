from models.face_engine import FaceEngine
import numpy as np
from insightface.app.common import Face
from typing import List


class FaceEngineRepository:
    """
    Thin wrapper around FaceEngine that provides low-level access to detection and embedding.
    """

    def __init__(self):
        self.engine = FaceEngine()  # Singleton is handled inside

    def infer_faces(self, pixels_rgb: np.ndarray) -> List[Face]:
        img_bgr = pixels_rgb[:, :, ::-1]
        return self.engine.app.get(img_bgr)



