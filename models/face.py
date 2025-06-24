from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class Face:
    bbox: Tuple[int, int, int, int]         # (x1, y1, x2, y2)
    pose: Tuple[float, float, float]        # (yaw, pitch, roll)
    det_score: float                        # detection confidence
    embedding: np.ndarray                   # shape: (512,)
    age: int | None = None  # 0-100 (None = not predicted)

    @classmethod
    def from_insightface(cls, raw_face) -> "Face":
        return cls(
            bbox=tuple(map(int, raw_face.bbox)),
            pose=tuple(raw_face.pose),
            det_score=float(raw_face.det_score),
            embedding=raw_face.embedding.copy(),
            age=int(raw_face.age) if hasattr(raw_face, "age") else None
        )