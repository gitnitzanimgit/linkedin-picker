from __future__ import annotations
from insightface.app import FaceAnalysis
import torch
import os
import numpy as np


class FaceEngine:
    """
    Singleton wrapper around InsightFace's FaceAnalysis (RetinaFace + ArcFace).

    Loads the model once and provides helper methods for detection and embedding.
    """

    _instance: FaceEngine | None = None  # Class-level cache for singleton

    def __new__(cls, *args, **kwargs):
        """
        Ensure model is loaded only once (Singleton pattern).

        Args:
            model_name (str, optional): Name of the InsightFace model to load.
            ctx_id (int, optional): Device index. 0 = CPU, 1+ = GPU index.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_engine(*args, **kwargs)
        return cls._instance

    def _init_engine(self, model_name: str = "buffalo_l", ctx_id: int = 0):
        """
        Load and prepare InsightFace model on first instantiation.

        Args:
            model_name (str): Model name from the InsightFace model zoo.
            ctx_id (int): 0 = CPU, 1+ = GPU index if using CUDA.
                         Note: MPS (Apple Silicon) not supported by InsightFace.
        """
        # ─── DEVICE SELECTION ────────────────────────────────────────────────
        # Priority: CUDA -> CPU -> MPS
        if torch.cuda.is_available():
            ctx_id = 0  # CUDA device 0
            device_name = "CUDA"
        else:
            ctx_id = 0   # CPU
            device_name = "CPU"
        # Note: MPS disabled - using CPU instead for better determinism
        
        # ─── FORCE SINGLE THREAD FOR DETERMINISM ─────────────────────────────
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        
        # ─── ONNX RUNTIME DETERMINISTIC SETTINGS ─────────────────────────────
        os.environ["ORT_DISABLE_ALL_OPTIMIZATION"] = "1"
        os.environ["ORT_FORCE_FALLBACK_TO_CPU"] = "1"
        os.environ["ORT_DETERMINISTIC_COMPUTE"] = "1"
        # ─────────────────────────────────────────────────────────────────────

        # ─── THREAD INFO ────────────────────────────────────────────────────
        try:
            # Get actual thread counts from numpy
            omp_threads = np.get_num_threads()
            print(f"🔍 InsightFace using: {device_name} (ctx_id={ctx_id}) | NumPy threads: {omp_threads}")
        except:
            print(f"🔍 InsightFace using: {device_name} (ctx_id={ctx_id}) | Threads: unknown")
        # ─────────────────────────────────────────────────────────────────────

        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id)
