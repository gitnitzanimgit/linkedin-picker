# models/clip_model.py
from __future__ import annotations
import threading
import torch
import open_clip


class ClipModel:
    """
    Singleton wrapper around OpenCLIP.
    • We freeze the weights once (requires_grad = False)
    • We keep autograd ON for encode_image(), so gradients
      can flow back to *inputs* (needed for optimisation).
    """

    _instance = None
    _lock = threading.RLock()

    # ───────────────────────── singleton ctor
    def __new__(cls,
                arch: str = "ViT-B-32",
                ckpt: str = "laion2b_s34b_b79k"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init(arch, ckpt)
            return cls._instance

    # ───────────────────────── actual init
    def _init(self, arch: str, ckpt: str):
        # Use MPS on Apple Silicon, CUDA on NVIDIA, CPU otherwise
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        model, _, _ = open_clip.create_model_and_transforms(
            arch, pretrained=ckpt
        )
        self.model = model.to(self.device).eval()

        # Freeze weights so we don't waste memory on their grad-buffers
        for p in self.model.parameters():
            p.requires_grad_(False)

        # CLIP's mean / std
        self.mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=self.device
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=self.device
        ).view(1, 3, 1, 1)

    # ───────────────────────── public API
    @torch.inference_mode()             # no grad needed for text
    def encode_text(self, prompt: str) -> torch.Tensor:
        tok = open_clip.tokenize([prompt]).to(self.device)
        feat = self.model.encode_text(tok).float()
        return feat / feat.norm(dim=-1, keepdim=True)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B,3,H,W) in [0,1] on self.device
        We *keep* autograd ON here—no decorator—
        because we want grads wrt the input.
        """
        n = (x - self.mean) / self.std
        feat = self.model.encode_image(n)
        return feat / feat.norm(dim=-1, keepdim=True)
