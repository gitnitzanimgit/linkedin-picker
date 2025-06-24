from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class BrightnessContrastGamma:
    """
    Value-object holding brightness/contrast/gamma offsets in
    human-friendly units (±0.3 ~ ±30 % or ±0.3 EV).
    """
    brightness: float = 0.0      # [-0.3 , +0.3]
    contrast:   float = 0.0      # [-0.3 , +0.3]
    gamma:      float = 0.0      # [-0.3 , +0.3]

    # ── Helpers for optimisation ─────────────────────────────────────
    def as_parameters(self, device: str = "cpu"):
        """
        Convert current scalars into three learnable nn.Parameters,
        mapping [-0.3,+0.3] → ℝ via atanh.
        """
        to_raw = lambda x: torch.atanh(torch.tensor(x / 0.3, device=device))
        return (
            torch.nn.Parameter(to_raw(self.brightness)),
            torch.nn.Parameter(to_raw(self.contrast)),
            torch.nn.Parameter(to_raw(self.gamma)),
        )

    # ── Core math (keeps gradients) ──────────────────────────────────
    @staticmethod
    def _apply_raw(x, p_b, p_c, p_g):      # x in [0,1]
        img_bright = x + torch.tanh(p_b) * 0.3

        contrast_fac = torch.tanh(p_c) * 0.3 + 1.0
        img_contrast = (img_bright - 0.5) * contrast_fac + 0.5

        gamma_pow = torch.tanh(p_g) * 0.3 + 1.0
        return img_contrast.clamp(0, 1).pow(gamma_pow)

    @classmethod
    def from_raw(cls, p_b, p_c, p_g):
        """Create a BCG object from raw (learnable) tensors."""
        return cls(
            brightness=(torch.tanh(p_b) * 0.3).item(),
            contrast=(torch.tanh(p_c) * 0.3).item(),
            gamma=(torch.tanh(p_g) * 0.3).item(),
        )

    # ── Apply after optimisation (no grads) ──────────────────────────
    def apply_to_tensor(self, x):
        to_raw = lambda v: torch.atanh(torch.tensor(v / 0.3, device=x.device))
        return self._apply_raw(x, to_raw(self.brightness),
                                  to_raw(self.contrast),
                                  to_raw(self.gamma))
