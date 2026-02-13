"""ViTMatte wrapper for edge refinement (Pass B) and intermediate refinement (Pass A′).

ViTMatte is the recommended default — strong detail recovery especially for hair,
trimap-guided, transformer backbone via Hugging Face.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class ViTMatteModel:
    """ViTMatte wrapper implementing EdgeRefiner protocol.

    Uses Hugging Face transformers for model loading.
    """

    def __init__(
        self,
        model_name: str = "hustvl/vitmatte-small-distinctions-646",
        device: str = "cuda",
        precision: str = "fp16",
    ):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.precision = precision
        self.model = None
        self.processor = None

    def load_weights(self, device: str = "cuda") -> None:
        """Load ViTMatte from Hugging Face."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        try:
            from transformers import VitMatteForImageMatting, VitMatteImageProcessor

            self.processor = VitMatteImageProcessor.from_pretrained(self.model_name)
            self.model = VitMatteForImageMatting.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            if self.precision == "fp16" and self.device.type == "cuda":
                self.model = self.model.half()

            logger.info(f"ViTMatte loaded: {self.model_name} on {self.device}")

        except ImportError:
            logger.error(
                "transformers package required for ViTMatte. "
                "Install with: pip install transformers"
            )
            raise

    @torch.no_grad()
    def infer_tile(
        self,
        rgb_tile: Tensor,
        trimap_tile: Tensor,
        alpha_prior: Tensor,
        bg_tile: Optional[Tensor] = None,
    ) -> Tensor:
        """Run ViTMatte on a single tile.

        Args:
            rgb_tile: (3, H, W) RGB float32 in [0, 1].
            trimap_tile: (1, H, W) trimap {0, 0.5, 1}.
            alpha_prior: (1, H, W) guidance (unused by ViTMatte, but part of protocol).
            bg_tile: Optional background tile (unused by ViTMatte).

        Returns:
            (1, H, W) refined alpha.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded — call load_weights() first")

        _, h, w = rgb_tile.shape

        # ViTMatte expects pixel_values (B, 4, H, W) = RGB + trimap concatenated
        # The trimap channel: 0=BG, 128=unknown, 255=FG (in uint8 convention)
        trimap_input = trimap_tile  # already 0/0.5/1

        # Concatenate RGB + trimap → (1, 4, H, W)  
        input_tensor = torch.cat([rgb_tile.unsqueeze(0), trimap_input.unsqueeze(0)], dim=1)
        input_tensor = input_tensor.to(self.device)

        if self.precision == "fp16" and self.device.type == "cuda":
            input_tensor = input_tensor.half()

        # Run model
        outputs = self.model(pixel_values=input_tensor)
        alpha = outputs.alphas  # (1, 1, H, W)

        if self.precision == "fp16":
            alpha = alpha.float()

        return alpha[0].clamp(0, 1)  # (1, H, W)

    def infer_frame(
        self,
        rgb: Tensor,
        trimap: Tensor,
    ) -> Tensor:
        """Run ViTMatte on a full frame (used in Pass A′).

        Args:
            rgb: (3, H, W) RGB.
            trimap: (1, H, W) trimap.

        Returns:
            (1, H, W) alpha.
        """
        return self.infer_tile(rgb, trimap, alpha_prior=trimap)
