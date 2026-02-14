"""ViTMatte wrapper for edge refinement (Pass B) and intermediate refinement (Pass A′).

ViTMatte is the recommended default — strong detail recovery especially for hair,
trimap-guided, transformer backbone via Hugging Face.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
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
        if trimap_tile.shape[-2:] != (h, w):
            raise RuntimeError(
                f"ViTMatte input mismatch: rgb={tuple(rgb_tile.shape)} trimap={tuple(trimap_tile.shape)}"
            )

        # ViTMatte expects pixel_values (B, 4, H, W) = RGB + trimap concatenated
        # The trimap channel: 0=BG, 128=unknown, 255=FG (in uint8 convention)
        trimap_input = trimap_tile  # already 0/0.5/1

        # Concatenate RGB + trimap → (1, 4, H, W)  
        input_tensor = torch.cat([rgb_tile.unsqueeze(0), trimap_input.unsqueeze(0)], dim=1)
        input_tensor = input_tensor.to(self.device)

        # ViTMatte's decoder can produce 1px mismatches on odd/unaligned sizes.
        # Pad to model stride and crop output back to original tile dimensions.
        model_stride = 32
        pad_h = (model_stride - (h % model_stride)) % model_stride
        pad_w = (model_stride - (w % model_stride)) % model_stride
        if pad_h or pad_w:
            input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode="replicate")

        if self.precision == "fp16" and self.device.type == "cuda":
            input_tensor = input_tensor.half()

        # Run model
        outputs = self.model(pixel_values=input_tensor)
        alpha = outputs.alphas  # (1, 1, H, W)
        if pad_h or pad_w:
            alpha = alpha[:, :, :h, :w]

        if self.precision == "fp16":
            alpha = alpha.float()

        return alpha[0].clamp(0, 1)  # (1, H, W)

    @torch.no_grad()
    def infer_tile_batch(
        self,
        rgb_tiles: list[Tensor],
        trimap_tiles: list[Tensor],
        alpha_priors: list[Tensor],
        bg_tiles: list[Optional[Tensor]] | None = None,
    ) -> list[Tensor]:
        """Run ViTMatte on a batch of same-size tiles in one forward pass.

        All tiles must have the same (H, W) dimensions. Falls back to
        sequential inference if sizes differ or OOM occurs.

        Args:
            rgb_tiles: List of (3, H, W) RGB tiles.
            trimap_tiles: List of (1, H, W) trimap tiles.
            alpha_priors: List of (1, H, W) guidance tiles.
            bg_tiles: Optional list of background tiles (unused by ViTMatte).

        Returns:
            List of (1, H, W) refined alpha tiles.
        """
        if not rgb_tiles:
            return []

        if self.model is None:
            raise RuntimeError("Model not loaded — call load_weights() first")

        # Check all tiles are same size
        h, w = rgb_tiles[0].shape[1], rgb_tiles[0].shape[2]
        same_size = all(t.shape[1] == h and t.shape[2] == w for t in rgb_tiles)

        if not same_size:
            # Fallback: sequential
            return [
                self.infer_tile(rgb, tri, ap, bg)
                for rgb, tri, ap, bg in zip(
                    rgb_tiles, trimap_tiles, alpha_priors,
                    bg_tiles or [None] * len(rgb_tiles),
                )
            ]

        batch_size = len(rgb_tiles)

        # Build batched input: (B, 4, H, W)
        batch_inputs = []
        for rgb, tri in zip(rgb_tiles, trimap_tiles):
            batch_inputs.append(torch.cat([rgb.unsqueeze(0), tri.unsqueeze(0)], dim=1))
        input_tensor = torch.cat(batch_inputs, dim=0).to(self.device)  # (B, 4, H, W)

        # Pad to model stride
        model_stride = 32
        pad_h = (model_stride - (h % model_stride)) % model_stride
        pad_w = (model_stride - (w % model_stride)) % model_stride
        if pad_h or pad_w:
            input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode="replicate")

        if self.precision == "fp16" and self.device.type == "cuda":
            input_tensor = input_tensor.half()

        try:
            outputs = self.model(pixel_values=input_tensor)
            alphas = outputs.alphas  # (B, 1, H_pad, W_pad)
            if pad_h or pad_w:
                alphas = alphas[:, :, :h, :w]
            if self.precision == "fp16":
                alphas = alphas.float()
            return [alphas[i].clamp(0, 1) for i in range(batch_size)]
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM on batch of {batch_size} tiles, falling back to sequential")
            torch.cuda.empty_cache()
            return [
                self.infer_tile(rgb, tri, ap, bg)
                for rgb, tri, ap, bg in zip(
                    rgb_tiles, trimap_tiles, alpha_priors,
                    bg_tiles or [None] * len(rgb_tiles),
                )
            ]

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
