"""RAFT optical flow wrapper for temporal stabilization (Pass C).

Uses torchvision's built-in RAFT implementation for forward-backward flow.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class RAFTFlowModel:
    """RAFT optical flow model from torchvision."""

    def __init__(self, device: str = "cuda", precision: str = "fp16"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.precision = precision
        self.model = None

    def load_weights(self, device: str = "cuda") -> None:
        """Load RAFT from torchvision."""
        import torchvision
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        weights = torchvision.models.optical_flow.Raft_Large_Weights.DEFAULT
        self.model = torchvision.models.optical_flow.raft_large(weights=weights)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"RAFT loaded on {self.device}")

    def _preprocess(self, img: Tensor) -> Tensor:
        """Preprocess image for RAFT (expects [0, 255] range)."""
        img = img * 255.0
        # Pad to multiple of 8
        _, _, h, w = img.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="replicate")
        return img

    @torch.no_grad()
    def compute_flow(
        self,
        frame1: Tensor,
        frame2: Tensor,
    ) -> Tensor:
        """Compute optical flow from frame1 to frame2.

        flow_{1→2}: maps pixel locations in frame1 to frame2.

        Args:
            frame1: (1, 3, H, W) RGB in [0, 1].
            frame2: (1, 3, H, W) RGB in [0, 1].

        Returns:
            (1, 2, H, W) flow field (dx, dy).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        _, _, h, w = frame1.shape
        img1 = self._preprocess(frame1.to(self.device))
        img2 = self._preprocess(frame2.to(self.device))

        # RAFT returns list of flow predictions at different iterations
        flow_predictions = self.model(img1, img2)
        flow = flow_predictions[-1]  # Use final iteration

        # Remove padding
        flow = flow[:, :, :h, :w]
        return flow

    @torch.no_grad()
    def compute_bidirectional_flow(
        self,
        frame1: Tensor,
        frame2: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute forward and backward flow between two frames.

        Args:
            frame1: (1, 3, H, W) frame at time t-1.
            frame2: (1, 3, H, W) frame at time t.

        Returns:
            (flow_forward, flow_backward):
                flow_forward: (1, 2, H, W) flow from t-1 to t.
                flow_backward: (1, 2, H, W) flow from t to t-1.
        """
        flow_forward = self.compute_flow(frame1, frame2)
        flow_backward = self.compute_flow(frame2, frame1)
        return flow_forward, flow_backward
