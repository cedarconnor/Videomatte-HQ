"""RobustVideoMatting (RVM) wrapper for Pass A.

RVM is the recommended default global backbone — designed for temporal
consistency with a recurrent architecture. Edge quality is mediocre
but that's fine since we refine edges in Passes A′ and B.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Default model URLs — TorchScript exports from the RVM repo
RVM_URLS = {
    "mobilenetv3_fp32": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.torchscript",
    "mobilenetv3_fp16": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp16.torchscript",
    "resnet50_fp32": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.torchscript",
    "resnet50_fp16": "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp16.torchscript",
}


class RVMModel:
    """RobustVideoMatting wrapper implementing GlobalMatteModel protocol."""

    def __init__(
        self,
        variant: str = "mobilenetv3",
        device: str = "cuda",
        precision: str = "fp16",
        cache_dir: str = ".cache/videomatte-hq/models",
    ):
        self.variant = variant
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.precision = precision
        self.cache_dir = Path(cache_dir)
        self.model = None

    def load_weights(self, device: str = "cuda") -> None:
        """Download and load RVM weights."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Select fp16 TorchScript on CUDA, fp32 on CPU
        prec = "fp16" if (self.precision == "fp16" and self.device.type == "cuda") else "fp32"
        key = f"{self.variant}_{prec}"
        url = RVM_URLS[key]
        weight_name = f"rvm_{key}.torchscript"
        weight_path = self.cache_dir / weight_name

        if not weight_path.exists():
            logger.info(f"Downloading RVM weights from {url}")
            torch.hub.download_url_to_file(url, str(weight_path))

        logger.info(f"Loading RVM ({self.variant}) from {weight_path}")
        self.model = torch.jit.load(str(weight_path), map_location=self.device)
        self.model.eval()
        logger.info(f"RVM running in {prec} mode")

    @torch.no_grad()
    def infer_chunk(
        self,
        frames: Tensor,
        recurrent_state: Any = None,
    ) -> Tuple[Tensor, Any]:
        """Run RVM on a chunk of frames.

        Args:
            frames: (T, C, H, W) RGB in [0, 1].
            recurrent_state: (rec0, rec1, rec2, rec3) or None.

        Returns:
            (alpha, state): (T, 1, H, W) alpha, updated recurrent state.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded — call load_weights() first")

        T, C, H, W = frames.shape
        frames = frames.to(self.device)
        if self.precision == "fp16" and self.device.type == "cuda":
            frames = frames.half()

        # RVM expects: src (B, C, H, W), recurrent states
        # Process frame by frame to maintain recurrent state
        if recurrent_state is None:
            rec = [None] * 4
        else:
            rec = list(recurrent_state)

        alphas = []
        for t in range(T):
            src = frames[t:t+1]  # (1, C, H, W)

            # RVM forward: returns (fgr, pha, *rec)
            result = self.model(src, *rec, downsample_ratio=0.25)
            fgr, pha = result[0], result[1]
            rec = list(result[2:])

            alphas.append(pha)  # (1, 1, H, W)

        alpha = torch.cat(alphas, dim=0)  # (T, 1, H, W)

        if self.precision == "fp16":
            alpha = alpha.float()

        return alpha.clamp(0, 1), tuple(rec)
