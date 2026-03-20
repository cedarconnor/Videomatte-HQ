"""MatAnyone 2 model wrapper for temporal video matting.

Follows the actual MatAnyone2 inference API from inference_matanyone2.py:
  1. Load model via get_matanyone2_model() (Hydra config + checkpoint)
  2. Create InferenceCore processor
  3. Frame 0: processor.step(image, mask, objects=[1]) to encode mask
  4. Frame 0: processor.step(image, first_frame_pred=True) for first prediction
  5. Warmup: repeat frame 0 with first_frame_pred=True
  6. Remaining frames: processor.step(image) for tracking
  7. output_prob_to_mask() to convert probabilities to alpha
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@contextmanager
def _prepend_sys_path(path: Path) -> Iterator[None]:
    p = str(path)
    already = p in sys.path
    if not already:
        sys.path.insert(0, p)
    try:
        yield
    finally:
        if not already:
            try:
                sys.path.remove(p)
            except ValueError:
                pass


class MatAnyone2Model:
    """Wrapper for MatAnyone 2 inference.

    Usage::

        model = MatAnyone2Model(repo_dir="third_party/MatAnyone2", device="cuda")
        model.load()
        alphas = model.process_video(source_frames, first_frame_mask)
        model.unload()
    """

    def __init__(
        self,
        repo_dir: str = "third_party/MatAnyone2",
        device: str = "cuda",
        precision: str = "fp16",
        max_size: int = 1080,
        warmup: int = 10,
        erode_kernel: int = 10,
        dilate_kernel: int = 10,
    ):
        self.repo_dir = str(repo_dir)
        self.device_str = device
        self.precision = precision
        self.max_size = int(max_size)
        self.warmup = int(warmup)
        self.erode_kernel = int(erode_kernel)
        self.dilate_kernel = int(dilate_kernel)
        self._processor = None
        self._network = None

    def load(self) -> None:
        """Load MatAnyone 2 model and inference core."""
        import torch

        repo_path = Path(self.repo_dir).expanduser().resolve()
        if not repo_path.exists():
            raise RuntimeError(f"MatAnyone2 repo not found: {repo_path}")

        device = torch.device(self.device_str if torch.cuda.is_available() else "cpu")

        # Find checkpoint
        ckpt_path = repo_path / "pretrained_models" / "matanyone2.pth"
        if not ckpt_path.exists():
            # Try downloading
            with _prepend_sys_path(repo_path):
                from hugging_face.tools.download_util import load_file_from_url
            url = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"
            ckpt_str = load_file_from_url(url, str(repo_path / "pretrained_models"))
            ckpt_path = Path(ckpt_str)

        with _prepend_sys_path(repo_path):
            from matanyone2.utils.get_default_model import get_matanyone2_model
            from matanyone2.inference.inference_core import InferenceCore

            network = get_matanyone2_model(str(ckpt_path), device)
            processor = InferenceCore(network, cfg=network.cfg)

        self._network = network
        self._processor = processor
        self._device = device

        logger.info(
            "MatAnyone2 loaded from %s on %s (max_size=%d, warmup=%d)",
            ckpt_path, device, self.max_size, self.warmup,
        )

    def unload(self) -> None:
        """Unload model and free VRAM."""
        from videomatte_hq.utils.vram import unload_model
        if self._network is not None:
            unload_model(self._network, "MatAnyone2")
        self._processor = None
        self._network = None

    def process_video(
        self,
        frames,
        first_frame_mask: np.ndarray,
        progress_callback=None,
    ) -> list[np.ndarray]:
        """Process a video sequence and return alpha mattes.

        Args:
            frames: Indexable sequence of BGR uint8 frames (list or FrameSource).
                    Frames are read lazily one at a time to avoid OOM on large videos.
            first_frame_mask: Binary mask (float32, 0-1) for frame 0 at native res.
            progress_callback: Optional callable(current_frame, total_frames).

        Returns:
            List of float32 alpha mattes at MatAnyone2 processing resolution.
        """
        import torch
        import torch.nn.functional as F

        if self._processor is None:
            raise RuntimeError("MatAnyone2 not loaded. Call load() first.")

        if not frames:
            return []

        device = self._device
        repo_path = Path(self.repo_dir).expanduser().resolve()

        with _prepend_sys_path(repo_path):
            from matanyone2.utils.inference_utils import gen_dilate, gen_erosion

        # Determine processing resolution from first frame
        h, w = frames[0].shape[:2]
        min_side = min(h, w)
        resize_needed = self.max_size > 0 and min_side > self.max_size
        if resize_needed:
            new_h = round(h / min_side * self.max_size)
            new_w = round(w / min_side * self.max_size)
            logger.info("Will resize %dx%d -> %dx%d for MatAnyone2 processing", w, h, new_w, new_h)
        else:
            new_h, new_w = h, w

        def _prepare_frame(frame: np.ndarray) -> torch.Tensor:
            """Convert a frame to a (3,H,W) float tensor in [0,255] at processing res.

            Handles both BGR uint8 (from cv2.VideoCapture) and float32 RGB
            [0,1] (from FrameSource).  The returned tensor is in [0,255]
            because MatAnyone2 normalizes with ``image / 255`` internally.
            """
            f = np.asarray(frame)
            # Detect float [0,1] input (FrameSource) vs uint8 [0,255] (raw cv2)
            is_float01 = f.dtype in (np.float32, np.float64) and float(f.max()) <= 1.01
            if is_float01:
                # FrameSource provides RGB float32 [0,1] — convert to uint8 RGB
                rgb = np.clip(f * 255.0, 0, 255).astype(np.uint8)
            elif f.ndim == 3 and f.shape[2] == 3:
                # BGR uint8 from cv2 — convert to RGB
                rgb = f[..., ::-1].copy()
            else:
                rgb = f.copy()
            if resize_needed:
                rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float()

        # Pre-compute frame 0 tensor (reused for warmup)
        frame0_tensor = _prepare_frame(frames[0])

        # Prepare mask: float32 0-255 range as expected by MatAnyone2
        mask_np = (np.clip(first_frame_mask, 0.0, 1.0) * 255.0).astype(np.uint8)

        # Apply erode/dilate to mask (MatAnyone2 convention)
        if self.dilate_kernel > 0:
            mask_np = gen_dilate(mask_np, self.dilate_kernel, self.dilate_kernel)
        if self.erode_kernel > 0:
            mask_np = gen_erosion(mask_np, self.erode_kernel, self.erode_kernel)

        mask_tensor = torch.from_numpy(mask_np.astype(np.float32)).to(device)

        # Resize mask if needed
        if resize_needed:
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0),
                size=(new_h, new_w),
                mode="nearest",
            )[0, 0]

        # Build the frame schedule: warmup copies of frame 0, then all source frames
        n_warmup = max(0, self.warmup)
        total_length = n_warmup + len(frames)

        # Run inference frame-by-frame under inference_mode (no gradient tracking)
        processor = self._processor
        alphas: list[np.ndarray] = []
        num_source_frames = len(frames)

        with torch.inference_mode():
            for ti in range(total_length):
                # Select the right frame tensor
                if ti < n_warmup:
                    image = frame0_tensor  # warmup: reuse frame 0
                else:
                    image = _prepare_frame(frames[ti - n_warmup])

                image_input = (image / 255.0).float().to(device)  # 3xHxW, [0,1]

                if ti == 0:
                    # Encode the given mask
                    processor.step(image_input, mask_tensor, objects=[1])
                    # First frame prediction
                    output_prob = processor.step(image_input, first_frame_pred=True)
                elif ti <= n_warmup:
                    # Warmup: reinit as first frame
                    output_prob = processor.step(image_input, first_frame_pred=True)
                else:
                    # Normal tracking
                    output_prob = processor.step(image_input)

                # Convert to alpha matte
                mask_out = processor.output_prob_to_mask(output_prob)

                # Only save non-warmup frames
                if ti >= n_warmup:
                    alpha = mask_out.detach().cpu().float().numpy()
                    # Ensure 2D (H, W) — output_prob_to_mask can return
                    # (H, W) or (1, H, W) depending on squeeze behavior.
                    while alpha.ndim > 2:
                        alpha = alpha[0]
                    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
                    alphas.append(alpha)

                    frame_idx = ti - n_warmup
                    if progress_callback is not None:
                        progress_callback(frame_idx + 1, num_source_frames)

                    if (frame_idx + 1) % 25 == 0 or frame_idx == 0:
                        logger.info("MatAnyone2 frame %d/%d", frame_idx + 1, num_source_frames)

        return alphas
