"""Frame and alpha writer with configurable format output.

Default output is 16-bit PNG. EXR output (DWAA/ZIP/raw) available as option.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def write_alpha_png16(path: Path, alpha: np.ndarray) -> None:
    """Write alpha as 16-bit single-channel PNG.

    Args:
        path: Output file path.
        alpha: (H, W) float32 in [0, 1].
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    alpha_u16 = np.round(np.clip(alpha, 0.0, 1.0) * 65535.0).astype(np.uint16)
    cv2.imwrite(str(path), alpha_u16)


def write_alpha_exr(
    path: Path,
    alpha: np.ndarray,
    compression: str = "dwaa",
    dwaa_quality: float = 45.0,
) -> None:
    """Write alpha as single-channel EXR with configurable compression.

    Args:
        path: Output file path (should end in .exr).
        alpha: (H, W) float32 in [0, 1].
        compression: One of 'dwaa', 'zip', 'none'.
        dwaa_quality: DWAA compression quality (lower = better quality, larger files).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import OpenEXR
        import Imath

        h, w = alpha.shape[:2]
        alpha_f32 = alpha.astype(np.float32)

        # Select compression
        if compression == "dwaa":
            comp = Imath.Compression(Imath.Compression.DWAA_COMPRESSION)
        elif compression == "zip":
            comp = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
        elif compression == "none":
            comp = Imath.Compression(Imath.Compression.NO_COMPRESSION)
        else:
            comp = Imath.Compression(Imath.Compression.DWAA_COMPRESSION)

        header = OpenEXR.Header(w, h)
        header["compression"] = comp
        header["channels"] = {"Y": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}

        if compression == "dwaa" and hasattr(header, "dwaCompressionLevel"):
            header["dwaCompressionLevel"] = dwaa_quality

        out = OpenEXR.OutputFile(str(path), header)
        out.writePixels({"Y": alpha_f32.tobytes()})
        out.close()

    except ImportError:
        # Fallback to imageio
        import imageio
        imageio.imwrite(str(path), alpha.astype(np.float32))


def write_rgb_frame(path: Path, rgb: np.ndarray, depth: int = 16) -> None:
    """Write an RGB frame as PNG.

    Args:
        path: Output file path.
        rgb: (H, W, 3) float32 in [0, 1].
        depth: Bit depth — 8 or 16.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # RGB to BGR for OpenCV
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.shape[2] == 3 else rgb

    if depth == 16:
        out = np.round(np.clip(bgr, 0.0, 1.0) * 65535.0).astype(np.uint16)
    else:
        out = np.clip(bgr * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), out)


class AlphaWriter:
    """Async alpha frame writer with configurable format.

    Writes using a thread pool so the GPU pipeline isn't blocked on IO.
    """

    def __init__(
        self,
        output_pattern: str,
        alpha_format: str = "png16",
        dwaa_quality: float = 45.0,
        workers: int = 4,
        base_dir: Optional[Path] = None,
    ):
        self.output_pattern = output_pattern
        self.alpha_format = alpha_format
        self.dwaa_quality = dwaa_quality
        self.base_dir = Path(base_dir) if base_dir else Path(".")
        self._executor = ThreadPoolExecutor(max_workers=workers)
        self._futures: list[Future] = []

    def _frame_path(self, frame_idx: int) -> Path:
        """Generate output path for a frame index."""
        try:
            name = self.output_pattern % frame_idx
        except TypeError:
            name = self.output_pattern.format(frame_idx)
        return self.base_dir / name

    def write(self, frame_idx: int, alpha: np.ndarray) -> None:
        """Queue an alpha frame for async writing.

        Args:
            frame_idx: Frame number (used for filename formatting).
            alpha: (H, W) float32 alpha in [0, 1].
        """
        path = self._frame_path(frame_idx)

        if self.alpha_format == "png16":
            future = self._executor.submit(write_alpha_png16, path, alpha.copy())
        elif self.alpha_format in ("exr_dwaa", "exr_dwaa_hq"):
            quality = 20.0 if self.alpha_format == "exr_dwaa_hq" else self.dwaa_quality
            future = self._executor.submit(
                write_alpha_exr, path, alpha.copy(), "dwaa", quality
            )
        elif self.alpha_format == "exr_lossless":
            future = self._executor.submit(
                write_alpha_exr, path, alpha.copy(), "zip"
            )
        elif self.alpha_format == "exr_raw":
            future = self._executor.submit(
                write_alpha_exr, path, alpha.copy(), "none"
            )
        else:
            logger.warning(
                "Unrecognized alpha_format=%r; falling back to png16.",
                self.alpha_format,
            )
            future = self._executor.submit(write_alpha_png16, path, alpha.copy())

        self._futures.append(future)

    def flush(self) -> None:
        """Wait for all queued writes to complete."""
        futures = list(self._futures)
        self._futures.clear()
        errors: list[BaseException] = []
        for f in futures:
            try:
                f.result()
            except BaseException as exc:  # noqa: BLE001 - collect all write failures before raising
                errors.append(exc)
        if not errors:
            return
        if len(errors) == 1:
            raise errors[0]
        detail = "; ".join(f"{type(e).__name__}: {e}" for e in errors[:3])
        raise RuntimeError(f"Multiple alpha write failures ({len(errors)}). {detail}")

    def close(self) -> None:
        """Flush and shut down the executor."""
        self.flush()
        self._executor.shutdown(wait=True)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
