"""Frame sequence and video reader with async prefetch support."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _parse_frame_pattern(pattern: str | Path) -> tuple[Path, str]:
    """Parse a C-style frame pattern like 'frames/%06d.png' into (directory, glob)."""
    p = Path(pattern)
    parent = p.parent
    name = p.name
    # Convert %06d style to glob *
    glob_pattern = re.sub(r"%\d*d", "*", name)
    return parent, glob_pattern


def _extract_frame_number(path: Path, pattern_name: str) -> int:
    """Extract frame number from filename using the pattern."""
    stem = path.stem
    # Try to extract leading digits from the stem
    match = re.search(r"(\d+)", stem)
    if match:
        return int(match.group(1))
    return 0


def discover_frames(
    pattern: str,
    base_dir: Optional[Path] = None,
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
) -> list[Path]:
    """Discover frame files matching a pattern.

    Args:
        pattern: C-style pattern like 'frames/%06d.png' or a video file path.
        base_dir: Base directory to resolve relative patterns against.
        frame_start: Optional first frame index (inclusive).
        frame_end: Optional last frame index (inclusive).

    Returns:
        Sorted list of frame file paths.
    """
    pattern_path = Path(pattern)
    if base_dir:
        pattern_path = base_dir / pattern_path

    # Check if the pattern points to a video file
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm"}
    if pattern_path.suffix.lower() in video_exts:
        if pattern_path.exists():
            return [pattern_path]
        raise FileNotFoundError(f"Video file not found: {pattern_path}")

    # Frame sequence
    parent, glob_pat = _parse_frame_pattern(pattern_path)

    if not parent.exists():
        raise FileNotFoundError(f"Frame directory not found: {parent}")

    frames = sorted(parent.glob(glob_pat), key=lambda p: _extract_frame_number(p, glob_pat))

    if not frames:
        raise FileNotFoundError(f"No frames found matching pattern: {parent / glob_pat}")

    # Filter by frame range
    if frame_start is not None or frame_end is not None:
        filtered = []
        for f in frames:
            num = _extract_frame_number(f, glob_pat)
            if frame_start is not None and num < frame_start:
                continue
            if frame_end is not None and num > frame_end:
                continue
            filtered.append(f)
        frames = filtered

    logger.info(f"Discovered {len(frames)} frames")
    return frames


def read_frame(path: Path, as_float: bool = True) -> np.ndarray:
    """Read a single frame from disk.

    Supports PNG (8/16-bit), EXR (float), and common image formats.

    Args:
        path: Path to the frame file.
        as_float: If True, normalize to [0, 1] float32.

    Returns:
        (H, W, C) numpy array in RGB order, float32 if as_float=True.
    """
    suffix = path.suffix.lower()

    if suffix == ".exr":
        return _read_exr(path)

    # Use OpenCV for PNG and other formats
    # IMREAD_UNCHANGED preserves 16-bit depth and alpha
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Failed to read image: {path}")

    # Convert BGR(A) to RGB(A)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if as_float:
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        elif img.dtype != np.float32:
            img = img.astype(np.float32)

    return img


def _read_exr(path: Path) -> np.ndarray:
    """Read an EXR file to float32 numpy array."""
    try:
        import OpenEXR
        import Imath

        exr_file = OpenEXR.InputFile(str(path))
        header = exr_file.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        # Determine available channels
        channels = list(header["channels"].keys())
        pt = Imath.PixelType(Imath.PixelType.FLOAT)

        if "R" in channels and "G" in channels and "B" in channels:
            r_str = exr_file.channel("R", pt)
            g_str = exr_file.channel("G", pt)
            b_str = exr_file.channel("B", pt)
            r = np.frombuffer(r_str, dtype=np.float32).reshape(height, width)
            g = np.frombuffer(g_str, dtype=np.float32).reshape(height, width)
            b = np.frombuffer(b_str, dtype=np.float32).reshape(height, width)

            if "A" in channels:
                a_str = exr_file.channel("A", pt)
                a = np.frombuffer(a_str, dtype=np.float32).reshape(height, width)
                return np.stack([r, g, b, a], axis=-1)
            return np.stack([r, g, b], axis=-1)

        elif "Y" in channels:
            # Single-channel (alpha or luminance)
            y_str = exr_file.channel("Y", pt)
            y = np.frombuffer(y_str, dtype=np.float32).reshape(height, width)
            return y[..., np.newaxis]

        else:
            # Try first available channel
            ch_name = channels[0]
            ch_str = exr_file.channel(ch_name, pt)
            ch = np.frombuffer(ch_str, dtype=np.float32).reshape(height, width)
            return ch[..., np.newaxis]

    except ImportError:
        # Fallback to imageio
        import imageio
        return imageio.imread(str(path)).astype(np.float32)


class VideoReader:
    """Read frames from a video file using OpenCV."""

    def __init__(self, path: Path, frame_start: int = 0, frame_end: Optional[int] = None):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise IOError(f"Failed to open video: {path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.frame_start = frame_start
        self.frame_end = frame_end if (frame_end is not None and frame_end >= 0) else self.total_frames - 1
        self._next_abs_idx = int(self.frame_start)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(self.frame_start))

        logger.info(
            f"Video: {path.name} — {self.width}×{self.height} @ {self.fps:.1f}fps, "
            f"{self.total_frames} frames, range [{self.frame_start}:{self.frame_end}]"
        )

    def __len__(self) -> int:
        return max(0, int(self.frame_end) - int(self.frame_start) + 1)

    def read_frame(self, index: int) -> np.ndarray:
        """Read a specific frame by index (0-based relative to frame_start)."""
        abs_idx = self.frame_start + index
        # Fast path: avoid expensive decoder seek for sequential access.
        if int(abs_idx) != int(self._next_abs_idx):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(abs_idx))
            self._next_abs_idx = int(abs_idx)
        ret, frame = self.cap.read()
        if not ret:
            raise IOError(f"Failed to read frame {abs_idx} from {self.path}")
        self._next_abs_idx = int(abs_idx) + 1
        # BGR to RGB, normalize to float32
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame.astype(np.float32) / 255.0

    def __iter__(self) -> Iterator[np.ndarray]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(self.frame_start))
        self._next_abs_idx = int(self.frame_start)
        for i in range(len(self)):
            ret, frame = self.cap.read()
            if not ret:
                break
            self._next_abs_idx = int(self.frame_start) + i + 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame.astype(np.float32) / 255.0

    def close(self):
        self.cap.release()

    def __del__(self):
        self.close()


class FrameSource:
    """Unified frame source that handles both image sequences and video files.

    Provides indexed access to frames with optional async prefetching.
    """

    def __init__(
        self,
        pattern: str,
        base_dir: Optional[Path] = None,
        frame_start: Optional[int] = None,
        frame_end: Optional[int] = None,
        prefetch_workers: int = 4,
    ):
        self.pattern = pattern
        self.base_dir = Path(base_dir) if base_dir else None
        self._prefetch_workers = prefetch_workers
        # Prefetch is not implemented yet; avoid creating an unused executor.
        self._executor = None
        self._resolution_cache: tuple[int, int] | None = None

        # Detect source type
        files = discover_frames(pattern, base_dir, frame_start, frame_end)
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm"}

        if len(files) == 1 and files[0].suffix.lower() in video_exts:
            effective_end = frame_end if (frame_end is not None and frame_end >= 0) else None
            self._video = VideoReader(files[0], frame_start or 0, effective_end)
            self._frames = None
            self._is_video = True
            self._resolution_cache = (self._video.height, self._video.width)
        else:
            self._video = None
            self._frames = files
            self._is_video = False

    @property
    def num_frames(self) -> int:
        if self._is_video:
            return len(self._video)
        return len(self._frames)

    @property
    def is_video(self) -> bool:
        return bool(self._is_video)

    @property
    def video_path(self) -> Path | None:
        if not self._is_video or self._video is None:
            return None
        return self._video.path

    @property
    def video_frame_start(self) -> int:
        if not self._is_video or self._video is None:
            return 0
        return int(self._video.frame_start)

    @property
    def video_frame_end(self) -> int:
        if not self._is_video or self._video is None:
            return max(0, len(self) - 1)
        return int(self._video.frame_end)

    @property
    def fps(self) -> float:
        if self._is_video:
            return self._video.fps
        return 30.0  # default for image sequences

    @property
    def resolution(self) -> tuple[int, int]:
        """(height, width) of frames."""
        if self._is_video:
            return (self._video.height, self._video.width)
        if self._resolution_cache is None:
            # Cache sequence resolution to avoid decoding the first frame repeatedly.
            first = read_frame(self._frames[0])
            self._resolution_cache = tuple(int(v) for v in first.shape[:2])
        return self._resolution_cache

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, index: int) -> np.ndarray:
        """Read frame at index, returns (H, W, C) float32 RGB in [0, 1]."""
        if index < 0 or index >= len(self):
            raise IndexError(f"Frame index {index} out of range [0, {len(self)})")

        if self._is_video:
            return self._video.read_frame(index)
        return read_frame(self._frames[index])

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(len(self)):
            yield self[i]

    def close(self):
        if self._video:
            self._video.close()
        if self._executor:
            self._executor.shutdown(wait=False)
