from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from videomatte_hq.io.reader import FrameSource, VideoReader


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray]):
        self._frames = frames
        self._pos = 0
        self.set_calls: list[tuple[int, float]] = []
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV naming
        return True

    def get(self, prop: int) -> float:  # noqa: N802 - OpenCV naming
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self._frames))
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._frames[0].shape[1])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._frames[0].shape[0])
        return 0.0

    def set(self, prop: int, value: float) -> bool:  # noqa: N802 - OpenCV naming
        self.set_calls.append((int(prop), float(value)))
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._pos = int(value)
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:  # noqa: N802 - OpenCV naming
        if self._pos < 0 or self._pos >= len(self._frames):
            return False, None
        frame = self._frames[self._pos]
        self._pos += 1
        return True, frame.copy()

    def release(self) -> None:  # noqa: N802 - OpenCV naming
        self.released = True


def test_video_reader_sequential_access_avoids_extra_seeks(monkeypatch) -> None:
    frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in (10, 20, 30)]
    fake = _FakeCapture(frames)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: fake)

    reader = VideoReader(Path("fake.mp4"), frame_start=0, frame_end=2)
    _ = reader.read_frame(0)
    _ = reader.read_frame(1)
    _ = reader.read_frame(2)
    reader.close()

    # One initial seek to frame_start; no additional sequential seeks.
    assert len(fake.set_calls) == 1
    assert fake.set_calls[0][0] == int(cv2.CAP_PROP_POS_FRAMES)
    assert fake.set_calls[0][1] == 0.0


def test_video_reader_random_access_triggers_seek(monkeypatch) -> None:
    frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in (10, 20, 30, 40)]
    fake = _FakeCapture(frames)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: fake)

    reader = VideoReader(Path("fake.mp4"), frame_start=0, frame_end=3)
    _ = reader.read_frame(0)
    _ = reader.read_frame(2)
    reader.close()

    # Initial seek to frame_start plus an explicit seek for random jump to frame 2.
    assert len(fake.set_calls) == 2
    assert fake.set_calls[1][0] == int(cv2.CAP_PROP_POS_FRAMES)
    assert fake.set_calls[1][1] == 2.0


def test_frame_source_video_metadata_accessors(monkeypatch, tmp_path: Path) -> None:
    fake_video_path = tmp_path / "fake.mp4"
    fake_video_path.write_bytes(b"")

    class _FakeVideoReader:
        def __init__(self, path: Path, frame_start: int, frame_end: int | None):
            self.path = path
            self.frame_start = int(frame_start)
            self.frame_end = int(frame_end if frame_end is not None else 9)
            self.width = 16
            self.height = 12
            self.fps = 30.0

        def __len__(self) -> int:
            return self.frame_end - self.frame_start + 1

        def read_frame(self, index: int) -> np.ndarray:
            return np.zeros((self.height, self.width, 3), dtype=np.float32)

        def close(self) -> None:
            pass

    monkeypatch.setattr("videomatte_hq.io.reader.discover_frames", lambda *args, **kwargs: [fake_video_path])
    monkeypatch.setattr("videomatte_hq.io.reader.VideoReader", _FakeVideoReader)

    source = FrameSource(
        pattern=str(fake_video_path),
        frame_start=2,
        frame_end=6,
        prefetch_workers=0,
    )
    assert source.is_video is True
    assert source.video_path == fake_video_path
    assert source.video_frame_start == 2
    assert source.video_frame_end == 6
