"""Core v2 protocol interfaces and data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FrameSourceLike(Protocol):
    """Minimal frame-source interface required by v2 stages."""

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> np.ndarray:
        ...

    @property
    def resolution(self) -> tuple[int, int]:
        ...


@dataclass(slots=True)
class SegmentPrompt:
    """Prompt payload accepted by segmentation backends."""

    bbox: tuple[float, float, float, float] | None = None
    positive_points: list[tuple[float, float]] = field(default_factory=list)
    negative_points: list[tuple[float, float]] = field(default_factory=list)
    mask: np.ndarray | None = None


@dataclass(slots=True)
class SegmentResult:
    """Stage-1 segmentation outputs."""

    masks: list[np.ndarray]
    logits: list[np.ndarray]
    anchored_frames: list[int] = field(default_factory=list)


@runtime_checkable
class PromptAdapter(Protocol):
    """Convert user-provided masks to backend prompt payloads."""

    def adapt(self, mask: np.ndarray, frame_shape: tuple[int, int]) -> SegmentPrompt:
        ...


@runtime_checkable
class Segmenter(Protocol):
    """Temporal sequence segmenter."""

    def segment_sequence(
        self,
        source: FrameSourceLike,
        prompt: SegmentPrompt,
        anchor_frame: int = 0,
        chunk_size: int = 100,
        chunk_overlap: int = 5,
    ) -> SegmentResult:
        ...


@runtime_checkable
class EdgeRefiner(Protocol):
    """High-resolution edge-aware alpha refinement."""

    def refine(self, rgb: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        ...
