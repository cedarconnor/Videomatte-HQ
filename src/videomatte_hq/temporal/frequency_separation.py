"""Frequency separation — structural vs detail delta split."""

from __future__ import annotations

import cv2
import numpy as np


def split_delta(
    delta: np.ndarray,
    structural_sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split delta into structural (low-freq) and detail (high-freq) components.

    Args:
        delta: (H, W) float32 delta = A1 - A0prime.
        structural_sigma: Gaussian blur sigma for structural extraction.

    Returns:
        (D_structural, D_detail) pair.
    """
    ksize = int(structural_sigma * 6) | 1  # ensure odd
    D_structural = cv2.GaussianBlur(delta, (ksize, ksize), structural_sigma)
    D_detail = delta - D_structural
    return D_structural.astype(np.float32), D_detail.astype(np.float32)


def classify_structural_regions(
    D_structural: np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    """Identify regions where structural delta is large.

    These regions get conservative stabilization to prevent rubber edges.

    Args:
        D_structural: (H, W) float32 structural delta.
        threshold: Magnitude above which to flag as structural correction.

    Returns:
        (H, W) bool mask of structural-correction regions.
    """
    return np.abs(D_structural) > threshold
