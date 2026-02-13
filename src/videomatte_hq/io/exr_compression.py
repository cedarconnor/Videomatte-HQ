"""EXR compression utilities and DWAA round-trip artifact QC.

When using DWAA compression, the round-trip check detects frames where compression
artifacts in the edge band exceed a configurable threshold.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def dwaa_roundtrip_check(
    alpha: np.ndarray,
    edge_band: np.ndarray,
    dwaa_quality: float = 45.0,
    threshold: float = 0.01,
) -> tuple[float, bool]:
    """Check DWAA compression artifacts via round-trip encode/decode.

    Writes alpha to a temp EXR with DWAA, reads it back, and computes
    the maximum absolute error within the edge band region.

    Args:
        alpha: (H, W) float32 alpha in [0, 1].
        edge_band: (H, W) bool mask of edge-band pixels.
        dwaa_quality: DWAA compression quality level.
        threshold: Maximum acceptable error in edge band.

    Returns:
        (max_error, passed): Maximum error in band, and whether it's under threshold.
    """
    import tempfile

    from videomatte_hq.io.writer import write_alpha_exr
    from videomatte_hq.io.reader import _read_exr

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "roundtrip_test.exr"

        # Write with DWAA
        write_alpha_exr(tmp_path, alpha, compression="dwaa", dwaa_quality=dwaa_quality)

        # Read back
        alpha_rt = _read_exr(tmp_path)
        if alpha_rt.ndim == 3:
            alpha_rt = alpha_rt[..., 0]

    # Compute error within the band only
    diff = np.abs(alpha - alpha_rt)

    if edge_band.any():
        max_error_band = float(diff[edge_band].max())
        mean_error_band = float(diff[edge_band].mean())
    else:
        max_error_band = 0.0
        mean_error_band = 0.0

    passed = max_error_band <= threshold

    if not passed:
        logger.warning(
            f"DWAA artifact check FAILED: max_error={max_error_band:.6f} "
            f"(threshold={threshold}), mean={mean_error_band:.6f}"
        )

    return max_error_band, passed


def compression_qc_report(
    frame_errors: dict[int, float],
    threshold: float = 0.01,
) -> dict:
    """Summarize compression artifact QC across all sampled frames.

    Args:
        frame_errors: {frame_idx: max_edge_band_error}.
        threshold: Failure threshold.

    Returns:
        Summary dict with stats and flagged frames.
    """
    if not frame_errors:
        return {"sampled": 0, "failed": 0, "flagged_frames": []}

    errors = list(frame_errors.values())
    flagged = [idx for idx, err in frame_errors.items() if err > threshold]

    return {
        "sampled": len(errors),
        "failed": len(flagged),
        "max_error": max(errors),
        "mean_error": sum(errors) / len(errors),
        "threshold": threshold,
        "flagged_frames": flagged,
    }
