"""Problem frame detector — flags frames exceeding QC thresholds."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProblemFrame:
    frame_idx: int
    issues: list[str] = field(default_factory=list)
    severity: str = "warning"  # "warning" or "critical"


def detect_problems(
    alphas: list[np.ndarray],
    per_frame_data: list[dict],
    flicker_threshold: float = 0.005,
    coverage_spike_threshold: float = 2.0,
) -> list[ProblemFrame]:
    """Detect problem frames by monitoring QC metrics.

    Args:
        alphas: Final alpha per frame.
        per_frame_data: Per-frame data with band info.
        flicker_threshold: Max acceptable frame-to-frame mean diff.
        coverage_spike_threshold: Flag if band coverage is Nx the running average.

    Returns:
        List of ProblemFrame objects.
    """
    problems = []
    num_frames = len(alphas)

    if num_frames < 2:
        return problems

    # Compute running band coverage average
    coverages = [d.get("band", np.zeros(1)).mean() for d in per_frame_data]
    running_avg = np.cumsum(coverages) / np.arange(1, num_frames + 1)

    for t in range(1, num_frames):
        issues = []

        # Flicker check
        flicker = float(np.abs(alphas[t] - alphas[t-1]).mean())
        if flicker > flicker_threshold:
            issues.append(f"flicker={flicker:.4f} (>{flicker_threshold})")

        # Band coverage spike
        if t > 5 and running_avg[t-1] > 0:
            spike = coverages[t] / running_avg[t-1]
            if spike > coverage_spike_threshold:
                issues.append(f"band_spike={spike:.1f}x")

        if issues:
            problems.append(ProblemFrame(
                frame_idx=t,
                issues=issues,
                severity="critical" if flicker > flicker_threshold * 3 else "warning",
            ))

    if problems:
        logger.warning(f"Detected {len(problems)} problem frames")
    return problems
