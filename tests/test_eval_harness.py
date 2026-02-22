from __future__ import annotations

import numpy as np

from videomatte_hq.eval.harness import compare_alpha_sequences, summarize_alpha_sequence


def _sequence(shift: int = 0) -> list[np.ndarray]:
    out = []
    for t in range(5):
        alpha = np.zeros((24, 24), dtype=np.float32)
        x0 = 6 + t + shift
        x1 = min(x0 + 6, 24)
        alpha[8:16, x0:x1] = 1.0
        out.append(alpha)
    return out


def test_summarize_alpha_sequence_has_expected_keys() -> None:
    summary = summarize_alpha_sequence(_sequence())
    for key in (
        "temporal_iou_mean",
        "temporal_iou_std",
        "area_jitter_mean",
        "centroid_jitter_mean",
        "mean_coverage",
    ):
        assert key in summary
    assert summary["num_frames"] == 5.0


def test_compare_alpha_sequences_reports_deltas() -> None:
    ref = _sequence(shift=0)
    cand = _sequence(shift=1)
    result = compare_alpha_sequences(ref, cand)
    assert result["frames_compared"] == 5
    assert "diff" in result
    assert result["frame_mae_mean"] > 0.0
