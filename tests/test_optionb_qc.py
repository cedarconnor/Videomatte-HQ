from __future__ import annotations

from pathlib import Path

import numpy as np

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.io.writer import AlphaWriter
from videomatte_hq.qc.optionb import (
    add_output_roundtrip_gate,
    evaluate_optionb_qc,
    failed_gate_names,
    write_optionb_qc_artifacts,
)


def _moving_square_sequence(num_frames: int, h: int = 32, w: int = 32) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for t in range(num_frames):
        alpha = np.zeros((h, w), dtype=np.float32)
        x0 = 4 + t
        alpha[10:22, x0 : x0 + 10] = 1.0
        out.append(alpha)
    return out


def test_optionb_qc_metrics_pass_on_stable_sequence() -> None:
    alphas = _moving_square_sequence(6)
    confidences = [np.full_like(a, 0.9, dtype=np.float32) for a in alphas]
    cfg = VideoMatteConfig()

    metrics = evaluate_optionb_qc(alphas=alphas, confidences=confidences, cfg=cfg)

    assert metrics["summary"]["num_frames"] == 6
    assert metrics["summary"]["invalid_alpha_frames"] == 0
    assert metrics["summary"]["range_violation_frames"] == 0
    assert failed_gate_names(metrics) == []


def test_optionb_qc_flags_invalid_and_flicker() -> None:
    alphas = [
        np.zeros((16, 16), dtype=np.float32),
        np.ones((16, 16), dtype=np.float32),
        np.full((16, 16), np.nan, dtype=np.float32),
        np.zeros((16, 16), dtype=np.float32),
    ]
    confidences = [np.ones((16, 16), dtype=np.float32) for _ in alphas]
    cfg = VideoMatteConfig(
        qc={
            "max_p95_flicker": 0.01,
            "max_p95_edge_flicker": 0.01,
            "min_mean_edge_confidence": 0.5,
        }
    )

    metrics = evaluate_optionb_qc(alphas=alphas, confidences=confidences, cfg=cfg)
    failed = failed_gate_names(metrics)

    assert "alpha_finite_range" in failed
    assert "temporal_flicker" in failed
    assert metrics["problem_frames"]


def test_optionb_qc_output_roundtrip_and_artifacts(tmp_path: Path) -> None:
    alphas = _moving_square_sequence(3, h=20, w=24)
    confidences = [np.full_like(a, 0.8, dtype=np.float32) for a in alphas]
    cfg = VideoMatteConfig(
        qc={
            "sample_output_frames": 3,
            "max_output_roundtrip_mae": 1e-3,
            "output_subdir": "qc",
            "metrics_filename": "metrics.json",
            "report_filename": "report.md",
        }
    )

    writer = AlphaWriter(
        output_pattern="alpha/%05d.png",
        alpha_format="png16",
        workers=1,
        base_dir=tmp_path,
    )
    for idx, alpha in enumerate(alphas):
        writer.write(idx, alpha)
    writer.close()

    metrics = evaluate_optionb_qc(alphas=alphas, confidences=confidences, cfg=cfg)
    metrics = add_output_roundtrip_gate(
        metrics,
        output_dir=tmp_path,
        output_pattern="alpha/%05d.png",
        alphas=alphas,
        frame_start=0,
        sample_count=cfg.qc.sample_output_frames,
        max_mae=cfg.qc.max_output_roundtrip_mae,
    )
    assert metrics["output_roundtrip"]["passed"] is True
    assert "output_roundtrip" not in failed_gate_names(metrics)

    metrics_path, report_path = write_optionb_qc_artifacts(
        metrics,
        output_dir=tmp_path,
        output_subdir=cfg.qc.output_subdir,
        metrics_filename=cfg.qc.metrics_filename,
        report_filename=cfg.qc.report_filename,
    )
    assert metrics_path.exists()
    assert report_path.exists()
