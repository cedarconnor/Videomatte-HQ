from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.pipeline.orchestrator import run_pipeline


def _write_test_frames(frames_dir: Path, count: int = 2) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        img = np.zeros((24, 24, 3), dtype=np.uint8)
        img[..., 0] = 50 + idx * 20
        img[..., 1] = 80
        img[..., 2] = 120
        cv2.imwrite(str(frames_dir / f"frame_{idx:05d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _patch_lightweight_pipeline(monkeypatch: pytest.MonkeyPatch, *, tmp_path: Path) -> dict[str, list[str]]:
    import videomatte_hq.pipeline.orchestrator as orch

    calls: dict[str, list[str]] = {"export": [], "diagnosis": []}

    key_mask = np.zeros((24, 24), dtype=np.float32)
    key_mask[6:18, 8:16] = 1.0

    def _fake_ensure_project(_cfg):
        return (tmp_path / "out" / "project.vmhqproj", object())

    monkeypatch.setattr(orch, "ensure_project", _fake_ensure_project)
    monkeypatch.setattr(orch, "load_keyframe_masks", lambda *_args, **_kwargs: {0: key_mask})
    monkeypatch.setattr(orch, "build_memory_region_priors", lambda **_kwargs: None)

    def _make_alphas(source, value: float) -> list[np.ndarray]:
        alphas: list[np.ndarray] = []
        for i in range(len(source)):
            a = np.full(source[i].shape[:2], value + 0.01 * i, dtype=np.float32)
            alphas.append(np.clip(a, 0.0, 1.0))
        return alphas

    def _make_confs(source, value: float) -> list[np.ndarray]:
        return [np.full(source[i].shape[:2], value, dtype=np.float32) for i in range(len(source))]

    monkeypatch.setattr(
        orch,
        "run_pass_memory",
        lambda source, **_kwargs: (_make_alphas(source, 0.35), _make_confs(source, 0.6)),
    )
    monkeypatch.setattr(
        orch,
        "run_pass_refine",
        lambda source, *_args, **_kwargs: _make_alphas(source, 0.45),
    )
    monkeypatch.setattr(
        orch,
        "run_pass_temporal_cleanup",
        lambda source, **_kwargs: _make_alphas(source, 0.5),
    )
    monkeypatch.setattr(
        orch,
        "run_pass_matte_tuning",
        lambda alphas, **_kwargs: [np.clip(np.asarray(a, dtype=np.float32), 0.0, 1.0) for a in alphas],
    )

    monkeypatch.setattr(
        orch,
        "evaluate_optionb_qc",
        lambda **_kwargs: {
            "summary": {
                "p95_flicker": 0.01,
                "p95_edge_flicker": 0.2,
                "mean_edge_confidence": 0.2,
            },
            "gates": [],
            "frames": [],
        },
    )
    monkeypatch.setattr(orch, "add_output_roundtrip_gate", lambda metrics, **_kwargs: metrics)

    def _fake_qc_artifacts(_metrics, output_dir, output_subdir, metrics_filename, report_filename):
        qc_dir = Path(output_dir) / output_subdir
        qc_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = qc_dir / metrics_filename
        report_path = qc_dir / report_filename
        metrics_path.write_text("{}", encoding="utf-8")
        report_path.write_text("qc", encoding="utf-8")
        return metrics_path, report_path

    monkeypatch.setattr(orch, "write_optionb_qc_artifacts", _fake_qc_artifacts)
    monkeypatch.setattr(orch, "failed_gate_names", lambda _metrics: ["edge_flicker"])

    def _fake_export_stage_samples(**kwargs):
        stage_name = str(kwargs["stage_name"])
        calls["export"].append(stage_name)
        frame_start = int(kwargs["frame_start"])
        sample_local_frames = [int(i) for i in kwargs["sample_local_frames"]]
        return {
            frame_start + idx: {
                "coverage": 0.1,
                "edge_band_coverage": 0.05,
                "largest_component_ratio": 1.0,
                "speckle_ratio": 0.0,
                "hole_ratio": 0.0,
            }
            for idx in sample_local_frames
        }

    def _fake_write_stage_diagnosis_report(**kwargs):
        calls["diagnosis"].append(",".join(str(s) for s in kwargs.get("stage_order", [])))
        stage_root = Path(kwargs["output_dir"]) / str(kwargs["stage_dir_name"])
        stage_root.mkdir(parents=True, exist_ok=True)
        json_path = stage_root / "diagnosis.json"
        md_path = stage_root / "diagnosis.md"
        json_path.write_text("{}", encoding="utf-8")
        md_path.write_text("diagnosis", encoding="utf-8")
        return json_path, md_path

    monkeypatch.setattr(orch, "export_stage_samples", _fake_export_stage_samples)
    monkeypatch.setattr(orch, "write_stage_diagnosis_report", _fake_write_stage_diagnosis_report)
    monkeypatch.setattr(orch, "save_project", lambda *_args, **_kwargs: None)
    return calls


def test_auto_stage_diagnosis_triggers_on_qc_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_test_frames(frames_dir, count=2)
    calls = _patch_lightweight_pipeline(monkeypatch, tmp_path=tmp_path)

    cfg = VideoMatteConfig(
        io={
            "input": str(frames_dir / "frame_%05d.png"),
            "output_dir": str(tmp_path / "out"),
            "output_alpha": "alpha/%05d.png",
            "frame_start": 0,
            "frame_end": 999999,
        },
        project={"path": str(tmp_path / "out" / "project.vmhqproj"), "autosave": False},
        runtime={"workers_io": 1, "cache_dir": str(tmp_path / ".cache"), "resume": False},
        qc={"enabled": True, "fail_on_regression": True, "auto_stage_diagnosis_on_fail": True},
        debug={
            "export_stage_samples": False,
            "auto_stage_samples_on_qc_fail": True,
            "sample_count": 2,
            "stage_dir": "debug_stages",
        },
    )

    with pytest.raises(RuntimeError, match="QC regression gates failed"):
        run_pipeline(cfg)

    assert calls["diagnosis"], "expected diagnosis report generation on QC failure"
    assert calls["export"] == ["stage2_memory", "stage3_refine", "stage4_temporal", "stage5_tuned"]


def test_auto_stage_diagnosis_respects_disable_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_test_frames(frames_dir, count=2)
    calls = _patch_lightweight_pipeline(monkeypatch, tmp_path=tmp_path)

    cfg = VideoMatteConfig(
        io={
            "input": str(frames_dir / "frame_%05d.png"),
            "output_dir": str(tmp_path / "out"),
            "output_alpha": "alpha/%05d.png",
            "frame_start": 0,
            "frame_end": 999999,
        },
        project={"path": str(tmp_path / "out" / "project.vmhqproj"), "autosave": False},
        runtime={"workers_io": 1, "cache_dir": str(tmp_path / ".cache"), "resume": False},
        qc={"enabled": True, "fail_on_regression": True, "auto_stage_diagnosis_on_fail": False},
        debug={
            "export_stage_samples": False,
            "auto_stage_samples_on_qc_fail": True,
            "sample_count": 2,
            "stage_dir": "debug_stages",
        },
    )

    with pytest.raises(RuntimeError, match="QC regression gates failed"):
        run_pipeline(cfg)

    assert calls["diagnosis"] == []
    assert calls["export"] == []
