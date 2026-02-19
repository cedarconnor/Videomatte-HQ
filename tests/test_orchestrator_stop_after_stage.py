from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.pipeline.orchestrator import run_pipeline


def _write_test_frames(frames_dir: Path, count: int = 2) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        rgb = np.zeros((24, 24, 3), dtype=np.uint8)
        rgb[..., 0] = 30 + idx * 20
        rgb[..., 1] = 80
        rgb[..., 2] = 120
        cv2.imwrite(str(frames_dir / f"frame_{idx:05d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def test_stop_after_memory_writes_stage_preview(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import videomatte_hq.pipeline.orchestrator as orch

    frames_dir = tmp_path / "frames"
    _write_test_frames(frames_dir, count=2)

    key_mask = np.zeros((24, 24), dtype=np.float32)
    key_mask[6:18, 8:16] = 1.0

    monkeypatch.setattr(orch, "ensure_project", lambda _cfg: (tmp_path / "out" / "project.vmhqproj", object()))
    monkeypatch.setattr(orch, "load_keyframe_masks", lambda *_args, **_kwargs: {0: key_mask})
    monkeypatch.setattr(orch, "build_memory_region_priors", lambda **_kwargs: None)
    monkeypatch.setattr(orch, "save_project", lambda *_args, **_kwargs: None)

    def _make_alphas(source, value: float) -> list[np.ndarray]:
        return [np.full(source[i].shape[:2], value, dtype=np.float32) for i in range(len(source))]

    monkeypatch.setattr(
        orch,
        "run_pass_memory",
        lambda source, **_kwargs: (_make_alphas(source, 0.4), _make_alphas(source, 0.6)),
    )
    monkeypatch.setattr(orch, "run_pass_refine", lambda source, *_args, **_kwargs: _make_alphas(source, 0.5))
    monkeypatch.setattr(
        orch,
        "run_pass_temporal_cleanup",
        lambda source, **_kwargs: _make_alphas(source, 0.55),
    )
    monkeypatch.setattr(
        orch,
        "run_pass_matte_tuning",
        lambda alphas, **_kwargs: [np.asarray(a, dtype=np.float32) for a in alphas],
    )

    cfg = VideoMatteConfig(
        io={
            "input": str(frames_dir / "frame_%05d.png"),
            "output_dir": str(tmp_path / "out"),
            "output_alpha": "alpha/frame_%05d.png",
            "frame_start": 0,
            "frame_end": 999999,
        },
        project={"path": str(tmp_path / "out" / "project.vmhqproj"), "autosave": False},
        runtime={
            "workers_io": 1,
            "cache_dir": str(tmp_path / ".cache"),
            "resume": False,
            "stop_after_stage": "memory",
        },
    )

    run_pipeline(cfg)

    stage_preview = tmp_path / "out" / "stages" / "stage2_memory" / "alpha" / "frame_00000.png"
    final_alpha = tmp_path / "out" / "alpha" / "frame_00000.png"
    assert stage_preview.exists(), "expected stage preview output for memory stop"
    assert not final_alpha.exists(), "final alpha output should not be written when stopping after memory stage"

    stage_hashes = json.loads((tmp_path / ".cache" / "stage_hashes.json").read_text(encoding="utf-8"))
    assert "memory" in stage_hashes
    assert "io" not in stage_hashes

