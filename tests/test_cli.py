from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from videomatte_hq.cli import _resolve_auto_anchor, _run_preflight_checks
from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.prompts.auto_anchor import AutoAnchorResult


def test_resolve_auto_anchor_updates_anchor_and_effective_start(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"dummy")

    out_dir = tmp_path / "out"
    generated = out_dir / "anchor_mask.auto.png"

    def _fake_build(*args, **kwargs):
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_bytes(b"mask")
        return AutoAnchorResult(mask_path=generated, method="fake", probe_frame=3)

    monkeypatch.setattr("videomatte_hq.cli.build_auto_anchor_mask_for_video", _fake_build)

    cfg = VideoMatteConfig(
        input=str(video_path),
        output_dir=str(out_dir),
        anchor_mask="",
        frame_start=0,
        refine_enabled=False,
        segment_backend="static",
    )
    args = Namespace(auto_anchor=None, auto_anchor_output=None)

    result = _resolve_auto_anchor(cfg, args)

    assert result is not None
    assert result.method == "fake"
    assert cfg.anchor_mask == str(generated)
    assert cfg.frame_start == 3


def test_preflight_checks_require_mematte_assets_when_refine_enabled(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    video_path = repo_root / "clip.mp4"
    video_path.write_bytes(b"dummy")
    anchor = repo_root / "anchor.png"
    anchor.write_bytes(b"dummy")
    missing_repo = repo_root / "missing_mematte"

    cfg = VideoMatteConfig(
        input=str(video_path),
        anchor_mask=str(anchor),
        refine_enabled=True,
        mematte_repo_dir=str(missing_repo),
        mematte_checkpoint=str(missing_repo / "checkpoints" / "MEMatte_ViTS_DIM.pth"),
        segment_backend="static",
    )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("videomatte_hq.cli._repo_root", lambda: repo_root)
    try:
        with pytest.raises(FileNotFoundError, match="MEMatte repo dir not found"):
            _run_preflight_checks(cfg)
    finally:
        monkeypatch.undo()


def test_preflight_checks_validate_invalid_frame_range(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"dummy")
    anchor = tmp_path / "anchor.png"
    anchor.write_bytes(b"dummy")

    cfg = VideoMatteConfig(
        input=str(video_path),
        anchor_mask=str(anchor),
        frame_start=10,
        frame_end=5,
        refine_enabled=False,
        segment_backend="static",
    )

    with pytest.raises(ValueError, match="Invalid frame range"):
        _run_preflight_checks(cfg)


def test_preflight_checks_reject_unsupported_anchor_frame(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"dummy")
    anchor = tmp_path / "anchor.png"
    anchor.write_bytes(b"dummy")

    cfg = VideoMatteConfig(
        input=str(video_path),
        anchor_mask=str(anchor),
        anchor_frame=3,
        refine_enabled=False,
        segment_backend="static",
    )

    with pytest.raises(ValueError, match="Only anchor_frame=0"):
        _run_preflight_checks(cfg)


def test_preflight_checks_reject_no_refine_mode(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"dummy")
    anchor = tmp_path / "anchor.png"
    anchor.write_bytes(b"dummy")

    cfg = VideoMatteConfig(
        input=str(video_path),
        anchor_mask=str(anchor),
        refine_enabled=False,
        segment_backend="static",
    )

    with pytest.raises(ValueError, match="MEMatte refinement is mandatory"):
        _run_preflight_checks(cfg)


def test_preflight_checks_reject_mematte_paths_outside_repo(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    local_video = repo_root / "clip.mp4"
    local_video.write_bytes(b"dummy")
    local_anchor = repo_root / "anchor.png"
    local_anchor.write_bytes(b"dummy")

    external_root = tmp_path / "external_mematte"
    external_root.mkdir(parents=True, exist_ok=True)
    (external_root / "inference.py").write_text("# dummy", encoding="utf-8")
    ckpt = external_root / "checkpoints" / "MEMatte_ViTS_DIM.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"dummy")

    monkeypatch.setattr("videomatte_hq.cli._repo_root", lambda: repo_root)

    cfg = VideoMatteConfig(
        input=str(local_video),
        anchor_mask=str(local_anchor),
        refine_enabled=True,
        mematte_repo_dir=str(external_root),
        mematte_checkpoint=str(ckpt),
        segment_backend="static",
    )

    with pytest.raises(ValueError, match="must be inside this repository"):
        _run_preflight_checks(cfg)


def test_preflight_checks_allow_mematte_paths_outside_repo_when_enabled(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    local_video = repo_root / "clip.mp4"
    local_video.write_bytes(b"dummy")
    local_anchor = repo_root / "anchor.png"
    local_anchor.write_bytes(b"dummy")

    external_root = tmp_path / "external_mematte"
    external_root.mkdir(parents=True, exist_ok=True)
    (external_root / "inference.py").write_text("# dummy", encoding="utf-8")
    ckpt = external_root / "checkpoints" / "MEMatte_ViTS_DIM.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"dummy")

    monkeypatch.setattr("videomatte_hq.cli._repo_root", lambda: repo_root)

    cfg = VideoMatteConfig(
        input=str(local_video),
        anchor_mask=str(local_anchor),
        refine_enabled=True,
        mematte_repo_dir=str(external_root),
        mematte_checkpoint=str(ckpt),
        segment_backend="static",
    )

    _run_preflight_checks(cfg, allow_external_paths=True)

    assert Path(cfg.mematte_repo_dir) == external_root.resolve()
    assert Path(cfg.mematte_checkpoint) == ckpt.resolve()


def test_preflight_checks_normalize_relative_mematte_paths_under_repo(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    local_video = repo_root / "clip.mp4"
    local_video.write_bytes(b"dummy")
    local_anchor = repo_root / "anchor.png"
    local_anchor.write_bytes(b"dummy")

    mematte_repo = repo_root / "third_party" / "MEMatte"
    mematte_repo.mkdir(parents=True, exist_ok=True)
    (mematte_repo / "inference.py").write_text("# dummy", encoding="utf-8")
    ckpt = mematte_repo / "checkpoints" / "MEMatte_ViTS_DIM.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"dummy")

    monkeypatch.setattr("videomatte_hq.cli._repo_root", lambda: repo_root)

    cfg = VideoMatteConfig(
        input=str(local_video),
        anchor_mask=str(local_anchor),
        refine_enabled=True,
        mematte_repo_dir="third_party/MEMatte",
        mematte_checkpoint="third_party/MEMatte/checkpoints/MEMatte_ViTS_DIM.pth",
        segment_backend="static",
    )

    _run_preflight_checks(cfg)

    assert Path(cfg.mematte_repo_dir) == mematte_repo.resolve()
    assert Path(cfg.mematte_checkpoint) == ckpt.resolve()
