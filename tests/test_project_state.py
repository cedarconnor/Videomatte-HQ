from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.project import (
    ensure_project,
    import_keyframe_mask,
    load_keyframe_masks,
    load_project,
    suggest_reprocess_range,
)


def test_project_and_keyframe_mask_roundtrip(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    project_path = out_dir / "project.vmhqproj"

    cfg = VideoMatteConfig(
        io={"input": "frames/%06d.png", "output_dir": str(out_dir)},
        project={"path": str(project_path), "autosave": True},
    )

    project_file, project = ensure_project(cfg)
    assert project_file == project_path
    assert project_file.exists()

    # Build a synthetic binary mask.
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 10:22] = 255
    src_mask = tmp_path / "mask.png"
    assert cv2.imwrite(str(src_mask), mask)

    assignment = import_keyframe_mask(
        cfg=cfg,
        project_path=project_file,
        project=project,
        frame=5,
        mask_path=src_mask,
        source="test",
    )
    assert assignment.frame == 5

    reloaded = load_project(project_file)
    assert reloaded is not None
    assert len(reloaded.keyframes) == 1
    assert reloaded.keyframes[0].frame == 5

    masks = load_keyframe_masks(project_file, reloaded, target_shape=(32, 32))
    assert 5 in masks
    assert masks[5].shape == (32, 32)
    assert float(masks[5][12, 12]) > 0.99
    assert float(masks[5][0, 0]) < 0.01


def test_suggest_reprocess_range_uses_neighbor_midpoints(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    project_path = out_dir / "project.vmhqproj"

    cfg = VideoMatteConfig(
        io={"input": "frames/%06d.png", "output_dir": str(out_dir), "frame_start": 0, "frame_end": 100},
        project={"path": str(project_path), "autosave": True},
        memory={"window": 20},
    )

    project_file, project = ensure_project(cfg)

    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    src_mask = tmp_path / "mask.png"
    assert cv2.imwrite(str(src_mask), mask)

    import_keyframe_mask(cfg, project_file, project, frame=10, mask_path=src_mask, source="test")
    import_keyframe_mask(cfg, project_file, project, frame=50, mask_path=src_mask, source="test")
    import_keyframe_mask(cfg, project_file, project, frame=90, mask_path=src_mask, source="test")

    start, end = suggest_reprocess_range(
        project=project,
        anchor_frame=50,
        memory_window=cfg.memory.window,
        clip_start=cfg.io.frame_start,
        clip_end=cfg.io.frame_end,
    )
    assert start == 30
    assert end == 70
