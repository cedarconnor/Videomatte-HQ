from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq_web.server import (
    AssignmentFramePreviewRequest,
    BuildAssignmentMaskRequest,
    PromptBox,
    PromptPoint,
    ImportAssignmentRequest,
    ProjectStateRequest,
    assignment_frame_preview,
    build_assignment_mask,
    import_assignment,
    project_state,
    SuggestAssignmentBoxesRequest,
    suggest_assignment_range,
    suggest_assignment_boxes,
    SuggestReprocessRangeRequest,
)


@pytest.mark.anyio
async def test_assignment_import_and_project_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    out_dir = tmp_path / "out_ui"
    cfg = VideoMatteConfig(
        io={
            "input": "frames/%06d.png",
            "output_dir": str(out_dir),
            "output_alpha": "alpha/%06d.png",
        },
        assignment={"require_assignment": True},
        project={"path": str(out_dir / "project.vmhqproj")},
    )

    # Initial project state: no keyframes yet.
    state_0 = await project_state(ProjectStateRequest(config=cfg.model_dump(mode="json")))
    assert state_0["keyframe_count"] == 0

    # Create a synthetic keyframe mask and import it.
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 5:11] = 255
    mask_path = tmp_path / "mask_0003.png"
    assert cv2.imwrite(str(mask_path), mask)

    imported = await import_assignment(
        ImportAssignmentRequest(
            config=cfg.model_dump(mode="json"),
            frame=3,
            mask_path=str(mask_path),
            source="test",
            kind="correction",
        )
    )
    assert imported["status"] == "ok"
    assert imported["keyframe_count"] == 1
    assert imported["assignment"]["frame"] == 3
    assert imported["assignment"]["kind"] == "correction"
    assert imported["suggested_reprocess_range"]["frame_start"] <= 3

    # State should now include the keyframe assignment.
    state_1 = await project_state(ProjectStateRequest(config=cfg.model_dump(mode="json")))
    assert state_1["keyframe_count"] == 1
    assert state_1["keyframes"][0]["frame"] == 3
    assert state_1["keyframes"][0]["kind"] == "correction"

    suggested = await suggest_assignment_range(
        SuggestReprocessRangeRequest(
            config=cfg.model_dump(mode="json"),
            frame=3,
        )
    )
    assert suggested["status"] == "ok"
    assert suggested["frame_start"] <= 3


@pytest.mark.anyio
async def test_assignment_mask_builder_preview_and_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    frame[20:70, 36:90, :] = 220
    assert cv2.imwrite(str(frames_dir / "frame_00000.png"), frame)

    out_dir = tmp_path / "out_builder"
    cfg = VideoMatteConfig(
        io={
            "input": "frames/frame_%05d.png",
            "output_dir": str(out_dir),
            "output_alpha": "alpha/%05d.png",
            "frame_start": 0,
            "frame_end": 0,
        },
        assignment={"require_assignment": True},
        project={"path": str(out_dir / "project.vmhqproj")},
    )

    preview = await assignment_frame_preview(
        AssignmentFramePreviewRequest(
            config=cfg.model_dump(mode="json"),
            frame=0,
        )
    )
    assert preview["status"] == "ok"
    assert preview["width"] == 120
    assert preview["height"] == 80
    assert str(preview["data_url"]).startswith("data:image/png;base64,")

    built = await build_assignment_mask(
        BuildAssignmentMaskRequest(
            config=cfg.model_dump(mode="json"),
            frame=0,
            kind="initial",
            box=PromptBox(x0=28, y0=14, x1=98, y1=74),
            fg_points=[PromptPoint(x=54, y=45)],
            bg_points=[PromptPoint(x=8, y=8)],
            point_radius=6,
            iter_count=4,
        )
    )
    assert built["status"] == "ok"
    assert built["keyframe_count"] == 1
    assert float(built["coverage"]) > 0.05
    assert str(built["mask_preview_data_url"]).startswith("data:image/png;base64,")

    suggested_boxes = await suggest_assignment_boxes(
        SuggestAssignmentBoxesRequest(
            config=cfg.model_dump(mode="json"),
            frame=0,
            prompt="object center",
            max_candidates=4,
        )
    )
    assert suggested_boxes["status"] == "ok"
    assert len(suggested_boxes["candidates"]) >= 1
