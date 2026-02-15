from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq_web.server import (
    ImportAssignmentRequest,
    ProjectStateRequest,
    import_assignment,
    project_state,
    suggest_assignment_range,
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
