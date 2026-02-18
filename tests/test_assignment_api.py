from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq_web.server import (
    AssignmentFramePreviewRequest,
    BuildAssignmentMaskRequest,
    BuildAssignmentMaskRangeRequest,
    ImportAssignmentRequest,
    PathInfoRequest,
    PromptBox,
    PromptPoint,
    PropagateAssignmentMasksRequest,
    ProjectStateRequest,
    SuggestAssignmentBoxesRequest,
    SuggestReprocessRangeRequest,
    assignment_frame_preview,
    build_assignment_mask,
    build_assignment_mask_range,
    import_assignment,
    list_input_suggestions,
    path_info,
    propagate_assignment_masks,
    project_state,
    suggest_assignment_range,
    suggest_assignment_boxes,
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


@pytest.mark.anyio
async def test_input_suggestions_and_path_info(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    test_files = tmp_path / "TestFiles"
    test_files.mkdir(parents=True, exist_ok=True)
    (test_files / "notes.txt").write_text("ignore", encoding="utf-8")
    (test_files / "shot_b.mp4").write_bytes(b"")
    (test_files / "shot_a.mov").write_bytes(b"")

    suggestions = await list_input_suggestions()
    assert suggestions["status"] == "ok"
    assert len(suggestions["paths"]) == 2
    assert suggestions["paths"][0].endswith("shot_a.mov")
    assert suggestions["paths"][1].endswith("shot_b.mp4")

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    info = await path_info(PathInfoRequest(path="output"))
    assert info["status"] == "ok"
    assert info["exists"] is True
    assert info["is_dir"] is True
    assert info["is_file"] is False


@pytest.mark.anyio
async def test_assignment_mask_builder_sam_fallback_to_grabcut(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame = np.zeros((72, 96, 3), dtype=np.uint8)
    frame[16:60, 30:74, :] = 210
    assert cv2.imwrite(str(frames_dir / "frame_00000.png"), frame)

    out_dir = tmp_path / "out_builder_fallback"
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

    # Force SAM path to fail so API must use grabcut fallback.
    import videomatte_hq_web.server as server_mod

    def _raise_sam(*args, **kwargs):
        raise RuntimeError("sam model unavailable in test")

    monkeypatch.setattr(server_mod, "build_prompt_mask_sam", _raise_sam)

    built = await build_assignment_mask(
        BuildAssignmentMaskRequest(
            config=cfg.model_dump(mode="json"),
            frame=0,
            kind="initial",
            backend="sam",
            sam_model_id="facebook/sam-vit-base",
            sam_local_files_only=True,
            sam_fallback_to_grabcut=True,
            box=PromptBox(x0=24, y0=10, x1=82, y1=64),
            fg_points=[PromptPoint(x=50, y=40)],
            bg_points=[PromptPoint(x=8, y=8)],
            point_radius=6,
            iter_count=4,
        )
    )
    assert built["status"] == "ok"
    assert built["backend_requested"] == "sam"
    assert built["backend_used"] == "grabcut_fallback"
    assert "SAM unavailable" in str(built.get("builder_note"))


@pytest.mark.anyio
async def test_assignment_mask_builder_sam_range_with_prompt_tracking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    h, w = 72, 112
    for idx in range(6):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x0 = 20 + idx * 4
        frame[18:62, x0 : x0 + 28, :] = 215
        assert cv2.imwrite(str(frames_dir / f"frame_{idx:05d}.png"), frame)

    out_dir = tmp_path / "out_builder_range"
    cfg = VideoMatteConfig(
        io={
            "input": "frames/frame_%05d.png",
            "output_dir": str(out_dir),
            "output_alpha": "alpha/%05d.png",
            "frame_start": 0,
            "frame_end": 5,
        },
        assignment={"require_assignment": True},
        project={"path": str(out_dir / "project.vmhqproj")},
    )

    import videomatte_hq_web.server as server_mod
    from videomatte_hq.prompt_mask_range import PromptMaskRangeResult

    def _fake_build_prompt_masks_range(**kwargs):
        start = int(kwargs["frame_start"])
        end = int(kwargs["frame_end"])
        masks: dict[int, np.ndarray] = {}
        for local_idx in range(start, end + 1):
            alpha = np.zeros((h, w), dtype=np.float32)
            x0 = 20 + local_idx * 4
            alpha[18:62, x0 : x0 + 28] = 1.0
            masks[int(local_idx)] = alpha
        return PromptMaskRangeResult(
            masks=masks,
            backend_used="sam2_video_predictor",
            note=None,
        )

    monkeypatch.setattr(server_mod, "build_prompt_masks_range", _fake_build_prompt_masks_range)

    built = await build_assignment_mask_range(
        BuildAssignmentMaskRangeRequest(
            config=cfg.model_dump(mode="json"),
            anchor_frame=0,
            frame_start=0,
            frame_end=5,
            backend="sam2_video_predictor",
            box=PromptBox(x0=18, y0=12, x1=58, y1=66),
            fg_points=[PromptPoint(x=30, y=36)],
            bg_points=[PromptPoint(x=6, y=6)],
            save_stride=1,
            track_prompts_with_flow=True,
            kind="initial",
        )
    )

    assert built["status"] == "ok"
    assert built["backend_requested"] == "sam2_video_predictor"
    assert built["backend_used"] == "sam2_video_predictor"
    assert built["inserted_count"] == 6
    assert built["keyframe_count"] == 6
    assert built["suggested_reprocess_range"]["frame_start"] == 0
    assert built["suggested_reprocess_range"]["frame_end"] == 5


@pytest.mark.anyio
async def test_assignment_mask_builder_range_surfaces_runtime_dependency_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    frame[16:52, 24:60, :] = 220
    assert cv2.imwrite(str(frames_dir / "frame_00000.png"), frame)

    out_dir = tmp_path / "out_builder_runtime_hint"
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

    import videomatte_hq_web.server as server_mod

    def _raise_runtime_error(**kwargs):
        raise RuntimeError("OSError: [WinError 127] The specified procedure could not be found (c10_cuda.dll)")

    monkeypatch.setattr(server_mod, "build_prompt_masks_range", _raise_runtime_error)

    with pytest.raises(Exception) as exc:
        await build_assignment_mask_range(
            BuildAssignmentMaskRangeRequest(
                config=cfg.model_dump(mode="json"),
                anchor_frame=0,
                frame_start=0,
                frame_end=0,
                backend="sam2_video_predictor",
                box=PromptBox(x0=20, y0=10, x1=62, y1=56),
                fg_points=[PromptPoint(x=34, y=34)],
                bg_points=[PromptPoint(x=4, y=4)],
                save_stride=1,
                kind="initial",
            )
        )
    err_text = str(exc.value)
    assert "Runtime dependency issue detected" in err_text
    assert "WinError 127" in err_text
    assert ".venv\\Scripts\\python" in err_text


@pytest.mark.anyio
async def test_assignment_phase4_propagation_flow_inserts_keyframes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    h, w = 72, 112
    for idx in range(12):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x0 = 16 + idx * 2
        x1 = x0 + 28
        frame[18:62, x0:x1, :] = 220
        assert cv2.imwrite(str(frames_dir / f"frame_{idx:05d}.png"), frame)

    out_dir = tmp_path / "out_phase4"
    cfg = VideoMatteConfig(
        io={
            "input": "frames/frame_%05d.png",
            "output_dir": str(out_dir),
            "output_alpha": "alpha/%05d.png",
            "frame_start": 0,
            "frame_end": 11,
        },
        assignment={"require_assignment": True},
        project={"path": str(out_dir / "project.vmhqproj")},
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[18:62, 16:44] = 255
    mask_path = tmp_path / "mask_anchor_0000.png"
    assert cv2.imwrite(str(mask_path), mask)

    imported = await import_assignment(
        ImportAssignmentRequest(
            config=cfg.model_dump(mode="json"),
            frame=0,
            mask_path=str(mask_path),
            source="test",
            kind="initial",
        )
    )
    assert imported["status"] == "ok"
    assert imported["keyframe_count"] == 1

    import videomatte_hq_web.server as server_mod
    from videomatte_hq.propagation_assist import PropagationAssistResult

    def _fake_propagate_masks_assist(**kwargs):
        start = int(kwargs["frame_start"])
        end = int(kwargs["frame_end"])
        assert str(kwargs.get("backend", "")) == "sam2_video_predictor"
        masks: dict[int, np.ndarray] = {}
        for local_idx in range(start, end + 1):
            alpha = np.zeros((h, w), dtype=np.float32)
            x0 = 16 + local_idx * 2
            alpha[18:62, x0:x0 + 28] = 1.0
            masks[int(local_idx)] = alpha
        return PropagationAssistResult(
            masks=masks,
            backend_used="sam2_video_predictor",
            note=None,
        )

    monkeypatch.setattr(server_mod, "propagate_masks_assist", _fake_propagate_masks_assist)

    propagated = await propagate_assignment_masks(
        PropagateAssignmentMasksRequest(
            config=cfg.model_dump(mode="json"),
            anchor_frame=0,
            frame_start=0,
            frame_end=11,
            backend="sam2_video_predictor",
            stride=3,
            max_new_keyframes=5,
            flow_downscale=0.5,
            flow_min_coverage=0.002,
            flow_max_coverage=0.98,
            flow_feather_px=1,
            fallback_to_flow=False,
        )
    )
    assert propagated["status"] == "ok"
    assert propagated["backend_requested"] == "sam2_video_predictor"
    assert propagated["backend_used"] == "sam2_video_predictor"
    assert propagated["inserted_count"] >= 3
    assert propagated["keyframe_count"] >= 4
    assert float(propagated["mean_inserted_coverage"]) > 0.02


@pytest.mark.anyio
async def test_assignment_phase4_propagation_sam2_no_flow_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    h, w = 64, 96
    for idx in range(8):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[16:52, 18 + idx:46 + idx, :] = 210
        assert cv2.imwrite(str(frames_dir / f"frame_{idx:05d}.png"), frame)

    out_dir = tmp_path / "out_phase4_fallback"
    cfg = VideoMatteConfig(
        io={
            "input": "frames/frame_%05d.png",
            "output_dir": str(out_dir),
            "output_alpha": "alpha/%05d.png",
            "frame_start": 0,
            "frame_end": 7,
        },
        assignment={"require_assignment": True},
        project={"path": str(out_dir / "project.vmhqproj")},
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[16:52, 18:46] = 255
    mask_path = tmp_path / "mask_anchor_0000.png"
    assert cv2.imwrite(str(mask_path), mask)

    imported = await import_assignment(
        ImportAssignmentRequest(
            config=cfg.model_dump(mode="json"),
            frame=0,
            mask_path=str(mask_path),
            source="test",
            kind="initial",
        )
    )
    assert imported["status"] == "ok"

    import videomatte_hq_web.server as server_mod

    def _raise_no_fallback(**kwargs):
        raise RuntimeError("sam2 runtime unavailable")

    monkeypatch.setattr(server_mod, "propagate_masks_assist", _raise_no_fallback)

    with pytest.raises(Exception) as exc:
        await propagate_assignment_masks(
            PropagateAssignmentMasksRequest(
                config=cfg.model_dump(mode="json"),
                anchor_frame=0,
                frame_start=0,
                frame_end=7,
                backend="sam2_video_predictor",
                fallback_to_flow=True,
                stride=2,
                max_new_keyframes=4,
            )
        )
    assert "sam2 runtime unavailable" in str(exc.value)


@pytest.mark.anyio
async def test_assignment_phase4_propagation_samurai_alias_normalized_to_sam2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    h, w = 64, 96
    for idx in range(8):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[16:52, 18 + idx:46 + idx, :] = 210
        assert cv2.imwrite(str(frames_dir / f"frame_{idx:05d}.png"), frame)

    out_dir = tmp_path / "out_phase4_samurai_fallback"
    cfg = VideoMatteConfig(
        io={
            "input": "frames/frame_%05d.png",
            "output_dir": str(out_dir),
            "output_alpha": "alpha/%05d.png",
            "frame_start": 0,
            "frame_end": 7,
        },
        assignment={"require_assignment": True},
        project={"path": str(out_dir / "project.vmhqproj")},
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[16:52, 18:46] = 255
    mask_path = tmp_path / "mask_anchor_0000.png"
    assert cv2.imwrite(str(mask_path), mask)

    imported = await import_assignment(
        ImportAssignmentRequest(
            config=cfg.model_dump(mode="json"),
            frame=0,
            mask_path=str(mask_path),
            source="test",
            kind="initial",
        )
    )
    assert imported["status"] == "ok"

    import videomatte_hq_web.server as server_mod
    from videomatte_hq.propagation_assist import PropagationAssistResult

    def _fake_propagate_masks_assist(**kwargs):
        start = int(kwargs["frame_start"])
        end = int(kwargs["frame_end"])
        assert str(kwargs.get("backend", "")) == "sam2_video_predictor"
        masks: dict[int, np.ndarray] = {}
        for local_idx in range(start, end + 1):
            alpha = np.zeros((h, w), dtype=np.float32)
            alpha[16:52, 18 + local_idx:46 + local_idx] = 1.0
            masks[int(local_idx)] = alpha
        return PropagationAssistResult(
            masks=masks,
            backend_used="sam2_video_predictor",
            note=None,
        )

    monkeypatch.setattr(server_mod, "propagate_masks_assist", _fake_propagate_masks_assist)

    propagated = await propagate_assignment_masks(
        PropagateAssignmentMasksRequest(
            config=cfg.model_dump(mode="json"),
            anchor_frame=0,
            frame_start=0,
            frame_end=7,
            backend="samurai_video_predictor",
            fallback_to_flow=False,
            stride=2,
            max_new_keyframes=4,
            samurai_model_cfg="sam2.1_hiera_l.yaml",
            samurai_checkpoint="checkpoints/sam2.1_hiera_large.pt",
        )
    )
    assert propagated["status"] == "ok"
    assert propagated["backend_requested"] == "sam2_video_predictor"
    assert propagated["backend_used"] == "sam2_video_predictor"
    assert propagated["inserted_count"] >= 2
