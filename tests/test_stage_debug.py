from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from videomatte_hq.diagnostics.stage_debug import (
    export_stage_samples,
    resolve_sample_local_frames,
    write_stage_diagnosis_report,
)


def test_resolve_sample_local_frames_maps_absolute_and_local_indices() -> None:
    local = resolve_sample_local_frames(
        num_frames=20,
        frame_start=100,
        sample_frames=[0, 100, 105, 119, 200],
        sample_count=5,
    )
    assert local == [0, 5, 19]


def test_export_stage_samples_and_diagnosis_flags_noisy_transition(tmp_path: Path) -> None:
    h, w = 64, 64
    source = [np.zeros((h, w, 3), dtype=np.uint8)]
    source[0][12:52, 18:46, :] = 200

    clean = np.zeros((h, w), dtype=np.float32)
    clean[14:50, 20:44] = 1.0

    noisy = clean.copy()
    noisy[24:28, 30:34] = 0.0  # hole
    rng = np.random.default_rng(0)
    ys = rng.integers(0, h, size=90)
    xs = rng.integers(0, w, size=90)
    noisy[ys, xs] = 1.0  # speckles

    per_stage = {}
    per_stage["stage2_memory"] = export_stage_samples(
        output_dir=tmp_path,
        stage_dir_name="debug_stages",
        stage_name="stage2_memory",
        source=source,
        alphas=[clean],
        sample_local_frames=[0],
        frame_start=0,
        save_rgb=True,
        save_overlay=True,
    )
    per_stage["stage3_refine"] = export_stage_samples(
        output_dir=tmp_path,
        stage_dir_name="debug_stages",
        stage_name="stage3_refine",
        source=source,
        alphas=[noisy],
        sample_local_frames=[0],
        frame_start=0,
        save_rgb=True,
        save_overlay=True,
    )

    json_path, md_path = write_stage_diagnosis_report(
        output_dir=tmp_path,
        stage_dir_name="debug_stages",
        stage_order=["stage2_memory", "stage3_refine"],
        per_stage_metrics=per_stage,
    )

    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["worst_transition"]["transition"] == "stage2_memory->stage3_refine"
    suspect_rows = [r for r in payload["transitions"] if r.get("suspect")]
    assert len(suspect_rows) >= 1

