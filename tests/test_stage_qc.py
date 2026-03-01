from __future__ import annotations

import numpy as np

from videomatte_hq.pipeline.stage_qc import area_jitter, centroid_jitter, check_drift, compute_boundary_iou, compute_iou


def test_compute_iou_expected_values() -> None:
    a = np.zeros((16, 16), dtype=np.float32)
    b = np.zeros((16, 16), dtype=np.float32)
    a[4:12, 4:12] = 1.0
    b[4:12, 4:12] = 1.0
    assert compute_iou(a, b) == 1.0

    c = np.zeros((16, 16), dtype=np.float32)
    c[:4, :4] = 1.0
    assert compute_iou(a, c) == 0.0


def test_check_drift_flags_large_change() -> None:
    prev_mask = np.zeros((32, 32), dtype=np.float32)
    prev_mask[8:24, 8:24] = 1.0

    cur_mask = np.zeros((32, 32), dtype=np.float32)
    cur_mask[0:8, 0:8] = 1.0

    drift = check_drift(cur_mask, prev_mask, iou_threshold=0.7, area_change_threshold=0.4)
    assert drift.drift is True
    assert drift.iou < 0.2


def test_compute_boundary_iou_detects_edge_shift() -> None:
    """Boundary IoU should be sensitive to edge displacement even when mask IoU is high."""
    a = np.zeros((128, 128), dtype=np.float32)
    a[24:104, 24:104] = 1.0

    # Shift by 2 pixels — mask IoU remains high, but boundary IoU drops.
    b = np.zeros((128, 128), dtype=np.float32)
    b[26:106, 26:106] = 1.0

    mask_iou = compute_iou(a, b)
    boundary_iou = compute_boundary_iou(a, b, boundary_width=2)
    assert mask_iou > 0.90
    assert boundary_iou < mask_iou


def test_compute_boundary_iou_identical_masks() -> None:
    a = np.zeros((32, 32), dtype=np.float32)
    a[8:24, 8:24] = 1.0
    assert compute_boundary_iou(a, a) == 1.0


def test_jitter_metrics_shapes() -> None:
    masks = []
    for x in range(4):
        m = np.zeros((20, 20), dtype=np.float32)
        m[8:12, 8 + x : 12 + x] = 1.0
        masks.append(m)
    assert area_jitter(masks).shape == (3,)
    assert centroid_jitter(masks).shape == (3,)
