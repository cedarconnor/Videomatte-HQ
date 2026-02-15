from __future__ import annotations

import numpy as np

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.pipeline.pass_matte_tuning import run_pass_matte_tuning


def test_matte_tuning_noop_returns_same_alpha() -> None:
    alpha = np.zeros((32, 32), dtype=np.float32)
    alpha[10:22, 12:20] = 1.0
    cfg = VideoMatteConfig(
        matte_tuning={
            "enabled": True,
            "shrink_grow_px": 0,
            "feather_px": 0,
            "offset_x_px": 0,
            "offset_y_px": 0,
        }
    )

    out = run_pass_matte_tuning([alpha], cfg)[0]
    assert np.allclose(out, alpha, atol=1e-6)


def test_matte_tuning_shrink_and_grow_change_coverage() -> None:
    alpha = np.zeros((48, 48), dtype=np.float32)
    alpha[16:32, 16:32] = 1.0

    cfg_grow = VideoMatteConfig(matte_tuning={"shrink_grow_px": 2})
    cfg_shrink = VideoMatteConfig(matte_tuning={"shrink_grow_px": -2})

    grown = run_pass_matte_tuning([alpha], cfg_grow)[0]
    shrunk = run_pass_matte_tuning([alpha], cfg_shrink)[0]

    assert float(grown.mean()) > float(alpha.mean())
    assert float(shrunk.mean()) < float(alpha.mean())


def test_matte_tuning_offset_translates_mask() -> None:
    alpha = np.zeros((20, 20), dtype=np.float32)
    alpha[8, 8] = 1.0

    cfg = VideoMatteConfig(
        matte_tuning={
            "offset_x_px": 3,
            "offset_y_px": -2,
        }
    )
    out = run_pass_matte_tuning([alpha], cfg)[0]

    assert float(out[6, 11]) > 0.9
    assert float(out[8, 8]) < 0.1


def test_matte_tuning_feather_softens_binary_edges() -> None:
    alpha = np.zeros((40, 40), dtype=np.float32)
    alpha[10:30, 10:30] = 1.0
    cfg = VideoMatteConfig(
        matte_tuning={
            "feather_px": 2,
        }
    )

    out = run_pass_matte_tuning([alpha], cfg)[0]
    edge_value = float(out[10, 20])
    center_value = float(out[20, 20])
    bg_value = float(out[0, 0])

    assert 0.05 < edge_value < 0.95
    assert center_value > edge_value
    assert bg_value < 0.01
