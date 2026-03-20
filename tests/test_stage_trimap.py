from __future__ import annotations

import numpy as np

from videomatte_hq.pipeline.stage_trimap import (
    build_trimap_gradient_adaptive,
    build_trimap_hybrid,
    build_trimap_from_logits,
    build_trimap_morphological,
    probability_to_logits,
    resize_binary_mask,
    resize_logits,
    sigmoid_logits,
)


def test_probability_logits_roundtrip() -> None:
    p = np.array([[0.01, 0.5, 0.99]], dtype=np.float32)
    logits = probability_to_logits(p)
    out = sigmoid_logits(logits)
    assert np.allclose(out, p, atol=1e-5)


def test_build_trimap_from_logits_thresholds() -> None:
    logits = np.array([[-8.0, 0.0, 8.0]], dtype=np.float32)
    trimap = build_trimap_from_logits(logits, fg_threshold=0.9, bg_threshold=0.1)
    assert np.allclose(trimap, np.array([[0.0, 0.5, 1.0]], dtype=np.float32))


def test_build_trimap_fallback_band_for_hard_mask_logits() -> None:
    prob = np.zeros((9, 9), dtype=np.float32)
    prob[2:7, 2:7] = 1.0
    logits = probability_to_logits(prob)

    trimap = build_trimap_from_logits(logits, fg_threshold=0.9, bg_threshold=0.1, fallback_band_px=1)

    assert trimap.shape == prob.shape
    assert bool((trimap == 0.5).any())
    assert float(trimap[4, 4]) == 1.0
    assert float(trimap[0, 0]) == 0.0


def test_build_trimap_fallback_triggers_on_thin_unknown_band() -> None:
    """Fallback should activate when the unknown band is < 0.1% of pixels."""
    # Create logits that produce a very thin (nearly empty) unknown band.
    # On a 100x100 grid, 0.1% = 10 pixels. We place only 5 unknown pixels.
    prob = np.zeros((100, 100), dtype=np.float32)
    prob[20:80, 20:80] = 0.95  # above fg_threshold => definite FG
    prob[20, 20:25] = 0.5  # only 5 unknown pixels (0.05% coverage)
    logits = probability_to_logits(prob)

    trimap = build_trimap_from_logits(logits, fg_threshold=0.9, bg_threshold=0.1, fallback_band_px=2)
    unknown_count = int((trimap == 0.5).sum())
    # Fallback band should produce more unknown pixels than the original 5.
    assert unknown_count > 5


def test_resize_helpers_preserve_shape_contract() -> None:
    logits = np.zeros((8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:6, 2:6] = 1.0

    logits_up = resize_logits(logits, (16, 20))
    mask_up = resize_binary_mask(mask, (16, 20), threshold=0.5)
    assert logits_up.shape == (16, 20)
    assert mask_up.shape == (16, 20)
    assert float(mask_up.mean()) > 0.0


def test_build_trimap_morphological_creates_wide_band() -> None:
    """Morphological trimap should create a visible unknown band."""
    mask = np.zeros((200, 200), dtype=np.float32)
    mask[50:150, 50:150] = 1.0

    trimap = build_trimap_morphological(mask, erosion_px=10, dilation_px=10)

    assert trimap.shape == mask.shape
    # Should have all three zones
    assert bool((trimap == 1.0).any()), "No definite FG found"
    assert bool((trimap == 0.0).any()), "No definite BG found"
    assert bool((trimap == 0.5).any()), "No unknown band found"

    # Unknown band should be wider than logit-based approach
    unknown_frac = float((trimap == 0.5).sum()) / trimap.size
    assert unknown_frac > 0.05, f"Unknown band too narrow: {unknown_frac:.4f}"


def test_build_trimap_morphological_band_width_scales() -> None:
    """Wider erosion/dilation should produce more unknown pixels."""
    mask = np.zeros((200, 200), dtype=np.float32)
    mask[50:150, 50:150] = 1.0

    narrow = build_trimap_morphological(mask, erosion_px=5, dilation_px=5)
    wide = build_trimap_morphological(mask, erosion_px=20, dilation_px=20)

    narrow_unknown = int((narrow == 0.5).sum())
    wide_unknown = int((wide == 0.5).sum())
    assert wide_unknown > narrow_unknown


def test_build_trimap_morphological_empty_mask() -> None:
    """Empty mask should return all-zero trimap."""
    mask = np.zeros((100, 100), dtype=np.float32)
    trimap = build_trimap_morphological(mask, erosion_px=10, dilation_px=10)
    assert float(trimap.max()) == 0.0


def test_build_trimap_morphological_full_mask() -> None:
    """Full mask should return all-one trimap."""
    mask = np.ones((100, 100), dtype=np.float32)
    trimap = build_trimap_morphological(mask, erosion_px=10, dilation_px=10)
    assert float(trimap.min()) == 1.0


def test_build_trimap_morphological_center_is_fg() -> None:
    """Center of a large foreground region should remain definite FG."""
    mask = np.zeros((200, 200), dtype=np.float32)
    mask[40:160, 40:160] = 1.0

    trimap = build_trimap_morphological(mask, erosion_px=15, dilation_px=10)
    assert float(trimap[100, 100]) == 1.0  # center is definite FG
    assert float(trimap[0, 0]) == 0.0  # corner is definite BG


def test_build_trimap_morphological_scales_with_resolution() -> None:
    """High-resolution masks should produce proportionally wider unknown bands.

    Pixel values are relative to 1080p (long side 1920).  A 4x larger image
    (long side 7680, i.e. 8K) should auto-scale so the unknown band stays
    visually consistent.
    """
    # 1080p-scale mask (below reference, no scaling)
    mask_1080 = np.zeros((1080, 1920), dtype=np.float32)
    mask_1080[200:880, 400:1520] = 1.0
    trimap_1080 = build_trimap_morphological(mask_1080, erosion_px=20, dilation_px=10)
    frac_1080 = float((trimap_1080 == 0.5).sum()) / trimap_1080.size

    # 4K-scale mask (2x reference, should scale up)
    mask_4k = np.zeros((2160, 3840), dtype=np.float32)
    mask_4k[400:1760, 800:3040] = 1.0
    trimap_4k = build_trimap_morphological(mask_4k, erosion_px=20, dilation_px=10)
    frac_4k = float((trimap_4k == 0.5).sum()) / trimap_4k.size

    # The 4K trimap should have a comparable unknown fraction to 1080p,
    # NOT 4x smaller (which is what would happen without scaling).
    assert frac_4k > frac_1080 * 0.5, (
        f"4K unknown band fraction ({frac_4k:.4f}) is much smaller than "
        f"1080p ({frac_1080:.4f}); resolution scaling may not be working"
    )


def test_build_trimap_hybrid_marks_logit_uncertainty_outside_morph_band_unknown() -> None:
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:24, 8:24] = 1.0

    prob = np.zeros((32, 32), dtype=np.float32)
    prob[8:24, 8:24] = 0.99
    prob[6:26, 6:26] = np.maximum(prob[6:26, 6:26], 0.5)
    logits = probability_to_logits(prob)

    morph = build_trimap_morphological(mask, erosion_px=1, dilation_px=1)
    hybrid = build_trimap_hybrid(
        mask,
        logits,
        erosion_px=1,
        dilation_px=1,
        fg_threshold=0.9,
        bg_threshold=0.1,
        fallback_band_px=1,
    )

    assert float(morph[5, 16]) == 0.0
    assert float(hybrid[5, 16]) == 0.0
    assert float(morph[6, 16]) == 0.0
    assert float(hybrid[6, 16]) == 0.5


def test_build_trimap_hybrid_relaxes_morph_fg_when_logits_are_not_confident() -> None:
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[8:24, 8:24] = 1.0

    prob = np.zeros((32, 32), dtype=np.float32)
    prob[8:24, 8:24] = 0.5
    logits = probability_to_logits(prob)

    hybrid = build_trimap_hybrid(
        mask,
        logits,
        erosion_px=2,
        dilation_px=2,
        fg_threshold=0.9,
        bg_threshold=0.1,
        fallback_band_px=1,
    )

    assert float(hybrid[16, 16]) == 0.5


# ---- Gradient-Adaptive Trimap Tests ----


def test_gradient_adaptive_trimap_output_shape_and_values() -> None:
    """Gradient-adaptive trimap should have correct shape and valid values."""
    h, w = 200, 200
    rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    # Create alpha with soft edges (transition zone) so boundary mask is non-empty
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[60:140, 60:140] = 1.0
    # Add gradient transition (values between bg_thresh and fg_thresh)
    for edge in range(10):
        val = 0.05 + edge * 0.09  # 0.05 to ~0.86
        alpha[50 + edge, 50:150] = val
        alpha[140 + edge, 50:150] = 1.0 - val
        alpha[50:150, 50 + edge] = val
        alpha[50:150, 140 + edge] = 1.0 - val

    trimap = build_trimap_gradient_adaptive(rgb, alpha)

    assert trimap.shape == (h, w)
    assert trimap.dtype == np.float32
    # Should contain all three zones
    unique_vals = set(np.unique(trimap).tolist())
    assert 0.0 in unique_vals, "No definite BG found"
    assert 1.0 in unique_vals, "No definite FG found"
    assert 0.5 in unique_vals, "No unknown band found"


def test_gradient_adaptive_trimap_center_is_fg() -> None:
    """Center of a large foreground region should be definite FG."""
    h, w = 200, 200
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[30:170, 30:170] = 1.0

    trimap = build_trimap_gradient_adaptive(rgb, alpha, fg_thresh=0.95, bg_thresh=0.05)

    assert float(trimap[100, 100]) == 1.0  # center is definite FG
    assert float(trimap[0, 0]) == 0.0  # corner is definite BG


def test_gradient_adaptive_trimap_high_gradient_widens_band() -> None:
    """Regions with high image gradients should produce wider unknown bands."""
    h, w = 200, 200

    # Uniform image (low gradient everywhere)
    rgb_flat = np.full((h, w, 3), 128, dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[50:150, 50:150] = 1.0
    trimap_flat = build_trimap_gradient_adaptive(rgb_flat, alpha, base_kernel=5, max_extra=15, gradient_scale=1.0)

    # Image with strong edges around the mask boundary (high gradient)
    rgb_edge = np.full((h, w, 3), 128, dtype=np.uint8)
    rgb_edge[48:52, :] = 255
    rgb_edge[148:152, :] = 255
    rgb_edge[:, 48:52] = 255
    rgb_edge[:, 148:152] = 255
    trimap_edge = build_trimap_gradient_adaptive(rgb_edge, alpha, base_kernel=5, max_extra=15, gradient_scale=1.0)

    unknown_flat = int((trimap_flat == 0.5).sum())
    unknown_edge = int((trimap_edge == 0.5).sum())
    # High-gradient image should produce at least as many unknown pixels
    assert unknown_edge >= unknown_flat


def test_gradient_adaptive_trimap_empty_alpha() -> None:
    """Empty alpha should return mostly background."""
    h, w = 100, 100
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.float32)

    trimap = build_trimap_gradient_adaptive(rgb, alpha)
    assert float(trimap.max()) == 0.0


def test_gradient_adaptive_trimap_full_alpha() -> None:
    """Full alpha should return mostly foreground."""
    h, w = 100, 100
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.ones((h, w), dtype=np.float32)

    trimap = build_trimap_gradient_adaptive(rgb, alpha)
    assert float(trimap.min()) == 1.0
