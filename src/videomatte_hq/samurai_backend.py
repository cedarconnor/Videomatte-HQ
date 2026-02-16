"""Optional Samurai/SAM2 video-predictor integration helpers."""

from __future__ import annotations

import importlib
import inspect
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is a project dependency, but keep adapter defensive.
    torch = None


SAMURAI_BACKEND_CANONICAL = "samurai_video_predictor"
SAMURAI_BACKEND_ALIASES = {
    "samurai",
    "samurai_video_predictor",
    "samurai_tracker",
    "samurai_video",
}


class SamuraiUnavailable(RuntimeError):
    """Raised when Samurai runtime is requested but not available."""


@dataclass(frozen=True)
class SamuraiRuntimeConfig:
    model_cfg: str = ""
    checkpoint: str = ""
    offload_video_to_cpu: bool = False
    offload_state_to_cpu: bool = False


def _call_with_supported_kwargs(fn, *args, **kwargs):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)

    params = sig.parameters
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kwargs:
        return fn(*args, **kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in params}
    return fn(*args, **filtered)


def _import_builder():
    candidates = (
        ("sam2.build_sam", "build_sam2_video_predictor"),
        ("samurai.sam2.build_sam", "build_sam2_video_predictor"),
        ("samurai.build_sam", "build_sam2_video_predictor"),
    )
    last_error: Exception | None = None
    for module_name, fn_name in candidates:
        try:
            mod = importlib.import_module(module_name)
            fn = getattr(mod, fn_name)
            return fn, module_name
        except Exception as exc:
            last_error = exc
            continue
    raise SamuraiUnavailable(
        "Samurai backend unavailable: could not import SAM2/Samurai predictor runtime. "
        "Install Samurai and ensure `sam2.build_sam` is importable."
    ) from last_error


def _build_predictor(runtime: SamuraiRuntimeConfig, device_hint: str):
    model_cfg = str(runtime.model_cfg or "").strip()
    checkpoint = str(runtime.checkpoint or "").strip()
    if not model_cfg or not checkpoint:
        raise SamuraiUnavailable(
            "Samurai backend requires both model config and checkpoint paths "
            "(samurai_model_cfg + samurai_checkpoint)."
        )

    builder, _module_name = _import_builder()
    attempts = [
        lambda: _call_with_supported_kwargs(builder, model_cfg, checkpoint, device=str(device_hint)),
        lambda: _call_with_supported_kwargs(
            builder,
            model_cfg=model_cfg,
            checkpoint=checkpoint,
            sam2_checkpoint=checkpoint,
            device=str(device_hint),
        ),
        lambda: builder(model_cfg, checkpoint),
    ]
    errors: list[str] = []
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:
            errors.append(str(exc))
            continue

    raise SamuraiUnavailable(
        "Failed to initialize Samurai video predictor. "
        "Check model cfg/checkpoint compatibility. "
        f"Attempts: {' | '.join(errors[:3])}"
    )


def _init_inference_state(predictor, frames_dir: Path, runtime: SamuraiRuntimeConfig):
    init_fn = getattr(predictor, "init_state", None)
    if init_fn is None:
        raise SamuraiUnavailable("Samurai predictor object has no `init_state` method.")

    kwargs = {
        "video_path": str(frames_dir),
        "offload_video_to_cpu": bool(runtime.offload_video_to_cpu),
        "offload_state_to_cpu": bool(runtime.offload_state_to_cpu),
    }
    attempts = [
        lambda: _call_with_supported_kwargs(init_fn, **kwargs),
        lambda: _call_with_supported_kwargs(
            init_fn,
            str(frames_dir),
            offload_video_to_cpu=bool(runtime.offload_video_to_cpu),
            offload_state_to_cpu=bool(runtime.offload_state_to_cpu),
        ),
        lambda: init_fn(str(frames_dir)),
    ]
    errors: list[str] = []
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:
            errors.append(str(exc))
            continue
    raise SamuraiUnavailable(
        "Failed to initialize Samurai inference state for video frames. "
        f"Attempts: {' | '.join(errors[:3])}"
    )


def _to_numpy(value) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            return value.detach().cpu().numpy()
        except Exception:
            pass
    return np.asarray(value)


def _normalize_rgb_u8(frame: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame)
    if rgb.ndim != 3:
        raise ValueError("Expected RGB-like frame for Samurai backend.")
    if rgb.shape[2] > 3:
        rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.asarray(np.clip(rgb, 0.0, 1.0) * 255.0, dtype=np.float32).round().astype(np.uint8)
    return rgb


def _clamp_point(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    return (
        float(max(0.0, min(width - 1.0, float(x)))),
        float(max(0.0, min(height - 1.0, float(y)))),
    )


def _normalize_box(
    box_xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box_xyxy
    xa = float(max(0, min(width - 1, int(round(min(x0, x1))))))
    xb = float(max(0, min(width - 1, int(round(max(x0, x1))))))
    ya = float(max(0, min(height - 1, int(round(min(y0, y1))))))
    yb = float(max(0, min(height - 1, int(round(max(y0, y1))))))
    if xb - xa < 2 or yb - ya < 2:
        raise ValueError("Selection box is too small for Samurai prompt initialization.")
    return xa, ya, xb, yb


def _write_frame_sequence(
    frame_loader: Callable[[int], np.ndarray],
    frame_start: int,
    frame_end: int,
    temp_dir: Path,
) -> tuple[int, int]:
    width: int | None = None
    height: int | None = None
    for seq_idx, local_idx in enumerate(range(int(frame_start), int(frame_end) + 1)):
        rgb_u8 = _normalize_rgb_u8(frame_loader(int(local_idx)))
        h, w = rgb_u8.shape[:2]
        if width is None or height is None:
            width, height = int(w), int(h)
        elif width != int(w) or height != int(h):
            raise ValueError(
                "All frames in Samurai range must share the same resolution. "
                f"Expected {width}x{height}, got {w}x{h} at frame {local_idx}."
            )
        bgr_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(
            str(temp_dir / f"{seq_idx:05d}.jpg"),
            bgr_u8,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )
        if not ok:
            raise RuntimeError(f"Failed to write temporary Samurai frame for local index {local_idx}.")
    if width is None or height is None:
        raise ValueError("No frames available for Samurai propagation.")
    return int(width), int(height)


def _resolve_obj_index(obj_ids, obj_id: int) -> int:
    if obj_ids is None:
        return 0
    ids = _to_numpy(obj_ids).reshape(-1)
    if ids.size == 0:
        return 0
    for i, candidate in enumerate(ids.tolist()):
        try:
            if int(candidate) == int(obj_id):
                return int(i)
        except Exception:
            continue
    return 0


def _mask_from_logits(mask_logits, obj_index: int, out_shape: tuple[int, int]) -> np.ndarray:
    logits = _to_numpy(mask_logits)
    if logits.size == 0:
        raise ValueError("Samurai returned empty mask logits.")

    arr = logits
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 3:
        # Common shape: [num_obj, H, W]
        if arr.shape[0] <= 32:
            arr = arr[max(0, min(int(obj_index), arr.shape[0] - 1))]
        else:
            arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported Samurai mask tensor shape: {tuple(logits.shape)}")

    arr = arr.astype(np.float32)
    if float(arr.min(initial=0.0)) >= 0.0 and float(arr.max(initial=1.0)) <= 1.0:
        alpha = (arr >= 0.5).astype(np.float32)
    else:
        alpha = (arr > 0.0).astype(np.float32)
    if alpha.shape[:2] != out_shape:
        alpha = cv2.resize(alpha, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def _parse_predictor_output(result, fallback_frame_idx: int) -> tuple[int, object, object]:
    frame_idx = int(fallback_frame_idx)
    obj_ids = None
    logits = None
    if isinstance(result, tuple):
        if len(result) >= 3:
            frame_idx = int(result[0])
            obj_ids = result[1]
            logits = result[2]
        elif len(result) == 2:
            obj_ids = result[0]
            logits = result[1]
        elif len(result) == 1:
            logits = result[0]
    else:
        logits = result
    return int(frame_idx), obj_ids, logits


def _start_propagation(predictor, inference_state):
    prop_fn = getattr(predictor, "propagate_in_video", None)
    if prop_fn is None:
        raise SamuraiUnavailable("Samurai predictor object has no `propagate_in_video` method.")
    try:
        return prop_fn(inference_state)
    except Exception:
        return _call_with_supported_kwargs(prop_fn, inference_state=inference_state)


def _seed_points_and_box(
    predictor,
    inference_state,
    anchor_seq_idx: int,
    box_xyxy: tuple[float, float, float, float],
    fg_points: Sequence[tuple[float, float]],
    bg_points: Sequence[tuple[float, float]],
    width: int,
    height: int,
    obj_id: int = 1,
) -> np.ndarray | None:
    seed_fn = getattr(predictor, "add_new_points_or_box", None)
    if seed_fn is None:
        raise SamuraiUnavailable("Samurai predictor object has no `add_new_points_or_box` method.")

    box = _normalize_box(box_xyxy, width=width, height=height)
    points: list[tuple[float, float]] = []
    labels: list[int] = []
    for px, py in fg_points:
        points.append(_clamp_point(px, py, width=width, height=height))
        labels.append(1)
    for px, py in bg_points:
        points.append(_clamp_point(px, py, width=width, height=height))
        labels.append(0)
    if not points:
        points = [((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)]
        labels = [1]

    point_coords = np.asarray(points, dtype=np.float32)
    point_labels = np.asarray(labels, dtype=np.int32)
    point_coords_batched = point_coords[None, ...]
    point_labels_batched = point_labels[None, ...]
    box_arr = np.asarray(box, dtype=np.float32)
    attempts = [
        lambda: _call_with_supported_kwargs(
            seed_fn,
            inference_state=inference_state,
            frame_idx=int(anchor_seq_idx),
            obj_id=int(obj_id),
            points=point_coords,
            labels=point_labels,
            box=box_arr,
        ),
        lambda: _call_with_supported_kwargs(
            seed_fn,
            inference_state=inference_state,
            frame_idx=int(anchor_seq_idx),
            obj_id=int(obj_id),
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_arr,
        ),
        lambda: _call_with_supported_kwargs(
            seed_fn,
            inference_state=inference_state,
            frame_idx=int(anchor_seq_idx),
            obj_id=int(obj_id),
            point_coords=point_coords_batched,
            point_labels=point_labels_batched,
            box=box_arr,
        ),
        lambda: seed_fn(inference_state, int(anchor_seq_idx), int(obj_id), point_coords, point_labels, box_arr),
    ]
    errors: list[str] = []
    for attempt in attempts:
        try:
            out = attempt()
            frame_idx, obj_ids, logits = _parse_predictor_output(out, fallback_frame_idx=int(anchor_seq_idx))
            if logits is None:
                return None
            obj_index = _resolve_obj_index(obj_ids, obj_id=obj_id)
            return _mask_from_logits(logits, obj_index=obj_index, out_shape=(height, width))
        except Exception as exc:
            errors.append(str(exc))
            continue
    raise SamuraiUnavailable(
        "Failed to seed Samurai predictor with points/box prompts. "
        f"Attempts: {' | '.join(errors[:3])}"
    )


def _prompts_from_anchor_mask(anchor_mask: np.ndarray) -> tuple[tuple[float, float, float, float], list[tuple[float, float]], list[tuple[float, float]]]:
    alpha = np.asarray(anchor_mask, dtype=np.float32)
    if alpha.ndim != 2:
        raise ValueError("Anchor mask for Samurai prompt derivation must be 2D.")
    binary = alpha >= 0.5
    ys, xs = np.where(binary)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("Anchor mask is empty; cannot derive Samurai prompts.")

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    box = (float(x0), float(y0), float(x1), float(y1))

    cx = int(round(float(xs.mean())))
    cy = int(round(float(ys.mean())))
    fg_points = [(float(cx), float(cy))]
    if xs.size >= 9:
        q_idx = np.linspace(0, xs.size - 1, num=4, dtype=int)
        fg_points.extend([(float(xs[i]), float(ys[i])) for i in q_idx.tolist()])

    h, w = alpha.shape[:2]
    margin = 8
    bg_points = [
        (float(max(0, x0 - margin)), float(max(0, y0 - margin))),
        (float(min(w - 1, x1 + margin)), float(max(0, y0 - margin))),
        (float(max(0, x0 - margin)), float(min(h - 1, y1 + margin))),
        (float(min(w - 1, x1 + margin)), float(min(h - 1, y1 + margin))),
    ]
    return box, fg_points, bg_points


def _seed_with_mask_or_fallback_prompts(
    predictor,
    inference_state,
    anchor_seq_idx: int,
    anchor_mask: np.ndarray,
    width: int,
    height: int,
    obj_id: int = 1,
) -> np.ndarray:
    alpha = np.asarray(anchor_mask, dtype=np.float32)
    if alpha.ndim != 2:
        raise ValueError("Anchor mask for Samurai backend must be a 2D alpha array.")
    if alpha.shape[:2] != (height, width):
        alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

    add_mask_fn = getattr(predictor, "add_new_mask", None)
    if add_mask_fn is not None:
        mask_input = (alpha >= 0.5).astype(np.uint8)
        attempts = [
            lambda: _call_with_supported_kwargs(
                add_mask_fn,
                inference_state=inference_state,
                frame_idx=int(anchor_seq_idx),
                obj_id=int(obj_id),
                mask=mask_input,
            ),
            lambda: _call_with_supported_kwargs(
                add_mask_fn,
                inference_state=inference_state,
                frame_idx=int(anchor_seq_idx),
                obj_id=int(obj_id),
                mask_input=mask_input,
            ),
            lambda: add_mask_fn(inference_state, int(anchor_seq_idx), int(obj_id), mask_input),
        ]
        for attempt in attempts:
            try:
                out = attempt()
                frame_idx, obj_ids, logits = _parse_predictor_output(out, fallback_frame_idx=int(anchor_seq_idx))
                if logits is None:
                    return alpha
                obj_index = _resolve_obj_index(obj_ids, obj_id=obj_id)
                return _mask_from_logits(logits, obj_index=obj_index, out_shape=(height, width))
            except Exception:
                continue

    box, fg_points, bg_points = _prompts_from_anchor_mask(alpha)
    seeded = _seed_points_and_box(
        predictor=predictor,
        inference_state=inference_state,
        anchor_seq_idx=int(anchor_seq_idx),
        box_xyxy=box,
        fg_points=fg_points,
        bg_points=bg_points,
        width=int(width),
        height=int(height),
        obj_id=int(obj_id),
    )
    if seeded is None:
        return alpha
    return seeded


def _collect_propagated_masks(
    predictor,
    inference_state,
    frame_start: int,
    frame_end: int,
    width: int,
    height: int,
    obj_id: int,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for result in _start_propagation(predictor, inference_state):
        frame_seq_idx, obj_ids, logits = _parse_predictor_output(result, fallback_frame_idx=0)
        if logits is None:
            continue
        local_idx = int(frame_start + int(frame_seq_idx))
        if local_idx < int(frame_start) or local_idx > int(frame_end):
            continue
        obj_index = _resolve_obj_index(obj_ids, obj_id=int(obj_id))
        alpha = _mask_from_logits(logits, obj_index=obj_index, out_shape=(height, width))
        out[int(local_idx)] = alpha
    return out


def _run_with_temp_video(
    frame_loader: Callable[[int], np.ndarray],
    frame_start: int,
    frame_end: int,
    fn: Callable[[Path, int, int], tuple[dict[int, np.ndarray], str | None]],
) -> tuple[dict[int, np.ndarray], str | None]:
    if int(frame_end) < int(frame_start):
        raise ValueError(f"Invalid Samurai frame range: {frame_start}..{frame_end}")
    with tempfile.TemporaryDirectory(prefix="vmhq_samurai_") as tmp:
        frames_dir = Path(tmp)
        width, height = _write_frame_sequence(
            frame_loader=frame_loader,
            frame_start=int(frame_start),
            frame_end=int(frame_end),
            temp_dir=frames_dir,
        )
        return fn(frames_dir, width, height)


def propagate_with_samurai_from_prompts(
    *,
    frame_loader: Callable[[int], np.ndarray],
    frame_start: int,
    frame_end: int,
    anchor_frame: int,
    box_xyxy: tuple[float, float, float, float],
    fg_points: Sequence[tuple[float, float]] = (),
    bg_points: Sequence[tuple[float, float]] = (),
    runtime: SamuraiRuntimeConfig,
    device_hint: str = "cuda",
    obj_id: int = 1,
) -> tuple[dict[int, np.ndarray], str | None]:
    if anchor_frame < frame_start or anchor_frame > frame_end:
        raise ValueError(f"Anchor frame {anchor_frame} is outside range {frame_start}..{frame_end}")

    def _run(frames_dir: Path, width: int, height: int) -> tuple[dict[int, np.ndarray], str | None]:
        predictor = _build_predictor(runtime=runtime, device_hint=device_hint)
        inference_state = _init_inference_state(predictor=predictor, frames_dir=frames_dir, runtime=runtime)
        anchor_seq_idx = int(anchor_frame - frame_start)
        anchor_alpha = _seed_points_and_box(
            predictor=predictor,
            inference_state=inference_state,
            anchor_seq_idx=anchor_seq_idx,
            box_xyxy=box_xyxy,
            fg_points=fg_points,
            bg_points=bg_points,
            width=width,
            height=height,
            obj_id=int(obj_id),
        )
        masks = _collect_propagated_masks(
            predictor=predictor,
            inference_state=inference_state,
            frame_start=int(frame_start),
            frame_end=int(frame_end),
            width=width,
            height=height,
            obj_id=int(obj_id),
        )
        if anchor_alpha is not None:
            masks[int(anchor_frame)] = np.clip(anchor_alpha, 0.0, 1.0).astype(np.float32)
        if not masks:
            raise SamuraiUnavailable("Samurai propagation returned no masks.")
        return masks, None

    return _run_with_temp_video(
        frame_loader=frame_loader,
        frame_start=int(frame_start),
        frame_end=int(frame_end),
        fn=_run,
    )


def propagate_with_samurai_from_mask(
    *,
    frame_loader: Callable[[int], np.ndarray],
    frame_start: int,
    frame_end: int,
    anchor_frame: int,
    anchor_mask: np.ndarray,
    runtime: SamuraiRuntimeConfig,
    device_hint: str = "cuda",
    obj_id: int = 1,
) -> tuple[dict[int, np.ndarray], str | None]:
    if anchor_frame < frame_start or anchor_frame > frame_end:
        raise ValueError(f"Anchor frame {anchor_frame} is outside range {frame_start}..{frame_end}")

    def _run(frames_dir: Path, width: int, height: int) -> tuple[dict[int, np.ndarray], str | None]:
        predictor = _build_predictor(runtime=runtime, device_hint=device_hint)
        inference_state = _init_inference_state(predictor=predictor, frames_dir=frames_dir, runtime=runtime)
        anchor_seq_idx = int(anchor_frame - frame_start)
        anchor_alpha = _seed_with_mask_or_fallback_prompts(
            predictor=predictor,
            inference_state=inference_state,
            anchor_seq_idx=anchor_seq_idx,
            anchor_mask=anchor_mask,
            width=width,
            height=height,
            obj_id=int(obj_id),
        )
        masks = _collect_propagated_masks(
            predictor=predictor,
            inference_state=inference_state,
            frame_start=int(frame_start),
            frame_end=int(frame_end),
            width=width,
            height=height,
            obj_id=int(obj_id),
        )
        masks[int(anchor_frame)] = np.clip(anchor_alpha, 0.0, 1.0).astype(np.float32)
        if not masks:
            raise SamuraiUnavailable("Samurai propagation returned no masks.")
        return masks, None

    return _run_with_temp_video(
        frame_loader=frame_loader,
        frame_start=int(frame_start),
        frame_end=int(frame_end),
        fn=_run,
    )


def is_samurai_backend(value: str | None) -> bool:
    return str(value or "").strip().lower() in SAMURAI_BACKEND_ALIASES


def canonicalize_samurai_backend(value: str | None) -> str:
    if is_samurai_backend(value):
        return SAMURAI_BACKEND_CANONICAL
    return str(value or "").strip().lower()
