"""Detectron2-free MEMatte wrapper for tile/frame alpha refinement.

This wrapper instantiates MEMatte directly via PyTorch and loads checkpoint
weights without detectron2's LazyConfig/DetectionCheckpointer path.
"""

from __future__ import annotations

import logging
import sys
import types
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


def _install_detectron2_shims() -> None:
    """Install minimal detectron2 Python shims if detectron2 is unavailable."""

    try:
        import detectron2.layers  # noqa: F401
        return
    except Exception:
        pass

    if "detectron2.layers" in sys.modules:
        return

    d2_mod = types.ModuleType("detectron2")
    layers_mod = types.ModuleType("detectron2.layers")
    structures_mod = types.ModuleType("detectron2.structures")
    modeling_mod = types.ModuleType("detectron2.modeling")
    modeling_backbone_mod = types.ModuleType("detectron2.modeling.backbone")
    fpn_mod = types.ModuleType("detectron2.modeling.backbone.fpn")

    class CNNBlockBase(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.stride = stride

    class LayerNorm2d(nn.Module):
        def __init__(self, normalized_shape: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
            self.eps = eps

        def forward(self, x: Tensor) -> Tensor:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            y = (x - u) / torch.sqrt(s + self.eps)
            return self.weight[:, None, None] * y + self.bias[:, None, None]

    class ShapeSpec:
        def __init__(self, *, channels: int | None = None, stride: int | None = None):
            self.channels = channels
            self.stride = stride

    class ImageList:
        def __init__(self, tensor: Tensor, image_sizes: list[tuple[int, int]]):
            self.tensor = tensor
            self.image_sizes = image_sizes

    def get_norm(norm: str | None, out_channels: int):
        if norm is None or norm == "":
            return None
        if norm == "BN":
            return nn.BatchNorm2d(out_channels)
        if norm == "LN":
            return LayerNorm2d(out_channels)
        raise KeyError(f"Unsupported norm in MEMatte shim: {norm}")

    def _assert_strides_are_log2_contiguous(_strides: list[int] | tuple[int, ...]) -> None:
        return None

    layers_mod.CNNBlockBase = CNNBlockBase
    layers_mod.Conv2d = nn.Conv2d
    layers_mod.get_norm = get_norm
    layers_mod.ShapeSpec = ShapeSpec
    structures_mod.ImageList = ImageList
    fpn_mod._assert_strides_are_log2_contiguous = _assert_strides_are_log2_contiguous

    sys.modules.setdefault("detectron2", d2_mod)
    sys.modules["detectron2.layers"] = layers_mod
    sys.modules["detectron2.structures"] = structures_mod
    sys.modules["detectron2.modeling"] = modeling_mod
    sys.modules["detectron2.modeling.backbone"] = modeling_backbone_mod
    sys.modules["detectron2.modeling.backbone.fpn"] = fpn_mod

    setattr(d2_mod, "layers", layers_mod)
    setattr(d2_mod, "structures", structures_mod)
    setattr(d2_mod, "modeling", modeling_mod)
    setattr(modeling_mod, "backbone", modeling_backbone_mod)
    setattr(modeling_backbone_mod, "fpn", fpn_mod)


def _install_timm_shims() -> None:
    """Install minimal timm.models.layers shims when timm is unavailable."""

    try:
        import timm.models.layers  # noqa: F401
        return
    except Exception:
        pass

    if "timm.models.layers" in sys.modules:
        return

    timm_mod = types.ModuleType("timm")
    timm_models_mod = types.ModuleType("timm.models")
    timm_layers_mod = types.ModuleType("timm.models.layers")

    def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
        if drop_prob <= 0.0 or not training:
            return x
        keep_prob = 1.0 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

    class DropPath(nn.Module):
        def __init__(self, drop_prob: float = 0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x: Tensor) -> Tensor:
            return drop_path(x, self.drop_prob, self.training)

    class Mlp(nn.Module):
        def __init__(
            self,
            in_features: int,
            hidden_features: int | None = None,
            out_features: int | None = None,
            act_layer=nn.GELU,
            drop: float = 0.0,
        ):
            super().__init__()
            hidden = int(hidden_features or in_features)
            out = int(out_features or in_features)
            self.fc1 = nn.Linear(in_features, hidden)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden, out)
            self.drop = nn.Dropout(drop)

        def forward(self, x: Tensor) -> Tensor:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x

    def trunc_normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0):
        return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

    timm_layers_mod.DropPath = DropPath
    timm_layers_mod.Mlp = Mlp
    timm_layers_mod.trunc_normal_ = trunc_normal_

    sys.modules.setdefault("timm", timm_mod)
    sys.modules["timm.models"] = timm_models_mod
    sys.modules["timm.models.layers"] = timm_layers_mod
    setattr(timm_mod, "models", timm_models_mod)
    setattr(timm_models_mod, "layers", timm_layers_mod)


def _install_fairscale_shims() -> None:
    """Install a minimal fairscale checkpoint shim when fairscale is unavailable."""

    try:
        import fairscale.nn.checkpoint  # noqa: F401
        return
    except Exception:
        pass

    if "fairscale.nn.checkpoint" in sys.modules:
        return

    fairscale_mod = types.ModuleType("fairscale")
    fairscale_nn_mod = types.ModuleType("fairscale.nn")
    fairscale_checkpoint_mod = types.ModuleType("fairscale.nn.checkpoint")

    def checkpoint_wrapper(module: nn.Module) -> nn.Module:
        return module

    fairscale_checkpoint_mod.checkpoint_wrapper = checkpoint_wrapper
    sys.modules.setdefault("fairscale", fairscale_mod)
    sys.modules["fairscale.nn"] = fairscale_nn_mod
    sys.modules["fairscale.nn.checkpoint"] = fairscale_checkpoint_mod
    setattr(fairscale_mod, "nn", fairscale_nn_mod)
    setattr(fairscale_nn_mod, "checkpoint", fairscale_checkpoint_mod)


def _install_einops_shims() -> None:
    """Install a tiny einops shim when einops is unavailable."""

    try:
        import einops  # noqa: F401
        return
    except Exception:
        pass

    if "einops" in sys.modules:
        return

    einops_mod = types.ModuleType("einops")

    def rearrange(tensor: Tensor, pattern: str, **kwargs) -> Tensor:
        raise RuntimeError(
            f"einops.rearrange is required for this path (pattern={pattern!r}). "
            "Install with: pip install einops"
        )

    def repeat(tensor: Tensor, pattern: str, **kwargs) -> Tensor:
        raise RuntimeError(
            f"einops.repeat is required for this path (pattern={pattern!r}). "
            "Install with: pip install einops"
        )

    def pack(tensors, pattern: str):
        if len(tensors) != 1:
            raise RuntimeError(
                f"einops.pack shim only supports one tensor (pattern={pattern!r}); "
                "install einops for full support."
            )
        return tensors[0], None

    def unpack(tensor: Tensor, ps, pattern: str):
        return [tensor]

    einops_mod.rearrange = rearrange
    einops_mod.repeat = repeat
    einops_mod.pack = pack
    einops_mod.unpack = unpack
    sys.modules["einops"] = einops_mod


def _install_fvcore_shims() -> None:
    """Install a minimal fvcore.nn.weight_init shim when fvcore is unavailable."""

    try:
        import fvcore.nn.weight_init  # noqa: F401
        return
    except Exception:
        pass

    if "fvcore.nn.weight_init" in sys.modules:
        return

    fvcore_mod = types.ModuleType("fvcore")
    fvcore_nn_mod = types.ModuleType("fvcore.nn")
    fvcore_weight_init_mod = types.ModuleType("fvcore.nn.weight_init")

    def c2_msra_fill(module: nn.Module) -> None:
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, 0.0)

    fvcore_weight_init_mod.c2_msra_fill = c2_msra_fill
    sys.modules.setdefault("fvcore", fvcore_mod)
    sys.modules["fvcore.nn"] = fvcore_nn_mod
    sys.modules["fvcore.nn.weight_init"] = fvcore_weight_init_mod
    setattr(fvcore_mod, "nn", fvcore_nn_mod)
    setattr(fvcore_nn_mod, "weight_init", fvcore_weight_init_mod)


@contextmanager
def _prepend_sys_path(path: Path) -> Iterator[None]:
    p = str(path)
    already = p in sys.path
    if not already:
        sys.path.insert(0, p)
    try:
        yield
    finally:
        if not already:
            try:
                sys.path.remove(p)
            except ValueError:
                pass


def _strip_module_prefix(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


class MEMatteModel:
    """MEMatte wrapper implementing the EdgeRefiner protocol."""

    def __init__(
        self,
        repo_dir: str = "third_party/MEMatte",
        checkpoint_path: str = "third_party/MEMatte/checkpoints/MEMatte_ViTS_DIM.pth",
        device: str = "cuda",
        precision: str = "fp16",
        max_number_token: int = 18500,
        patch_decoder: bool = True,
    ):
        self.repo_dir = str(repo_dir)
        self.checkpoint_path = str(checkpoint_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.precision = str(precision).lower()
        self.max_number_token = int(max_number_token)
        self.patch_decoder = bool(patch_decoder)
        self.model: Optional[nn.Module] = None

    def load_weights(self, device: str = "cuda") -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        repo_dir = Path(self.repo_dir).expanduser().resolve()
        checkpoint_path = Path(self.checkpoint_path).expanduser().resolve()

        if not repo_dir.exists():
            raise RuntimeError(f"MEMatte repo directory not found: {repo_dir}")
        if not checkpoint_path.exists():
            raise RuntimeError(f"MEMatte checkpoint not found: {checkpoint_path}")

        _install_detectron2_shims()
        _install_timm_shims()
        _install_fairscale_shims()
        _install_einops_shims()
        _install_fvcore_shims()

        with _prepend_sys_path(repo_dir):
            from modeling import Detail_Capture, MEMatte, ViT

            backbone = ViT(
                in_chans=4,
                img_size=512,
                patch_size=16,
                embed_dim=384,
                depth=12,
                num_heads=6,
                drop_path_rate=0,
                window_size=14,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
                residual_block_indexes=[2, 5, 8, 11],
                use_rel_pos=True,
                out_feature="last_feat",
                topk=0.25,
                multi_score=True,
                max_number_token=max(1, int(self.max_number_token)),
            )

            model = MEMatte(
                teacher_backbone=None,
                backbone=backbone,
                criterion=None,
                pixel_mean=[123.675 / 255.0, 116.280 / 255.0, 103.530 / 255.0],
                pixel_std=[58.395 / 255.0, 57.120 / 255.0, 57.375 / 255.0],
                input_format="RGB",
                size_divisibility=32,
                decoder=Detail_Capture(),
                distill=False,
                distill_loss_ratio=0.0,
                token_loss_ratio=0.0,
            )

        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        if isinstance(checkpoint, dict):
            if "model" in checkpoint and isinstance(checkpoint["model"], dict):
                checkpoint = checkpoint["model"]
            elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"Unsupported MEMatte checkpoint format at {checkpoint_path}")

        state_dict = _strip_module_prefix(checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.info("MEMatte load: missing %d keys (expected for teacher/training-only parts).", len(missing))
        if unexpected:
            logger.info("MEMatte load: unexpected %d keys ignored.", len(unexpected))

        model.to(self.device)
        model.eval()
        # Keep MEMatte in FP32 even when global precision is FP16.
        # Upstream MEMatte allocates some internal tensors in float32 during
        # patch inference; forcing model half can trigger dtype mismatch errors.
        if self.precision == "fp16" and self.device.type == "cuda":
            logger.warning(
                "MEMatte requested fp16, but running MEMatte in fp32 for compatibility."
            )

        self.model = model
        logger.info(
            "MEMatte loaded from %s on %s (max_tokens=%d patch_decoder=%s)",
            checkpoint_path,
            self.device,
            self.max_number_token,
            self.patch_decoder,
        )

    @torch.no_grad()
    def infer_tile(
        self,
        rgb_tile: Tensor,
        trimap_tile: Tensor,
        alpha_prior: Tensor,
        bg_tile: Optional[Tensor] = None,
    ) -> Tensor:
        if self.model is None:
            raise RuntimeError("MEMatte model not loaded. Call load_weights() first.")

        if rgb_tile.ndim != 3 or rgb_tile.shape[0] != 3:
            raise RuntimeError(f"MEMatte rgb_tile must be (3,H,W), got {tuple(rgb_tile.shape)}")
        if trimap_tile.ndim != 3 or trimap_tile.shape[0] != 1:
            raise RuntimeError(f"MEMatte trimap_tile must be (1,H,W), got {tuple(trimap_tile.shape)}")
        if rgb_tile.shape[-2:] != trimap_tile.shape[-2:]:
            raise RuntimeError(
                f"MEMatte input mismatch: rgb={tuple(rgb_tile.shape)} trimap={tuple(trimap_tile.shape)}"
            )

        image = rgb_tile.unsqueeze(0).to(self.device, non_blocking=True).float()
        trimap = trimap_tile.unsqueeze(0).to(self.device, non_blocking=True).float().clamp(0.0, 1.0)
        # MEMatte runs in fp32 for compatibility with upstream patch-inference path.

        outputs, _, _ = self.model(
            {"image": image, "trimap": trimap},
            patch_decoder=bool(self.patch_decoder),
        )
        alpha = outputs["phas"][0].float().clamp(0.0, 1.0)
        alpha = alpha.cpu()

        tri_cpu = trimap_tile.float().cpu()
        alpha[tri_cpu <= 0.0] = 0.0
        alpha[tri_cpu >= 1.0] = 1.0
        return alpha

    @torch.no_grad()
    def infer_tile_batch(
        self,
        rgb_tiles: list[Tensor],
        trimap_tiles: list[Tensor],
        alpha_priors: list[Tensor],
        bg_tiles: list[Optional[Tensor]] | None = None,
    ) -> list[Tensor]:
        if not rgb_tiles:
            return []
        if self.model is None:
            raise RuntimeError("MEMatte model not loaded. Call load_weights() first.")

        h = rgb_tiles[0].shape[-2]
        w = rgb_tiles[0].shape[-1]
        same_size = all(tile.shape[-2:] == (h, w) for tile in rgb_tiles) and all(
            tile.shape[-2:] == (h, w) for tile in trimap_tiles
        )
        if not same_size:
            return [
                self.infer_tile(rgb, tri, ap, bg)
                for rgb, tri, ap, bg in zip(
                    rgb_tiles,
                    trimap_tiles,
                    alpha_priors,
                    bg_tiles or [None] * len(rgb_tiles),
                )
            ]

        image = torch.stack([t.float() for t in rgb_tiles], dim=0).to(self.device, non_blocking=True)
        trimap = torch.stack([t.float().clamp(0.0, 1.0) for t in trimap_tiles], dim=0).to(
            self.device,
            non_blocking=True,
        )
        # MEMatte runs in fp32 for compatibility with upstream patch-inference path.

        outputs, _, _ = self.model(
            {"image": image, "trimap": trimap},
            patch_decoder=bool(self.patch_decoder),
        )
        alphas = outputs["phas"].float().clamp(0.0, 1.0).cpu()
        trimap_cpu = torch.stack([t.float() for t in trimap_tiles], dim=0).cpu()
        alphas[trimap_cpu <= 0.0] = 0.0
        alphas[trimap_cpu >= 1.0] = 1.0
        return [alphas[i] for i in range(alphas.shape[0])]

    def infer_frame(self, rgb: Tensor, trimap: Tensor) -> Tensor:
        return self.infer_tile(rgb, trimap, alpha_prior=trimap)
