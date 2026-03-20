"""VRAM management utilities for sequential model stages."""

from __future__ import annotations

import gc
import logging

logger = logging.getLogger(__name__)


def unload_model(model: object, label: str = "model") -> None:
    """Delete a model and free GPU memory.

    Used between sequential pipeline stages (e.g. SAM3 → MatAnyone2 → MEMatte)
    to reclaim VRAM before the next model loads.
    """
    if model is None:
        return

    try:
        import torch

        if hasattr(model, "cpu"):
            model.cpu()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            logger.info("Unloaded %s — VRAM after cleanup: %.2f GB allocated.", label, allocated)
        else:
            logger.info("Unloaded %s (no CUDA device).", label)
    except ImportError:
        del model
        gc.collect()
        logger.info("Unloaded %s (torch not available).", label)
