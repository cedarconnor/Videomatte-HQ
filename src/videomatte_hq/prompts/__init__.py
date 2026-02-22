"""Prompt adapters for segmentation backends."""

from videomatte_hq.prompts.box_adapter import BoxPromptAdapter
from videomatte_hq.prompts.mask_adapter import MaskPromptAdapter

__all__ = ["MaskPromptAdapter", "BoxPromptAdapter"]
