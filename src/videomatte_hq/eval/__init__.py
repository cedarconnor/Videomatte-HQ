"""Evaluation helpers for v1-v2 comparisons."""

from videomatte_hq.eval.harness import (
    compare_alpha_sequences,
    load_alpha_sequence,
    run_v1_v2_comparison,
    summarize_alpha_sequence,
)

__all__ = [
    "load_alpha_sequence",
    "summarize_alpha_sequence",
    "compare_alpha_sequences",
    "run_v1_v2_comparison",
]
