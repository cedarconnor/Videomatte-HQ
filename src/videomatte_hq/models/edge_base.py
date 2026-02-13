"""Protocol for edge refinement models — Pass B."""

from __future__ import annotations

from typing import Optional, Protocol

from torch import Tensor


class EdgeRefiner(Protocol):
    """Interface for trimap-guided edge refinement models.

    Used in Pass B for recovering high-frequency edge detail at 8K.
    """

    def infer_tile(
        self,
        rgb_tile: Tensor,
        trimap_tile: Tensor,
        alpha_prior: Tensor,
        bg_tile: Optional[Tensor] = None,
    ) -> Tensor:
        """Refine alpha on a single tile.

        Args:
            rgb_tile: (C, H, W) RGB tile at native resolution.
            trimap_tile: (1, H, W) values in {0.0, 0.5, 1.0}.
            alpha_prior: (1, H, W) guidance alpha from A0prime.
            bg_tile: (C, H, W) optional BG plate tile (high-confidence only).

        Returns:
            (1, H, W) refined alpha tile.
        """
        ...

    def load_weights(self, device: str = "cuda") -> None:
        """Load model weights."""
        ...
