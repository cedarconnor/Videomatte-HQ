"""Protocol for global (temporal) matting models — Pass A backbone."""

from __future__ import annotations

from typing import Any, Protocol, Tuple

from torch import Tensor


class GlobalMatteModel(Protocol):
    """Interface for temporal video matting models used in Pass A.

    Implementations must accept a chunk of frames and return alpha + state.
    The recurrent state enables temporal consistency across chunks.
    """

    def infer_chunk(
        self,
        frames: Tensor,
        recurrent_state: Any = None,
    ) -> Tuple[Tensor, Any]:
        """Run inference on a temporal chunk.

        Args:
            frames: (T, C, H, W) RGB float32 in [0, 1].
            recurrent_state: Optional state from previous chunk for continuity.

        Returns:
            (alpha, state):
                alpha: (T, 1, H, W) float32 alpha in [0, 1].
                state: Updated recurrent state for next chunk.
        """
        ...

    def load_weights(self, device: str = "cuda") -> None:
        """Load model weights."""
        ...
