"""Tests for v2 pipeline orchestrator routing and resolution logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from videomatte_hq.config import VideoMatteConfig
from videomatte_hq.pipeline.orchestrator import PipelineRunResult, run_pipeline


def test_run_pipeline_dispatches_v1_by_default() -> None:
    """Default pipeline_mode='v1' should call _run_pipeline_v1."""
    cfg = VideoMatteConfig(pipeline_mode="v1")
    with patch("videomatte_hq.pipeline.orchestrator._run_pipeline_v1") as mock_v1:
        mock_v1.return_value = PipelineRunResult(
            segment_result=MagicMock(),
            refine_result=MagicMock(),
            output_dir=MagicMock(),
        )
        run_pipeline(cfg)
        mock_v1.assert_called_once_with(cfg)


def test_run_pipeline_dispatches_v2() -> None:
    """pipeline_mode='v2' should call _run_pipeline_v2."""
    cfg = VideoMatteConfig(pipeline_mode="v2")
    with patch("videomatte_hq.pipeline.orchestrator._run_pipeline_v2") as mock_v2:
        mock_v2.return_value = PipelineRunResult(
            segment_result=None,
            refine_result=MagicMock(),
            output_dir=MagicMock(),
        )
        run_pipeline(cfg)
        mock_v2.assert_called_once_with(cfg)


def test_pipeline_run_result_segment_result_optional() -> None:
    """v2 pipeline sets segment_result=None."""
    result = PipelineRunResult(
        segment_result=None,
        refine_result=MagicMock(),
        output_dir=MagicMock(),
    )
    assert result.segment_result is None
