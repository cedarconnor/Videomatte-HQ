from __future__ import annotations

from concurrent.futures import Future

import pytest

from videomatte_hq.io.writer import AlphaWriter


def _future_with_exception(exc: BaseException) -> Future:
    f: Future = Future()
    f.set_exception(exc)
    return f


def test_alpha_writer_flush_collects_multiple_exceptions() -> None:
    writer = AlphaWriter(output_pattern="alpha/%06d.png", workers=1)
    try:
        writer._futures = [
            _future_with_exception(RuntimeError("first failure")),
            _future_with_exception(ValueError("second failure")),
        ]
        with pytest.raises(RuntimeError, match="Multiple alpha write failures"):
            writer.flush()
        assert writer._futures == []
    finally:
        writer._executor.shutdown(wait=False)
