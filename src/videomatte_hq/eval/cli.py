"""CLI for v1-v2 alpha sequence comparison."""

from __future__ import annotations

import argparse
import json
import sys

from videomatte_hq.eval.harness import run_v1_v2_comparison


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare v1 and v2 alpha outputs.")
    p.add_argument("--reference", required=True, help="Reference alpha pattern/path (typically v1 output).")
    p.add_argument("--candidate", required=True, help="Candidate alpha pattern/path (typically v2 output).")
    p.add_argument("--frame-start", type=int, default=None, help="Optional frame start.")
    p.add_argument("--frame-end", type=int, default=None, help="Optional frame end.")
    p.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold for temporal metrics.")
    p.add_argument("--output-json", default="", help="Optional path to write full comparison JSON.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_v1_v2_comparison(
        reference_pattern=args.reference,
        candidate_pattern=args.candidate,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        threshold=args.threshold,
        output_json=(args.output_json or None),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
