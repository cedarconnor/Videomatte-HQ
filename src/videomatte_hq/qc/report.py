"""QC report generation — end-of-run summary with metrics and thumbnails."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from videomatte_hq.config import VideoMatteConfig

logger = logging.getLogger(__name__)

REPORT_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <title>VideoMatte-HQ QC Report</title>
  <style>
    body { font-family: -apple-system, sans-serif; margin: 2rem; background: #1a1a2e; color: #e0e0e0; }
    h1 { color: #e94560; }
    h2 { color: #0f3460; background: #16213e; padding: 0.5rem; border-radius: 4px; }
    table { border-collapse: collapse; width: 100%%; margin: 1rem 0; }
    th, td { padding: 8px 12px; border: 1px solid #333; text-align: left; }
    th { background: #16213e; }
    .warning { color: #f5a623; font-weight: bold; }
    .ok { color: #4caf50; }
    .metric { font-family: monospace; }
    .section { margin: 2rem 0; }
  </style>
</head>
<body>
  <h1>VideoMatte-HQ — QC Report</h1>

  <div class="section">
    <h2>Pipeline Summary</h2>
    <table>
      <tr><th>Parameter</th><th>Value</th></tr>
      <tr><td>Total frames</td><td>%(num_frames)d</td></tr>
      <tr><td>Resolution</td><td>%(resolution)s</td></tr>
      <tr><td>Alpha format</td><td>%(alpha_format)s</td></tr>
      <tr><td>Shot type</td><td>%(shot_type)s</td></tr>
      <tr><td>Global model</td><td>%(global_model)s</td></tr>
      <tr><td>Refiner model</td><td>%(refiner_model)s</td></tr>
      <tr><td>Tile size</td><td>%(tile_size)s</td></tr>
      <tr><td>Temporal method</td><td>%(temporal_method)s</td></tr>
    </table>
  </div>

  <div class="section">
    <h2>Band Statistics</h2>
    <table>
      <tr><th>Metric</th><th>Mean</th><th>Max</th><th>Min</th></tr>
      <tr>
        <td>Tiles per frame</td>
        <td class="metric">%(avg_tiles).1f</td>
        <td class="metric">%(max_tiles)d</td>
        <td class="metric">%(min_tiles)d</td>
      </tr>
      <tr>
        <td>Band coverage</td>
        <td class="metric">%(avg_coverage).2f%%</td>
        <td class="metric">%(max_coverage).2f%%</td>
        <td class="metric">%(min_coverage).2f%%</td>
      </tr>
    </table>
  </div>

  %(extra_sections)s

</body>
</html>
"""


def generate_report(
    cfg: VideoMatteConfig,
    num_frames: int,
    per_frame_data: list[dict],
    output_dir: Path,
) -> None:
    """Generate end-of-run QC report.

    Args:
        cfg: Pipeline config.
        num_frames: Total frames processed.
        per_frame_data: Per-frame data with band/tile info.
        output_dir: Output directory.
    """
    report_dir = output_dir / "qc"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Compute band/tile stats
    tile_counts = [len(d.get("tiles", [])) for d in per_frame_data]
    band_coverages = []
    for d in per_frame_data:
        band = d.get("band")
        if band is not None:
            band_coverages.append(band.mean() * 100)
        else:
            band_coverages.append(0.0)

    avg_tiles = sum(tile_counts) / max(len(tile_counts), 1)
    max_tiles = max(tile_counts) if tile_counts else 0
    min_tiles = min(tile_counts) if tile_counts else 0

    avg_coverage = sum(band_coverages) / max(len(band_coverages), 1)
    max_coverage = max(band_coverages) if band_coverages else 0
    min_coverage = min(band_coverages) if band_coverages else 0

    # Render HTML
    html = REPORT_TEMPLATE % {
        "num_frames": num_frames,
        "resolution": "N/A",
        "alpha_format": cfg.io.alpha_format.value,
        "shot_type": cfg.io.shot_type.value,
        "global_model": cfg.globals.model,
        "refiner_model": cfg.refine.model,
        "tile_size": str(cfg.tiles.tile_size),
        "temporal_method": cfg.temporal.method.value,
        "avg_tiles": avg_tiles,
        "max_tiles": max_tiles,
        "min_tiles": min_tiles,
        "avg_coverage": avg_coverage,
        "max_coverage": max_coverage,
        "min_coverage": min_coverage,
        "extra_sections": "",
    }

    report_path = report_dir / "report.html"
    report_path.write_text(html)
    logger.info(f"QC report written to {report_path}")

    # Write metrics JSON
    metrics = {
        "num_frames": num_frames,
        "tile_counts": tile_counts,
        "band_coverages": band_coverages,
        "config_hashes": {
            stage: cfg.stage_hash(stage)
            for stage in ["background", "roi", "global", "intermediate", "band", "refine", "temporal"]
        },
    }
    metrics_path = report_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
