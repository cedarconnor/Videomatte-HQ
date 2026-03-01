export function HelpPage() {
  return (
    <div className="help-page">
      <div className="panel-head">
        <h2>Help &amp; Guide</h2>
        <p>Workflow overview, subject selection, and MEMatte tuning reference.</p>
      </div>

      <details className="stacked-details" open>
        <summary>Quick Start Workflow</summary>
        <div>
          <ol className="help-steps">
            <li><strong>Load video</strong> — paste or browse to an input video file (.mp4, .mov, .mkv).</li>
            <li><strong>Set frame range</strong> — start with a short range (e.g. 0–29) for testing. <code>frame_end</code> is the last frame index (inclusive), not the count.</li>
            <li><strong>Choose subject selection method</strong> — use the <em>Auto-Anchor</em> or <em>Point Picker</em> tab.</li>
            <li><strong>Run preflight</strong> — validates paths, MEMatte assets, and frame range before submitting.</li>
            <li><strong>Submit job</strong> — queues segmentation + refinement. Monitor progress on the Jobs tab.</li>
            <li><strong>Check QC</strong> — inspect alpha output, trimap overlays, and wipe comparisons on the QC tab.</li>
          </ol>
        </div>
      </details>

      <details className="stacked-details">
        <summary>Subject Selection</summary>
        <div>
          <h4>Auto-Anchor</h4>
          <p>
            YOLO person detection generates a mask on the first non-black frame, then SAM3 uses it as a prompt to segment the subject across all frames.
            Best for simple single-person shots with a clear subject.
          </p>

          <h4>Point Picker</h4>
          <p>
            Click directly on the video frame to tell SAM3 what to segment:
          </p>
          <ul>
            <li><strong>Left-click</strong> = foreground point (green) — marks the subject.</li>
            <li><strong>Right-click</strong> = background point (red) — marks areas to exclude.</li>
          </ul>

          <div className="tip-box">
            <strong>Tip:</strong> Place 2–5 positive points on the subject and 1–2 negative points on confusing background areas.
            Points apply to frame 0 for initial segmentation — SAM3 automatically propagates the selection across all frames.
          </div>
        </div>
      </details>

      <details className="stacked-details">
        <summary>Trimap &amp; MEMatte Tuning</summary>
        <div>
          <p>
            The trimap defines three zones that guide MEMatte refinement: <strong>definite foreground</strong>,
            <strong>definite background</strong>, and an <strong>unknown band</strong> where MEMatte computes soft alpha values.
          </p>

          <pre className="help-diagram">{`
  SAM binary mask edge
  ◄── erosion (inward) ──┤── unknown band ──┤── dilation (outward) ──►
                         │                  │
  ┌──────────────────────┤                  ├──────────────────────────┐
  │  Definite Foreground │  MEMatte refines │  Definite Background    │
  │  (alpha = 1.0)       │  alpha here      │  (alpha = 0.0)          │
  └──────────────────────┤                  ├──────────────────────────┘
                         │                  │
                    FG boundary        BG boundary
`}</pre>

          <table className="settings-table">
            <thead>
              <tr>
                <th>Setting</th>
                <th>Default</th>
                <th>What it does</th>
                <th>Hair detail tip</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Trimap Erosion (px)</td>
                <td>20</td>
                <td>Shrinks "definite FG" inward from the SAM edge. Larger = wider unknown band inside.</td>
                <td><strong>Decrease to 10–12</strong> to keep more subject as definite FG, focusing MEMatte on the true edge.</td>
              </tr>
              <tr>
                <td>Trimap Dilation (px)</td>
                <td>10</td>
                <td>Expands "unknown" region outward beyond the SAM edge.</td>
                <td><strong>Increase to 20–25</strong> to capture wispy hair strands SAM missed.</td>
              </tr>
              <tr>
                <td>Tile Size (px)</td>
                <td>1536</td>
                <td>MEMatte processes tiles of this size. Larger = more spatial context per tile.</td>
                <td><strong>Increase to 2048</strong> if VRAM allows (~10–12 GB).</td>
              </tr>
              <tr>
                <td>Tile Overlap (px)</td>
                <td>96</td>
                <td>Overlap between adjacent tiles, blended with Hann windows to eliminate seams.</td>
                <td><strong>Increase to 128–192</strong> to reduce seam artifacts near hair.</td>
              </tr>
              <tr>
                <td>Trimap Mode</td>
                <td>morphological</td>
                <td><code>morphological</code> = erosion/dilation bands. <code>logit</code> = SAM confidence thresholds.</td>
                <td>Keep <code>morphological</code> for predictable results.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </details>

      <details className="stacked-details">
        <summary>Common Issues</summary>
        <div>
          <dl className="help-issues">
            <dt>"Hair looks hard-edged"</dt>
            <dd>
              Increase <strong>Trimap Dilation</strong> (e.g. 20–25px) to widen the unknown band outward, giving MEMatte
              more room to find soft edges. Optionally decrease <strong>Trimap Erosion</strong> to 10–12px.
            </dd>

            <dt>"MEMatte tile failure" or out-of-memory</dt>
            <dd>
              Reduce <strong>Tile Size</strong> (e.g. 1024px). Each tile is processed independently — smaller tiles use less VRAM
              but may produce more visible seams. Increase <strong>Tile Overlap</strong> to compensate.
            </dd>

            <dt>"Frame out of range" error</dt>
            <dd>
              <code>frame_end</code> is the last frame <em>index</em> (inclusive), not the frame count.
              A 30-frame video has indices 0–29, so set <code>frame_end = 29</code>.
              The UI auto-clamps this when video info loads.
            </dd>

            <dt>SAM-only preview (no soft edges)</dt>
            <dd>
              If <strong>MEMatte Refine</strong> is unchecked, the pipeline outputs hard binary masks from SAM only.
              Enable it and ensure MEMatte repo/checkpoint paths are set correctly.
            </dd>
          </dl>
        </div>
      </details>
    </div>
  );
}
