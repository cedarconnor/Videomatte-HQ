import { useMemo, useState } from "react";

interface Props {
  leftSrc: string;
  rightSrc: string;
  leftLabel?: string;
  rightLabel?: string;
  zoom?: number;
}

export function WipeCompare({ leftSrc, rightSrc, leftLabel = "Input", rightLabel = "Alpha", zoom = 1 }: Props) {
  const [split, setSplit] = useState(55);
  const clip = useMemo(() => ({ clipPath: `inset(0 ${100 - split}% 0 0)` }), [split]);

  const shellStyle = zoom > 1
    ? { overflow: "auto" as const, maxHeight: "70vh" }
    : {};
  const stageStyle = zoom > 1
    ? { width: `${zoom * 100}%`, aspectRatio: "16 / 9" }
    : {};

  return (
    <div className="wipe-shell" style={shellStyle}>
      <div
        className="wipe-stage"
        style={stageStyle}
        onMouseMove={(e) => {
          if ((e.buttons & 1) !== 1) return;
          const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
          const pct = ((e.clientX - rect.left) / rect.width) * 100;
          setSplit(Math.max(0, Math.min(100, pct)));
        }}
      >
        <img src={rightSrc} alt={rightLabel} className="wipe-img" />
        <img src={leftSrc} alt={leftLabel} className="wipe-img wipe-img-left" style={clip} />
        <div className="wipe-divider" style={{ left: `${split}%` }}>
          <span>||</span>
        </div>
        <div className="wipe-label wipe-label-left">{leftLabel}</div>
        <div className="wipe-label wipe-label-right">{rightLabel}</div>
      </div>
      <input
        type="range"
        min={0}
        max={100}
        value={split}
        onChange={(e) => setSplit(Number(e.target.value))}
        className="wipe-slider"
      />
    </div>
  );
}
