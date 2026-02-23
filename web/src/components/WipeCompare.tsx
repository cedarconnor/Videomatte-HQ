import { useMemo, useState } from "react";

interface Props {
  leftSrc: string;
  rightSrc: string;
  leftLabel?: string;
  rightLabel?: string;
}

export function WipeCompare({ leftSrc, rightSrc, leftLabel = "Input", rightLabel = "Alpha" }: Props) {
  const [split, setSplit] = useState(55);
  const clip = useMemo(() => ({ clipPath: `inset(0 ${100 - split}% 0 0)` }), [split]);

  return (
    <div className="wipe-shell">
      <div
        className="wipe-stage"
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
