import { useCallback, useEffect } from "react";

interface Props {
  frame: number;
  frameStart: number;
  frameEnd: number;
  onFrameChange: (frame: number) => void;
  disabled?: boolean;
}

export function FrameTimeline({ frame, frameStart, frameEnd, onFrameChange, disabled = false }: Props) {
  const clamp = useCallback(
    (v: number) => Math.max(frameStart, Math.min(frameEnd, Math.round(v))),
    [frameStart, frameEnd],
  );

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (disabled) return;
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "ArrowRight") {
        e.preventDefault();
        onFrameChange(clamp(frame + (e.shiftKey ? 10 : 1)));
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        onFrameChange(clamp(frame - (e.shiftKey ? 10 : 1)));
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [frame, frameStart, frameEnd, disabled, onFrameChange, clamp]);

  return (
    <div className="frame-timeline">
      <button
        type="button"
        onClick={() => onFrameChange(clamp(frame - 1))}
        disabled={disabled || frame <= frameStart}
        title="Previous frame"
      >
        &lt;
      </button>
      <input
        type="range"
        min={frameStart}
        max={frameEnd}
        step={1}
        value={frame}
        onChange={(e) => onFrameChange(clamp(Number(e.target.value)))}
        disabled={disabled}
        className="frame-timeline-slider"
      />
      <button
        type="button"
        onClick={() => onFrameChange(clamp(frame + 1))}
        disabled={disabled || frame >= frameEnd}
        title="Next frame"
      >
        &gt;
      </button>
      <span className="frame-timeline-num">
        Frame <strong>{frame}</strong>
      </span>
    </div>
  );
}
