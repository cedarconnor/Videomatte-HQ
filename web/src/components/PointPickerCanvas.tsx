import { useCallback, useEffect, useRef, useState } from "react";

export interface PickerPoint {
  x: number; // normalized 0-1
  y: number; // normalized 0-1
  label: "positive" | "negative";
}

interface Props {
  frameSrc: string | null;
  points: PickerPoint[];
  onPointsChange: (points: PickerPoint[]) => void;
  overlayDataUrl?: string | null;
  overlayOpacity?: number;
  disabled?: boolean;
}

const HIT_RADIUS = 14; // px distance to "click on" existing dot
const DOT_RADIUS = 8;
const POSITIVE_COLOR = "#62d394";
const NEGATIVE_COLOR = "#ff6b6b";

export function PointPickerCanvas({
  frameSrc,
  points,
  onPointsChange,
  overlayDataUrl,
  overlayOpacity = 0.5,
  disabled = false,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameImgRef = useRef<HTMLImageElement | null>(null);
  const overlayImgRef = useRef<HTMLImageElement | null>(null);
  const [canvasSize, setCanvasSize] = useState({ w: 640, h: 360 });
  const [imgNatural, setImgNatural] = useState({ w: 640, h: 360 });

  // Load frame image
  useEffect(() => {
    if (!frameSrc) {
      frameImgRef.current = null;
      return;
    }
    const img = new Image();
    img.onload = () => {
      frameImgRef.current = img;
      setImgNatural({ w: img.naturalWidth, h: img.naturalHeight });
    };
    img.src = frameSrc;
  }, [frameSrc]);

  // Load overlay image
  useEffect(() => {
    if (!overlayDataUrl) {
      overlayImgRef.current = null;
      redraw();
      return;
    }
    const img = new Image();
    img.onload = () => {
      overlayImgRef.current = img;
      redraw();
    };
    img.src = overlayDataUrl;
  }, [overlayDataUrl]);

  // Resize canvas to fit container while maintaining aspect ratio
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const obs = new ResizeObserver(() => {
      const rect = container.getBoundingClientRect();
      const containerW = Math.floor(rect.width);
      const aspect = imgNatural.h / Math.max(1, imgNatural.w);
      const canvasW = containerW;
      const canvasH = Math.floor(containerW * aspect);
      setCanvasSize({ w: canvasW, h: canvasH });
    });
    obs.observe(container);
    return () => obs.disconnect();
  }, [imgNatural]);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw frame
    if (frameImgRef.current) {
      ctx.drawImage(frameImgRef.current, 0, 0, canvas.width, canvas.height);
    }

    // Draw overlay
    if (overlayImgRef.current && overlayDataUrl) {
      ctx.globalAlpha = overlayOpacity;
      ctx.drawImage(overlayImgRef.current, 0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1.0;
    }

    // Draw points
    for (const pt of points) {
      const px = pt.x * canvas.width;
      const py = pt.y * canvas.height;
      const color = pt.label === "positive" ? POSITIVE_COLOR : NEGATIVE_COLOR;

      // Outer ring (white border)
      ctx.beginPath();
      ctx.arc(px, py, DOT_RADIUS + 2, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255,0.9)";
      ctx.fill();

      // Colored dot
      ctx.beginPath();
      ctx.arc(px, py, DOT_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      // +/- label
      ctx.fillStyle = "#fff";
      ctx.font = "bold 11px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(pt.label === "positive" ? "+" : "-", px, py + 0.5);
    }
  }, [points, overlayDataUrl, overlayOpacity]);

  // Redraw when dependencies change
  useEffect(() => {
    redraw();
  }, [canvasSize, redraw]);

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (disabled || !canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    const normX = clickX / rect.width;
    const normY = clickY / rect.height;

    // Check if clicking on an existing point to remove it
    for (let i = 0; i < points.length; i++) {
      const px = points[i].x * rect.width;
      const py = points[i].y * rect.height;
      const dist = Math.sqrt((clickX - px) ** 2 + (clickY - py) ** 2);
      if (dist <= HIT_RADIUS) {
        const next = [...points];
        next.splice(i, 1);
        onPointsChange(next);
        return;
      }
    }

    // Left click = positive, right click handled in contextmenu
    const newPoint: PickerPoint = { x: normX, y: normY, label: "positive" };
    onPointsChange([...points, newPoint]);
  }

  function handleContextMenu(e: React.MouseEvent<HTMLCanvasElement>) {
    e.preventDefault();
    if (disabled || !canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    const normX = clickX / rect.width;
    const normY = clickY / rect.height;

    // Check remove first
    for (let i = 0; i < points.length; i++) {
      const px = points[i].x * rect.width;
      const py = points[i].y * rect.height;
      const dist = Math.sqrt((clickX - px) ** 2 + (clickY - py) ** 2);
      if (dist <= HIT_RADIUS) {
        const next = [...points];
        next.splice(i, 1);
        onPointsChange(next);
        return;
      }
    }

    const newPoint: PickerPoint = { x: normX, y: normY, label: "negative" };
    onPointsChange([...points, newPoint]);
  }

  return (
    <div className="point-picker-wrap" ref={containerRef}>
      <canvas
        ref={canvasRef}
        width={canvasSize.w}
        height={canvasSize.h}
        onClick={handleClick}
        onContextMenu={handleContextMenu}
        className="point-picker-canvas"
        style={{ cursor: disabled ? "not-allowed" : "crosshair" }}
      />
      <div className="point-picker-legend">
        <span><span className="dot-legend dot-positive" /> Left-click: foreground</span>
        <span><span className="dot-legend dot-negative" /> Right-click: background</span>
        <span className="muted">Click a dot to remove it</span>
      </div>
    </div>
  );
}
