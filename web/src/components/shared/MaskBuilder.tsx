import type { MouseEvent, RefObject } from 'react'

interface BuilderPoint {
    x: number
    y: number
}

interface BuilderBox {
    x0: number
    y0: number
    x1: number
    y1: number
}

interface MaskBuilderProps {
    frameDataUrl: string | null
    frameSize: { width: number; height: number } | null
    maskPreviewUrl: string | null
    builderBox: BuilderBox | null
    activeBuilderBox: BuilderBox | null
    fgPoints: BuilderPoint[]
    bgPoints: BuilderPoint[]
    pointRadius: number
    builderTool: 'box' | 'fg' | 'bg'
    imageRef: RefObject<HTMLImageElement>
    onMouseDown: (e: MouseEvent<HTMLDivElement>) => void
    onMouseMove: (e: MouseEvent<HTMLDivElement>) => void
    onMouseUp: (e: MouseEvent<HTMLDivElement>) => void
    onMouseLeave: () => void
    onClick: (e: MouseEvent<HTMLDivElement>) => void
}

export default function MaskBuilder({
    frameDataUrl,
    frameSize,
    maskPreviewUrl,
    builderBox,
    activeBuilderBox,
    fgPoints,
    bgPoints,
    pointRadius,
    builderTool,
    imageRef,
    onMouseDown,
    onMouseMove,
    onMouseUp,
    onMouseLeave,
    onClick,
}: MaskBuilderProps) {
    if (!frameDataUrl || !frameSize) {
        return (
            <div className="text-xs text-gray-500">
                Click "Load Frame" to start building an initial mask from box and points.
            </div>
        )
    }

    return (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
            <div>
                <div className="text-xs text-gray-400 mb-1">
                    Step 1: Prompt auto-detect or draw a subject box. Step 2: Add FG/BG points if needed.
                </div>
                <div className="text-[11px] text-gray-500 mb-1">
                    Shortcuts: <span className="font-mono">F</span> foreground point, <span className="font-mono">B</span> background point, <span className="font-mono">Enter</span> build mask.
                </div>
                <div
                    className={`relative inline-block border border-gray-700 rounded overflow-hidden ${builderTool === 'box' ? 'cursor-crosshair' : 'cursor-cell'}`}
                    onMouseDown={onMouseDown}
                    onMouseMove={onMouseMove}
                    onMouseUp={onMouseUp}
                    onMouseLeave={onMouseLeave}
                    onClick={onClick}
                >
                    <img
                        ref={imageRef}
                        src={frameDataUrl}
                        alt="Assignment frame preview"
                        className="block max-h-[420px] w-auto select-none"
                        draggable={false}
                    />
                    <svg
                        className="absolute inset-0 w-full h-full pointer-events-none"
                        viewBox={`0 0 ${frameSize.width} ${frameSize.height}`}
                        preserveAspectRatio="xMinYMin meet"
                    >
                        {activeBuilderBox && (
                            <rect
                                x={Math.min(activeBuilderBox.x0, activeBuilderBox.x1)}
                                y={Math.min(activeBuilderBox.y0, activeBuilderBox.y1)}
                                width={Math.max(1, Math.abs(activeBuilderBox.x1 - activeBuilderBox.x0))}
                                height={Math.max(1, Math.abs(activeBuilderBox.y1 - activeBuilderBox.y0))}
                                fill="rgba(56, 189, 248, 0.18)"
                                stroke="rgb(56, 189, 248)"
                                strokeWidth={2}
                            />
                        )}
                        {fgPoints.map((p, idx) => (
                            <circle
                                key={`fg-${idx}`}
                                cx={p.x}
                                cy={p.y}
                                r={Math.max(2, pointRadius)}
                                fill="rgba(34, 197, 94, 0.55)"
                                stroke="rgb(34, 197, 94)"
                                strokeWidth={1.5}
                            />
                        ))}
                        {bgPoints.map((p, idx) => (
                            <circle
                                key={`bg-${idx}`}
                                cx={p.x}
                                cy={p.y}
                                r={Math.max(2, pointRadius)}
                                fill="rgba(239, 68, 68, 0.55)"
                                stroke="rgb(239, 68, 68)"
                                strokeWidth={1.5}
                            />
                        ))}
                    </svg>
                </div>
            </div>
            <div>
                <div className="text-xs text-gray-400 mb-1">
                    Prompt summary: box={builderBox ? "yes" : "no"}, FG points={fgPoints.length}, BG points={bgPoints.length}
                </div>
                {maskPreviewUrl ? (
                    <img
                        src={maskPreviewUrl}
                        alt="Built mask preview"
                        className="block max-h-[420px] w-auto border border-gray-700 rounded"
                    />
                ) : (
                    <div className="h-[220px] border border-dashed border-gray-700 rounded flex items-center justify-center text-xs text-gray-500 px-3 text-center">
                        Built mask preview appears here after you click "Build + Import Mask".
                    </div>
                )}
            </div>
        </div>
    )
}
