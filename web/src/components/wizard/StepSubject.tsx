import type { MouseEvent, RefObject } from 'react'
import { Input } from '../ui/Input'
import { Select } from '../ui/Select'
import MaskBuilder from '../shared/MaskBuilder'

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

interface KeyframeItem {
    frame: number
    kind: 'initial' | 'correction'
    mask_asset: string
}

interface StepSubjectProps {
    assignmentFrame: number
    assignmentKind: 'initial' | 'correction'
    builderPrompt: string
    assignmentBusy: boolean
    builderLoadingFrame: boolean
    builderSuggestingBoxes: boolean
    builderBuildingMask: boolean
    builderBuildingRange: boolean
    hasAssignment: boolean
    keyframeCount: number
    keyframes: KeyframeItem[]
    builderFrameDataUrl: string | null
    builderFrameSize: { width: number; height: number } | null
    builderMaskPreviewUrl: string | null
    builderBox: BuilderBox | null
    activeBuilderBox: BuilderBox | null
    builderFgPoints: BuilderPoint[]
    builderBgPoints: BuilderPoint[]
    builderPointRadius: number
    builderTool: 'box' | 'fg' | 'bg'
    builderImgRef: RefObject<HTMLImageElement>
    onBuilderToolChange: (tool: 'box' | 'fg' | 'bg') => void
    onAssignmentFrameChange: (v: number) => void
    onAssignmentKindChange: (v: 'initial' | 'correction') => void
    onBuilderPromptChange: (v: string) => void
    onLoadFrame: () => void
    onAutoDetect: () => void
    onBuildMask: () => void
    onTrackForward: () => void
    onBack: () => void
    onNext: () => void
    onMouseDown: (e: MouseEvent<HTMLDivElement>) => void
    onMouseMove: (e: MouseEvent<HTMLDivElement>) => void
    onMouseUp: (e: MouseEvent<HTMLDivElement>) => void
    onMouseLeave: () => void
    onClick: (e: MouseEvent<HTMLDivElement>) => void
}

export default function StepSubject({
    assignmentFrame,
    assignmentKind,
    builderPrompt,
    assignmentBusy,
    builderLoadingFrame,
    builderSuggestingBoxes,
    builderBuildingMask,
    builderBuildingRange,
    hasAssignment,
    keyframeCount,
    keyframes,
    builderFrameDataUrl,
    builderFrameSize,
    builderMaskPreviewUrl,
    builderBox,
    activeBuilderBox,
    builderFgPoints,
    builderBgPoints,
    builderPointRadius,
    builderTool,
    builderImgRef,
    onBuilderToolChange,
    onAssignmentFrameChange,
    onAssignmentKindChange,
    onBuilderPromptChange,
    onLoadFrame,
    onAutoDetect,
    onBuildMask,
    onTrackForward,
    onBack,
    onNext,
    onMouseDown,
    onMouseMove,
    onMouseUp,
    onMouseLeave,
    onClick,
}: StepSubjectProps) {
    return (
        <div className="space-y-3">
            <h3 className="text-lg font-semibold text-white">Who is the subject?</h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                <Input
                    label="Keyframe Index"
                    type="number"
                    value={assignmentFrame}
                    onChange={e => onAssignmentFrameChange(parseInt(e.target.value || "0", 10))}
                />
                <Select
                    label="Anchor Type"
                    value={assignmentKind}
                    onChange={e => onAssignmentKindChange(e.target.value as 'initial' | 'correction')}
                    options={[
                        { value: 'initial', label: 'Initial Anchor' },
                        { value: 'correction', label: 'Correction Anchor' },
                    ]}
                />
                <Input
                    label="Auto-Detect Prompt"
                    value={builderPrompt}
                    onChange={e => onBuilderPromptChange(e.target.value)}
                    placeholder="person"
                />
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-[220px_minmax(0,1fr)_260px] gap-3">
                <div className="rounded border border-gray-700 bg-gray-900 p-3 space-y-2">
                    <div className="text-xs uppercase tracking-wide text-gray-500">Tool Palette</div>
                    <div className="grid grid-cols-3 gap-1">
                        <button
                            type="button"
                            onClick={() => onBuilderToolChange('box')}
                            className={`px-2 py-1.5 rounded text-xs font-semibold border ${
                                builderTool === 'box'
                                    ? 'bg-brand-500/20 border-brand-400 text-brand-200'
                                    : 'bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700'
                            }`}
                        >
                            Box
                        </button>
                        <button
                            type="button"
                            onClick={() => onBuilderToolChange('fg')}
                            className={`px-2 py-1.5 rounded text-xs font-semibold border ${
                                builderTool === 'fg'
                                    ? 'bg-green-500/20 border-green-400 text-green-200'
                                    : 'bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700'
                            }`}
                        >
                            FG Point
                        </button>
                        <button
                            type="button"
                            onClick={() => onBuilderToolChange('bg')}
                            className={`px-2 py-1.5 rounded text-xs font-semibold border ${
                                builderTool === 'bg'
                                    ? 'bg-red-500/20 border-red-400 text-red-200'
                                    : 'bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700'
                            }`}
                        >
                            BG Point
                        </button>
                    </div>
                    <button
                        type="button"
                        onClick={onLoadFrame}
                        disabled={assignmentBusy}
                        className="w-full px-3 py-2 rounded bg-gray-700 hover:bg-gray-600 text-white text-sm disabled:opacity-50"
                    >
                        {builderLoadingFrame ? "Loading..." : "Load Frame"}
                    </button>
                    <button
                        type="button"
                        onClick={onAutoDetect}
                        disabled={assignmentBusy || !builderFrameDataUrl}
                        className="w-full px-3 py-2 rounded bg-purple-600 hover:bg-purple-500 text-white text-sm disabled:opacity-50"
                    >
                        {builderSuggestingBoxes ? "Detecting..." : "Auto-Detect"}
                    </button>
                    <button
                        type="button"
                        onClick={onBuildMask}
                        disabled={assignmentBusy || !builderFrameDataUrl || !builderBox}
                        className="w-full px-3 py-2 rounded bg-brand-500 hover:bg-brand-600 text-white text-sm disabled:opacity-50"
                    >
                        {builderBuildingMask ? "Building..." : "Add Keyframe"}
                    </button>
                    <button
                        type="button"
                        onClick={onTrackForward}
                        disabled={assignmentBusy || !builderFrameDataUrl || !builderBox}
                        className="w-full px-3 py-2 rounded bg-cyan-600 hover:bg-cyan-500 text-white text-sm disabled:opacity-50"
                    >
                        {builderBuildingRange ? "Tracking..." : "Track Forward"}
                    </button>
                    <div className="text-[11px] text-gray-500 pt-1">
                        Draw a box first, then add FG/BG points.
                    </div>
                </div>

                <div className="rounded border border-gray-700 bg-gray-900/50 p-3">
                    <MaskBuilder
                        frameDataUrl={builderFrameDataUrl}
                        frameSize={builderFrameSize}
                        maskPreviewUrl={builderMaskPreviewUrl}
                        builderBox={builderBox}
                        activeBuilderBox={activeBuilderBox}
                        fgPoints={builderFgPoints}
                        bgPoints={builderBgPoints}
                        pointRadius={builderPointRadius}
                        builderTool={builderTool}
                        imageRef={builderImgRef}
                        onMouseDown={onMouseDown}
                        onMouseMove={onMouseMove}
                        onMouseUp={onMouseUp}
                        onMouseLeave={onMouseLeave}
                        onClick={onClick}
                    />
                </div>

                <div className="rounded border border-gray-700 bg-gray-900 p-3 space-y-2">
                    <div className="text-xs uppercase tracking-wide text-gray-500">Keyframes</div>
                    <div className="text-sm text-gray-300">
                        {hasAssignment
                            ? `Loaded ${keyframeCount} keyframe assignment(s).`
                            : "No keyframes yet. Build or import at least one mask."}
                    </div>
                    <div className="max-h-64 overflow-auto border border-gray-700 rounded p-2 bg-gray-900/40">
                        {keyframes.length === 0 ? (
                            <div className="text-xs text-gray-500">No keyframes added yet.</div>
                        ) : (
                            keyframes.map((kf) => (
                                <div key={`${kf.frame}:${kf.mask_asset}`} className="text-xs text-gray-300 font-mono">
                                    frame={kf.frame} kind={kf.kind}
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>

            <div className="flex justify-between">
                <button
                    type="button"
                    onClick={onBack}
                    className="px-4 py-2 rounded border border-gray-700 bg-gray-900 text-gray-200 hover:bg-gray-800"
                >
                    Back
                </button>
                <button
                    type="button"
                    disabled={!hasAssignment}
                    onClick={onNext}
                    className="px-4 py-2 rounded bg-brand-500 hover:bg-brand-600 text-white font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    Next: Refine Edges
                </button>
            </div>
        </div>
    )
}
