import { useState, useEffect, useCallback, useRef } from 'react'
import WipeComparison from './WipeComparison'
import { FaRegImage, FaChevronLeft, FaChevronRight, FaStepBackward, FaStepForward } from 'react-icons/fa'

interface QCInfo {
    input: { prefix: string; padding: number; ext: string; count: number }
    output: { prefix: string; padding: number; ext: string; count: number }
}

type CompositeMode = 'alpha' | 'checker' | 'white' | 'black'

const COMPOSITE_MODES: { value: CompositeMode; label: string }[] = [
    { value: 'alpha', label: 'Alpha (Raw)' },
    { value: 'checker', label: 'Checkerboard' },
    { value: 'white', label: 'White BG' },
    { value: 'black', label: 'Black BG' },
]

export default function QCTab() {
    const [frame, setFrame] = useState(0)
    const [compositeMode, setCompositeMode] = useState<CompositeMode>('alpha')
    const [qcInfo, setQcInfo] = useState<QCInfo | null>(null)
    const containerRef = useRef<HTMLDivElement>(null)

    // Auto-detect frame info from API
    useEffect(() => {
        fetch('/api/qc/info')
            .then(r => r.ok ? r.json() : null)
            .then(data => { if (data) setQcInfo(data) })
            .catch(() => {})
    }, [])

    const maxFrame = qcInfo ? Math.max(qcInfo.input.count, qcInfo.output.count) - 1 : 162
    const inputInfo = qcInfo?.input ?? { prefix: 'frame_', padding: 5, ext: 'png' }
    const outputInfo = qcInfo?.output ?? { prefix: '', padding: 6, ext: 'png' }

    const inputFrameStr = frame.toString().padStart(inputInfo.padding, '0')
    const outputFrameStr = frame.toString().padStart(outputInfo.padding, '0')

    const inputUrl = `/files/input_frames/${inputInfo.prefix}${inputFrameStr}.${inputInfo.ext}`
    const outputUrl = `/files/out/alpha/${outputInfo.prefix}${outputFrameStr}.${outputInfo.ext}`

    const clampFrame = useCallback((n: number) => Math.max(0, Math.min(n, maxFrame)), [maxFrame])

    const goTo = useCallback((n: number) => setFrame(clampFrame(n)), [clampFrame])
    const prev = useCallback(() => goTo(frame - 1), [frame, goTo])
    const next = useCallback(() => goTo(frame + 1), [frame, goTo])
    const prevJump = useCallback(() => goTo(frame - 10), [frame, goTo])
    const nextJump = useCallback(() => goTo(frame + 10), [frame, goTo])

    // Keyboard shortcuts: arrows for prev/next, shift+arrows for jump
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            // Don't capture if user is typing in an input
            if ((e.target as HTMLElement)?.tagName === 'INPUT') return

            if (e.key === 'ArrowLeft' || e.key === 'j') {
                e.preventDefault()
                e.shiftKey ? prevJump() : prev()
            } else if (e.key === 'ArrowRight' || e.key === 'k') {
                e.preventDefault()
                e.shiftKey ? nextJump() : next()
            } else if (e.key === 'Home') {
                e.preventDefault()
                goTo(0)
            } else if (e.key === 'End') {
                e.preventDefault()
                goTo(maxFrame)
            }
        }
        window.addEventListener('keydown', handler)
        return () => window.removeEventListener('keydown', handler)
    }, [prev, next, prevJump, nextJump, goTo, maxFrame])

    return (
        <div className="space-y-4" ref={containerRef}>
            {/* Header row */}
            <div className="flex justify-between items-end border-b border-gray-800 pb-3">
                <div>
                    <h2 className="text-2xl font-bold text-white">Quality Control</h2>
                    <p className="text-gray-400 text-sm mt-0.5">Inspect matte quality with A/B wipe comparison.</p>
                </div>
                <div className="flex items-center gap-3">
                    {/* Composite mode dropdown */}
                    <select
                        value={compositeMode}
                        onChange={e => setCompositeMode(e.target.value as CompositeMode)}
                        className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-xs text-gray-300 focus:outline-none focus:border-brand-500"
                    >
                        {COMPOSITE_MODES.map(m => (
                            <option key={m.value} value={m.value}>{m.label}</option>
                        ))}
                    </select>
                </div>
            </div>

            {/* Wipe comparison */}
            <div className="bg-gray-800 rounded-xl overflow-hidden shadow-2xl border border-gray-700">
                <WipeComparison
                    leftImage={inputUrl}
                    rightImage={outputUrl}
                    leftLabel="Input (RGB)"
                    rightLabel="Output (Alpha)"
                    compositeMode={compositeMode}
                />
            </div>

            {/* Frame scrubber */}
            <div className="bg-gray-800/50 rounded-xl border border-gray-700 p-3">
                <div className="flex items-center gap-3">
                    {/* Navigation buttons */}
                    <div className="flex items-center gap-1">
                        <button onClick={() => goTo(0)} className="p-1.5 text-gray-400 hover:text-white transition-colors rounded hover:bg-gray-700" title="First frame">
                            <FaStepBackward className="text-xs" />
                        </button>
                        <button onClick={prevJump} className="p-1.5 text-gray-400 hover:text-white transition-colors rounded hover:bg-gray-700" title="Back 10 (Shift+Left)">
                            <FaChevronLeft className="text-xs" />
                            <FaChevronLeft className="text-xs -ml-1.5" />
                        </button>
                        <button onClick={prev} className="p-1.5 text-gray-400 hover:text-white transition-colors rounded hover:bg-gray-700" title="Previous (Left / J)">
                            <FaChevronLeft className="text-xs" />
                        </button>
                    </div>

                    {/* Slider */}
                    <input
                        type="range"
                        min={0}
                        max={maxFrame}
                        value={frame}
                        onChange={e => setFrame(parseInt(e.target.value))}
                        className="flex-1 h-1.5 accent-brand-500 bg-gray-700 rounded-full cursor-pointer"
                    />

                    {/* Forward buttons */}
                    <div className="flex items-center gap-1">
                        <button onClick={next} className="p-1.5 text-gray-400 hover:text-white transition-colors rounded hover:bg-gray-700" title="Next (Right / K)">
                            <FaChevronRight className="text-xs" />
                        </button>
                        <button onClick={nextJump} className="p-1.5 text-gray-400 hover:text-white transition-colors rounded hover:bg-gray-700" title="Forward 10 (Shift+Right)">
                            <FaChevronRight className="text-xs" />
                            <FaChevronRight className="text-xs -ml-1.5" />
                        </button>
                        <button onClick={() => goTo(maxFrame)} className="p-1.5 text-gray-400 hover:text-white transition-colors rounded hover:bg-gray-700" title="Last frame">
                            <FaStepForward className="text-xs" />
                        </button>
                    </div>

                    {/* Frame number input */}
                    <div className="flex items-center gap-2 bg-gray-900 px-2 py-1 rounded-lg border border-gray-700 min-w-[100px]">
                        <input
                            type="number"
                            value={frame}
                            onChange={e => goTo(parseInt(e.target.value) || 0)}
                            className="w-14 bg-transparent text-right font-mono text-white text-sm focus:outline-none"
                            min={0}
                            max={maxFrame}
                        />
                        <span className="text-gray-500 text-xs font-mono">/ {maxFrame}</span>
                    </div>
                </div>
                <div className="text-center mt-1">
                    <span className="text-[10px] text-gray-600">J/K or Arrow keys to navigate &middot; Shift for x10 &middot; Home/End for first/last</span>
                </div>
            </div>

            {/* Side-by-side previews */}
            <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-800/50 p-3 rounded-xl border border-gray-700">
                    <h3 className="font-semibold text-gray-300 mb-2 flex items-center gap-2 text-sm"><FaRegImage /> Input Source</h3>
                    <div className="aspect-video bg-black rounded flex items-center justify-center text-gray-600 overflow-hidden">
                        <img src={inputUrl} className="w-full h-full object-contain" alt="Input Preview" />
                    </div>
                </div>
                <div className="bg-gray-800/50 p-3 rounded-xl border border-gray-700">
                    <h3 className="font-semibold text-gray-300 mb-2 flex items-center gap-2 text-sm"><FaRegImage /> Matte Output</h3>
                    <div className="aspect-video bg-black rounded flex items-center justify-center text-gray-600 overflow-hidden">
                        <img src={outputUrl} className="w-full h-full object-contain" alt="Output Preview" />
                    </div>
                </div>
            </div>
        </div>
    )
}
