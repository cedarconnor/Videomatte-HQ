import { FaPlay, FaSpinner } from 'react-icons/fa'

interface StepRenderProps {
    inputBasename: string
    frameStart: number
    frameEnd: number
    loading: boolean
    canStart: boolean
    jobId: string | null
    jobState: 'idle' | 'queued' | 'running' | 'completed' | 'failed'
    progressPct: number
    progressLabel: string
    timeRemaining: string
    jobError: string | null
    onBack: () => void
    onStart: () => void
    onLaunchQC: () => void
}

export default function StepRender({
    inputBasename,
    frameStart,
    frameEnd,
    loading,
    canStart,
    jobId,
    jobState,
    progressPct,
    progressLabel,
    timeRemaining,
    jobError,
    onBack,
    onStart,
    onLaunchQC,
}: StepRenderProps) {
    const safePct = Math.max(0, Math.min(100, Math.round(progressPct * 100)))
    const showProgress = jobState === 'queued' || jobState === 'running' || jobState === 'completed' || jobState === 'failed'

    return (
        <div className="space-y-3">
            <h3 className="text-lg font-semibold text-white">Ready to Process</h3>
            <div className="rounded border border-gray-700 bg-gray-900 p-3 text-sm text-gray-300">
                Processing <span className="font-semibold text-white">{inputBasename}</span> from frame{" "}
                <span className="font-semibold text-white">{frameStart}</span> to{" "}
                <span className="font-semibold text-white">{frameEnd >= 0 ? frameEnd : "end"}</span>.
            </div>

            {showProgress && (
                <div className="rounded border border-gray-700 bg-gray-900/70 p-3 space-y-2">
                    <div className="flex items-center justify-between text-xs">
                        <span className="text-gray-300 font-semibold uppercase tracking-wide">Render Progress</span>
                        <span className="text-brand-300 font-mono">{progressLabel}</span>
                    </div>
                    <div className="h-3 rounded bg-gray-800 overflow-hidden border border-gray-700">
                        <div
                            className={`h-full transition-all duration-300 ${
                                jobState === 'failed'
                                    ? 'bg-red-500'
                                    : jobState === 'completed'
                                        ? 'bg-green-500'
                                        : 'bg-brand-500'
                            }`}
                            style={{ width: `${safePct}%` }}
                        />
                    </div>
                    <div className="flex items-center justify-between text-xs text-gray-400">
                        <span>
                            {jobId ? `Job ${jobId.slice(0, 8)}` : 'Waiting for job id'}
                        </span>
                        <span>{timeRemaining}</span>
                    </div>
                    {jobError && (
                        <div className="text-xs text-red-300 bg-red-500/10 border border-red-500/30 rounded px-2 py-1">
                            {jobError}
                        </div>
                    )}
                </div>
            )}

            <div className="flex justify-between">
                <button
                    type="button"
                    onClick={onBack}
                    className="px-4 py-2 rounded border border-gray-700 bg-gray-900 text-gray-200 hover:bg-gray-800"
                >
                    Back
                </button>
                <div className="flex gap-2">
                    {jobState === 'completed' && (
                        <button
                            type="button"
                            onClick={onLaunchQC}
                            className="px-4 py-2 rounded border border-green-500/40 bg-green-500/10 text-green-300 hover:bg-green-500/20 font-semibold"
                        >
                            Launch Quality Control
                        </button>
                    )}
                    <button
                        type="button"
                        onClick={onStart}
                        disabled={loading || !canStart || jobState === 'running' || jobState === 'queued'}
                        className="px-5 py-2 rounded bg-brand-500 hover:bg-brand-600 text-white font-bold disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                        {loading ? <FaSpinner className="animate-spin" /> : <FaPlay />}
                        {jobState === 'running' || jobState === 'queued' ? 'Rendering...' : 'Start Render'}
                    </button>
                </div>
            </div>
        </div>
    )
}

