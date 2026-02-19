import { FaCheckCircle, FaPlay, FaSpinner } from 'react-icons/fa'

type StageJobState = 'idle' | 'queued' | 'running' | 'completed' | 'failed'

interface StepStageApprovalProps {
    title: string
    description: string
    reviewPath: string
    runLabel: string
    approveLabel: string
    approved: boolean
    canApprove: boolean
    loading: boolean
    jobId: string | null
    jobState: StageJobState
    progressPct: number
    progressLabel: string
    timeRemaining: string
    jobError: string | null
    onBack: () => void
    onRun: () => void
    onApprove: () => void
    onNext: () => void
    nextLabel: string
}

export default function StepStageApproval({
    title,
    description,
    reviewPath,
    runLabel,
    approveLabel,
    approved,
    canApprove,
    loading,
    jobId,
    jobState,
    progressPct,
    progressLabel,
    timeRemaining,
    jobError,
    onBack,
    onRun,
    onApprove,
    onNext,
    nextLabel,
}: StepStageApprovalProps) {
    const safePct = Math.max(0, Math.min(100, Math.round(progressPct * 100)))
    const showProgress = jobState === 'queued' || jobState === 'running' || jobState === 'completed' || jobState === 'failed'
    return (
        <div className="space-y-3">
            <h3 className="text-lg font-semibold text-white">{title}</h3>
            <p className="text-xs text-gray-400">{description}</p>

            <div className="rounded border border-gray-700 bg-gray-900/60 p-3 text-xs text-gray-300">
                Review output path: <span className="font-mono text-gray-100">{reviewPath}</span>
            </div>

            {showProgress && (
                <div className="rounded border border-gray-700 bg-gray-900/70 p-3 space-y-2">
                    <div className="flex items-center justify-between text-xs">
                        <span className="text-gray-300 font-semibold uppercase tracking-wide">Stage Progress</span>
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
                        <span>{jobId ? `Job ${jobId.slice(0, 8)}` : 'Waiting for job id'}</span>
                        <span>{timeRemaining}</span>
                    </div>
                    {jobError && (
                        <div className="text-xs text-red-300 bg-red-500/10 border border-red-500/30 rounded px-2 py-1">
                            {jobError}
                        </div>
                    )}
                </div>
            )}

            <div className="rounded border border-gray-700 bg-gray-900/60 p-3 flex flex-wrap gap-2">
                <button
                    type="button"
                    onClick={onRun}
                    disabled={loading || jobState === 'running' || jobState === 'queued'}
                    className="px-4 py-2 rounded bg-brand-500 hover:bg-brand-600 text-white font-semibold disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-2"
                >
                    {loading || jobState === 'running' || jobState === 'queued'
                        ? <FaSpinner className="animate-spin" />
                        : <FaPlay />}
                    {runLabel}
                </button>
                <button
                    type="button"
                    onClick={onApprove}
                    disabled={!canApprove}
                    className="px-4 py-2 rounded border border-green-500/40 bg-green-500/10 text-green-300 hover:bg-green-500/20 font-semibold disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-2"
                >
                    <FaCheckCircle />
                    {approveLabel}
                </button>
                {approved && (
                    <div className="text-xs text-green-300 self-center">Approved.</div>
                )}
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
                    onClick={onNext}
                    disabled={!approved}
                    className="px-4 py-2 rounded bg-brand-500 hover:bg-brand-600 text-white font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {nextLabel}
                </button>
            </div>
        </div>
    )
}
