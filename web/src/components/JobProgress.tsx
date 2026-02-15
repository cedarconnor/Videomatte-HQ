import { useState, useEffect } from 'react'
import { clsx } from 'clsx'


interface JobProgressProps {
    jobId: string | null
}

const STAGES = [
    { id: 'load', label: 'Load', match: ['Stage 0:'], color: 'bg-gray-500' },
    { id: 'assignment', label: 'Assign', match: ['Stage 1:'], color: 'bg-blue-500' },
    { id: 'memory', label: 'Memory', match: ['Stage 2:'], color: 'bg-purple-500' },
    { id: 'refine', label: 'Refine', match: ['Stage 3:'], color: 'bg-cyan-500' },
    { id: 'temporal_cleanup', label: 'Temporal', match: ['Stage 4:'], color: 'bg-emerald-500' },
    { id: 'matte_tuning', label: 'Matte', match: ['Stage 5:'], color: 'bg-orange-500' },
    { id: 'output', label: 'Output', match: ['Stage 6:'], color: 'bg-red-500' },
]

export default function JobProgress({ jobId }: JobProgressProps) {
    const [logs, setLogs] = useState('')
    const [activeStageIndex, setActiveStageIndex] = useState(-1)
    const [progressText, setProgressText] = useState('')

    useEffect(() => {
        if (!jobId) {
            setLogs('')
            setActiveStageIndex(-1)
            setProgressText('')
            return
        }

        const interval = setInterval(async () => {
            try {
                const res = await fetch(`/api/jobs/${jobId}/logs`)
                if (res.ok) {
                    const data = await res.json()
                    const newLogs = data.logs
                    if (newLogs !== logs) {
                        setLogs(newLogs)
                        parseLogs(newLogs)
                    }
                }
            } catch (e) {
                console.error("Failed to poll logs", e)
            }
        }, 2000)

        return () => clearInterval(interval)
    }, [jobId, logs])

    const parseLogs = (logText: string) => {
        const lines = logText.split('\n')
        let lastStageIdx = -1
        let lastProgress = ''

        for (const line of lines) {
            // Check for stage markers
            for (let i = 0; i < STAGES.length; i++) {
                if (STAGES[i].match.some(m => line.includes(m))) {
                    lastStageIdx = i
                }
            }

            // Check for frame progress once per-frame stages start
            // e.g. "Stage 2: frame 10/100"
            if (line.includes('frame ') && line.includes('/')) {
                lastProgress = line.split('frame ')[1].split(']')[0] // extract "10/100" roughly
            }
        }

        if (lastStageIdx > activeStageIndex) {
            setActiveStageIndex(lastStageIdx)
        }

        // Update progress text if we found relevant frame info
        if (lastProgress && lastStageIdx >= 2) { // Show frame counts once per-frame processing starts
            setProgressText(lastProgress)
        } else if (lastStageIdx >= 0) {
            setProgressText(STAGES[lastStageIdx].label)
        }
    }

    if (!jobId || activeStageIndex === -1) return null

    return (
        <div className="w-full max-w-4xl mx-auto mb-6">
            <div className="flex justify-between items-center mb-1">
                <h3 className="text-xs font-bold text-gray-400 uppercase tracking-wider">Pipeline Progress</h3>
                <span className="text-xs font-mono text-brand-400 animate-pulse">{progressText}</span>
            </div>

            <div className="flex h-2 w-full bg-gray-800 rounded-full overflow-hidden">
                {STAGES.map((stage, idx) => {
                    const isActive = idx === activeStageIndex
                    const isComplete = idx < activeStageIndex

                    return (
                        <div
                            key={stage.id}
                            className={clsx(
                                "flex-1 transition-all duration-500 first:rounded-l-full last:rounded-r-full relative group",
                                isComplete ? stage.color : "bg-gray-800",
                                isActive ? clsx(stage.color, "animate-pulse brightness-125") : "opacity-30"
                            )}
                        >
                            {/* Separator */}
                            {idx < STAGES.length - 1 && (
                                <div className="absolute right-0 top-0 bottom-0 w-0.5 bg-gray-900/50 z-10" />
                            )}

                            {/* Hover Tooltip */}
                            <div className="opacity-0 group-hover:opacity-100 absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-xs text-white rounded pointer-events-none transition-opacity whitespace-nowrap z-20 border border-gray-700">
                                {stage.label}
                                {(isActive || isComplete) && " ✓"}
                            </div>
                        </div>
                    )
                })}
            </div>
            <div className="flex justify-between mt-1 px-1">
                {STAGES.map((stage, idx) => (
                    <span
                        key={stage.id}
                        className={clsx(
                            "text-[10px] uppercase tracking-wide font-semibold transition-colors duration-300",
                            idx <= activeStageIndex ? "text-gray-300" : "text-gray-700"
                        )}
                    >
                        {stage.label}
                    </span>
                ))}
            </div>
        </div>
    )
}
