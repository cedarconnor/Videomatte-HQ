import { useState, useEffect, useRef } from 'react'
import { FaTerminal, FaArrowDown } from 'react-icons/fa'

export default function JobDetail({ jobId }: { jobId: string }) {
    const [logs, setLogs] = useState("")
    const logEndRef = useRef<HTMLDivElement>(null)
    const scrollContainerRef = useRef<HTMLDivElement>(null)
    const [autoScroll, setAutoScroll] = useState(true)
    const [isAtBottom, setIsAtBottom] = useState(true)

    useEffect(() => {
        fetchLogs()
        const interval = setInterval(fetchLogs, 2000)
        return () => clearInterval(interval)
    }, [jobId])

    useEffect(() => {
        if (autoScroll && logEndRef.current) {
            logEndRef.current.scrollIntoView({ behavior: 'smooth' })
        }
    }, [logs, autoScroll])

    // Detect manual scroll to auto-disable auto-scroll
    useEffect(() => {
        const container = scrollContainerRef.current
        if (!container) return
        const handleScroll = () => {
            const atBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 40
            setIsAtBottom(atBottom)
            if (!atBottom && autoScroll) {
                setAutoScroll(false)
            }
        }
        container.addEventListener('scroll', handleScroll)
        return () => container.removeEventListener('scroll', handleScroll)
    }, [autoScroll])

    async function fetchLogs() {
        try {
            const res = await fetch(`/api/jobs/${jobId}/logs`)
            const data = await res.json()
            setLogs(data.logs)
        } catch (err) {
            console.error("Failed to fetch logs", err)
        }
    }

    const scrollToBottom = () => {
        setAutoScroll(true)
        logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }

    return (
        <div className="h-full flex flex-col">
            <div className="bg-gray-800 p-3 border-b border-gray-700 flex justify-between items-center">
                <div className="flex items-center gap-2 text-gray-300 font-mono text-sm">
                    <FaTerminal />
                    <span>Console Output</span>
                </div>
                <div className="text-xs flex items-center gap-3">
                    {autoScroll && (
                        <span className="flex items-center gap-1.5 text-brand-400 animate-pulse">
                            <span className="w-1.5 h-1.5 rounded-full bg-brand-400" />
                            Live
                        </span>
                    )}
                    <label className="flex items-center gap-2 text-gray-400 cursor-pointer select-none">
                        <input
                            type="checkbox"
                            checked={autoScroll}
                            onChange={e => setAutoScroll(e.target.checked)}
                            className="accent-brand-500"
                        />
                        Auto-scroll
                    </label>
                </div>
            </div>
            <div className="relative flex-1">
                <div
                    ref={scrollContainerRef}
                    className="absolute inset-0 bg-[#0d1117] p-4 overflow-y-auto font-mono text-xs text-gray-300 whitespace-pre-wrap leading-relaxed"
                >
                    {logs || <span className="text-gray-600 italic">Waiting for logs...</span>}
                    <div ref={logEndRef} />
                </div>
                {/* Scroll-to-bottom FAB when not at bottom */}
                {!isAtBottom && !autoScroll && (
                    <button
                        onClick={scrollToBottom}
                        className="absolute bottom-4 right-4 p-2 bg-brand-500 hover:bg-brand-600 text-white rounded-full shadow-lg transition-all animate-bounce"
                        title="Scroll to bottom & resume auto-scroll"
                    >
                        <FaArrowDown className="text-sm" />
                    </button>
                )}
            </div>
        </div>
    )
}
