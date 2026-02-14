import { useState, useEffect, useRef } from 'react'
import { FaTerminal } from 'react-icons/fa'

export default function JobDetail({ jobId }: { jobId: string }) {
    const [logs, setLogs] = useState("")
    const logEndRef = useRef<HTMLDivElement>(null)
    const [autoScroll, setAutoScroll] = useState(true)

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

    async function fetchLogs() {
        try {
            const res = await fetch(`/api/jobs/${jobId}/logs`)
            const data = await res.json()
            setLogs(data.logs)
        } catch (err) {
            console.error("Failed to fetch logs", err)
        }
    }

    return (
        <div className="h-full flex flex-col">
            <div className="bg-gray-800 p-3 border-b border-gray-700 flex justify-between">
                <div className="flex items-center gap-2 text-gray-300 font-mono text-sm">
                    <FaTerminal />
                    <span>Console Output</span>
                </div>
                <div className="text-xs flex items-center gap-2">
                    <label className="flex items-center gap-2 text-gray-400 cursor-pointer">
                        <input type="checkbox" checked={autoScroll} onChange={e => setAutoScroll(e.target.checked)} />
                        Auto-scroll
                    </label>
                </div>
            </div>
            <div className="flex-1 bg-[#0d1117] p-4 overflow-y-auto font-mono text-xs text-gray-300 whitespace-pre-wrap leading-relaxed">
                {logs || <span className="text-gray-600 italic">Waiting for logs...</span>}
                <div ref={logEndRef} />
            </div>
        </div>
    )
}
