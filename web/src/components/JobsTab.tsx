import { useState, useEffect } from 'react'
import { FaSync, FaClock } from 'react-icons/fa'
import { clsx } from 'clsx'
import JobDetail from './JobDetail'

interface Job {
    id: string
    status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
    created_at: string
    completed_at?: string
    error?: string
}

export default function JobsTab() {
    const [jobs, setJobs] = useState<Job[]>([])
    const [selectedJobId, setSelectedJobId] = useState<string | null>(null)

    useEffect(() => {
        fetchJobs()
        const interval = setInterval(fetchJobs, 2000)
        return () => clearInterval(interval)
    }, [])

    async function fetchJobs() {
        try {
            const res = await fetch('/api/jobs')
            const data = await res.json()
            // Sort by newest first
            setJobs(data.sort((a: Job, b: Job) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()))
        } catch (err) {
            console.error("Failed to fetch jobs", err)
        }
    }

    return (
        <div className="flex h-[calc(100vh-8rem)] gap-6">
            {/* Job List */}
            <div className="w-1/3 bg-gray-800/50 border border-gray-700 rounded-xl overflow-hidden flex flex-col">
                <div className="p-4 border-b border-gray-700 flex justify-between items-center bg-gray-800">
                    <h3 className="font-semibold text-gray-200">Job History</h3>
                    <button onClick={fetchJobs} className="text-gray-400 hover:text-white transition-colors">
                        <FaSync />
                    </button>
                </div>
                <div className="flex-1 overflow-y-auto p-2 space-y-2">
                    {jobs.map(job => (
                        <div
                            key={job.id}
                            onClick={() => setSelectedJobId(job.id)}
                            className={clsx(
                                "p-3 rounded-lg cursor-pointer border transition-all",
                                selectedJobId === job.id
                                    ? "bg-brand-500/10 border-brand-500/50 shadow-sm"
                                    : "bg-gray-800 border-gray-700 hover:border-gray-600"
                            )}
                        >
                            <div className="flex justify-between items-start mb-2">
                                <span className={clsx(
                                    "text-xs font-bold px-2 py-0.5 rounded-full uppercase tracking-wide",
                                    getStatusColor(job.status)
                                )}>
                                    {job.status}
                                </span>
                                <span className="text-xs text-gray-500 flex items-center gap-1">
                                    <FaClock className="text-[10px]" />
                                    {new Date(job.created_at).toLocaleTimeString()}
                                </span>
                            </div>
                            <div className="text-xs text-gray-400 truncate font-mono">
                                {job.id.slice(0, 8)}...
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Job Detail View */}
            <div className="flex-1 bg-gray-800/30 border border-gray-700 rounded-xl overflow-hidden">
                {selectedJobId ? (
                    <JobDetail jobId={selectedJobId} />
                ) : (
                    <div className="h-full flex items-center justify-center text-gray-500">
                        Select a job to view details
                    </div>
                )}
            </div>
        </div>
    )
}

function getStatusColor(status: string) {
    switch (status) {
        case 'completed': return 'bg-green-500/20 text-green-400'
        case 'failed': return 'bg-red-500/20 text-red-400'
        case 'running': return 'bg-blue-500/20 text-blue-400 animate-pulse'
        case 'queued': return 'bg-yellow-500/20 text-yellow-400'
        default: return 'bg-gray-700 text-gray-400'
    }
}
