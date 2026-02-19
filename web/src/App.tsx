import { useState, useEffect, useRef } from 'react'
import { FaLayerGroup, FaPlay, FaList, FaCog, FaMagic } from 'react-icons/fa'
import { clsx } from 'clsx'
import RunTab, { RUN_STEPS_BASE } from './components/RunTab'
import type { RunStepId, RunViewMode } from './components/RunTab'
import JobsTab from './components/JobsTab'
import SettingsTab from './components/SettingsTab'
import QCTab from './components/QCTab'
import JobProgress from './components/JobProgress'
import { ToastProvider, useToast } from './components/Toast'

type Tab = 'run' | 'jobs' | 'settings' | 'qc'

function SidebarItem({ active, icon, label, shortcut, onClick }: { active: boolean, icon: React.ReactNode, label: string, shortcut?: string, onClick: () => void }) {
    return (
        <button
            onClick={onClick}
            className={clsx(
                "w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 text-sm font-medium group",
                active
                    ? "bg-brand-500/10 text-brand-400 shadow-sm shadow-brand-500/5 border border-brand-500/20"
                    : "text-gray-400 hover:bg-gray-800 hover:text-gray-200"
            )}
        >
            <span className={clsx("text-lg", active ? "text-brand-500" : "text-gray-500")}>{icon}</span>
            <span className="flex-1 text-left">{label}</span>
            {shortcut && (
                <span className="text-[10px] text-gray-600 font-mono opacity-0 group-hover:opacity-100 transition-opacity">{shortcut}</span>
            )}
        </button>
    )
}

function AppInner() {
    const [activeTab, setActiveTab] = useState<Tab>('run')
    const [activeJobId, setActiveJobId] = useState<string | null>(null)
    const [runViewMode, setRunViewMode] = useState<RunViewMode>('wizard')
    const [activeRunStage, setActiveRunStage] = useState<RunStepId>('io')
    const [requestedRunStage, setRequestedRunStage] = useState<RunStepId | null>(null)
    const [runStageRequestNonce, setRunStageRequestNonce] = useState(0)
    const prevJobStatuses = useRef<Record<string, string>>({})
    const { addToast } = useToast()

    // Poll for active jobs — drive progress bar + toast notifications
    useEffect(() => {
        const checkActiveJobs = async () => {
            try {
                const res = await fetch('/api/jobs')
                if (res.ok) {
                    const jobs = await res.json()

                    // Find running job
                    const running = jobs.find((j: any) => j.status === 'running')
                    if (running) {
                        setActiveJobId(running.id)
                    } else {
                        setActiveJobId(null)
                    }

                    // Check for status transitions → toast
                    for (const job of jobs) {
                        const prevStatus = prevJobStatuses.current[job.id]
                        if (prevStatus && prevStatus !== job.status) {
                            if (job.status === 'completed') {
                                addToast(
                                    `Job ${job.id.slice(0, 8)} completed successfully!`,
                                    'success',
                                    {
                                        label: 'View Result (A/B)',
                                        onClick: () => setActiveTab('jobs'),
                                    }
                                )
                            } else if (job.status === 'failed') {
                                addToast(`Job ${job.id.slice(0, 8)} failed: ${job.error || 'Unknown error'}`, 'error')
                            }
                        }
                        prevJobStatuses.current[job.id] = job.status
                    }
                }
            } catch (e) {
                // silent
            }
        }

        const interval = setInterval(checkActiveJobs, 3000)
        checkActiveJobs()
        return () => clearInterval(interval)
    }, [addToast])

    // Global keyboard shortcuts
    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            // Tab switching: Ctrl+1-4
            if ((e.ctrlKey || e.metaKey) && !e.shiftKey) {
                const tabs: Tab[] = ['run', 'jobs', 'qc', 'settings']
                const num = parseInt(e.key)
                if (num >= 1 && num <= 4) {
                    e.preventDefault()
                    setActiveTab(tabs[num - 1])
                }
            }
        }
        window.addEventListener('keydown', handler)
        return () => window.removeEventListener('keydown', handler)
    }, [])

    return (
        <div className="min-h-screen bg-gray-900 text-gray-100 font-sans flex flex-col">
            {/* Header */}
            <header className="bg-gray-800/80 border-b border-gray-700/50 px-6 py-4 shadow-md backdrop-blur-md z-20">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="bg-gradient-to-br from-brand-500 to-indigo-600 p-2 rounded-lg shadow-lg shadow-brand-500/20">
                            <FaLayerGroup className="text-white text-xl" />
                        </div>
                        <div>
                            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">VideoMatte-HQ</h1>
                            <p className="text-xs text-gray-500 font-medium">Production Matting Pipeline</p>
                        </div>
                    </div>
                    {/* Keyboard hint */}
                    <div className="text-[10px] text-gray-600 font-mono hidden lg:block">
                        Ctrl+1-4 switch tabs
                    </div>
                </div>

                {activeJobId && (
                    <div className="mt-4 animate-in fade-in slide-in-from-top-2 duration-500">
                        <JobProgress jobId={activeJobId} />
                    </div>
                )}
            </header>

            {/* Main Content */}
            <div className="flex flex-1 overflow-hidden">
                {/* Sidebar */}
                <aside className="w-64 bg-gray-800/30 border-r border-gray-700/50 flex flex-col backdrop-blur-sm">
                    <nav className="p-3 space-y-1">
                        <SidebarItem
                            active={activeTab === 'run'}
                            icon={<FaPlay />}
                            label="Run Job"
                            shortcut="Ctrl+1"
                            onClick={() => setActiveTab('run')}
                        />
                        {activeTab === 'run' && runViewMode === 'pro' && (
                            <div className="ml-4 mr-1 mt-1 mb-2 space-y-1 border-l border-gray-700/60 pl-2">
                                {RUN_STEPS_BASE.map((step) => (
                                    <button
                                        key={step.id}
                                        type="button"
                                        onClick={() => {
                                            setActiveTab('run')
                                            setRequestedRunStage(step.id)
                                            setRunStageRequestNonce((n) => n + 1)
                                        }}
                                        className={clsx(
                                            'w-full text-left text-xs rounded px-2 py-1.5 transition-colors',
                                            activeRunStage === step.id
                                                ? 'bg-brand-500/15 text-brand-300'
                                                : 'text-gray-400 hover:bg-gray-800 hover:text-gray-200'
                                        )}
                                    >
                                        {step.label}
                                    </button>
                                ))}
                            </div>
                        )}
                        <SidebarItem
                            active={activeTab === 'jobs'}
                            icon={<FaList />}
                            label="Job Queue"
                            shortcut="Ctrl+2"
                            onClick={() => setActiveTab('jobs')}
                        />
                        <div className="my-3 border-t border-gray-700/50 mx-2" />
                        <SidebarItem
                            active={activeTab === 'qc'}
                            icon={<FaMagic />}
                            label="Quality Control"
                            shortcut="Ctrl+3"
                            onClick={() => setActiveTab('qc')}
                        />
                        <SidebarItem
                            active={activeTab === 'settings'}
                            icon={<FaCog />}
                            label="Settings"
                            shortcut="Ctrl+4"
                            onClick={() => setActiveTab('settings')}
                        />
                    </nav>
                </aside>

                {/* Tab Content */}
                <main className="flex-1 overflow-auto bg-gray-900 p-6 relative">
                    <div className={clsx(activeTab === 'run' && runViewMode === 'pro' ? 'max-w-none' : 'max-w-5xl mx-auto')}>
                        <div className={clsx(activeTab === 'run' ? 'block' : 'hidden')}>
                            <RunTab
                                onSuccess={() => setActiveTab('jobs')}
                                onLaunchQC={() => setActiveTab('qc')}
                                onRunModeChange={setRunViewMode}
                                onProStageChange={setActiveRunStage}
                                requestedProStage={requestedRunStage}
                                requestedProStageNonce={runStageRequestNonce}
                            />
                        </div>
                        {activeTab === 'jobs' && <JobsTab />}
                        {activeTab === 'settings' && <SettingsTab />}
                        {activeTab === 'qc' && <QCTab />}
                    </div>
                </main>
            </div>
        </div>
    )
}

function App() {
    return (
        <ToastProvider>
            <AppInner />
        </ToastProvider>
    )
}

export default App
