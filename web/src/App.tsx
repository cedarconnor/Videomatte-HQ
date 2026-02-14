import { useState } from 'react'
import { FaPlay, FaHistory, FaCog, FaChartBar } from 'react-icons/fa'
import { clsx } from 'clsx'
import RunTab from './components/RunTab'
import JobsTab from './components/JobsTab'
import SettingsTab from './components/SettingsTab'
import QCTab from './components/QCTab'

type Tab = 'run' | 'jobs' | 'qc' | 'settings'

function App() {
    const [activeTab, setActiveTab] = useState<Tab>('run')

    return (
        <div className="min-h-screen bg-gray-900 text-gray-100 font-sans flex flex-col">
            {/* Header */}
            <header className="bg-gray-800 border-b border-gray-700 px-6 py-4 flex items-center justify-between shadow-md">
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-gradient-to-br from-brand-500 to-purple-600 rounded-lg flex items-center justify-center font-bold text-white">
                        VM
                    </div>
                    <h1 className="text-xl font-bold tracking-tight">VideoMatte-HQ</h1>
                </div>
                <div className="text-sm text-gray-400">v0.1.0</div>
            </header>

            {/* Main Content */}
            <div className="flex flex-1 overflow-hidden">
                {/* Sidebar */}
                <aside className="w-64 bg-gray-800/50 border-r border-gray-700 flex flex-col">
                    <nav className="p-4 space-y-2">
                        <SidebarItem
                            active={activeTab === 'run'}
                            icon={<FaPlay />}
                            label="Run Job"
                            onClick={() => setActiveTab('run')}
                        />
                        <SidebarItem
                            active={activeTab === 'jobs'}
                            icon={<FaHistory />}
                            label="Job History"
                            onClick={() => setActiveTab('jobs')}
                        />
                        <SidebarItem
                            active={activeTab === 'qc'}
                            icon={<FaChartBar />}
                            label="Quality Control"
                            onClick={() => setActiveTab('qc')}
                        />
                        <SidebarItem
                            active={activeTab === 'settings'}
                            icon={<FaCog />}
                            label="Settings"
                            onClick={() => setActiveTab('settings')}
                        />
                    </nav>
                </aside>

                {/* Tab Content */}
                <main className="flex-1 overflow-auto bg-gray-900 p-8">
                    <div className="max-w-6xl mx-auto">
                        {activeTab === 'run' && <RunTab onSuccess={() => setActiveTab('jobs')} />}
                        {activeTab === 'jobs' && <JobsTab />}
                        {activeTab === 'qc' && <QCTab />}
                        {activeTab === 'settings' && <SettingsTab />}
                    </div>
                </main>
            </div>
        </div>
    )
}

function SidebarItem({ active, icon, label, onClick }: { active: boolean; icon: React.ReactNode; label: string; onClick: () => void }) {
    return (
        <button
            onClick={onClick}
            className={clsx(
                "w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 text-sm font-medium",
                active
                    ? "bg-brand-500/10 text-brand-500 shadow-sm border border-brand-500/20"
                    : "text-gray-400 hover:bg-gray-800 hover:text-gray-200"
            )}
        >
            <span className="text-lg">{icon}</span>
            {label}
        </button>
    )
}

export default App
