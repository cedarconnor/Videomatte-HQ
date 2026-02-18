import { ReactNode } from 'react'

interface DashboardStage {
    id: string
    label: string
}

interface DashboardLayoutProps {
    title: string
    subtitle?: string
    stages: DashboardStage[]
    activeStage?: string
    onStageClick?: (id: string) => void
    onSwitchToWizard: () => void
    showLeftNav?: boolean
    headerActions?: ReactNode
    rightPanel?: ReactNode
    children: ReactNode
}

export default function DashboardLayout({
    title,
    subtitle,
    stages,
    activeStage,
    onStageClick,
    onSwitchToWizard,
    showLeftNav = true,
    headerActions,
    rightPanel,
    children,
}: DashboardLayoutProps) {
    const gridClass = showLeftNav
        ? 'grid grid-cols-1 xl:grid-cols-[260px_minmax(0,1fr)_300px] gap-4 pb-20'
        : 'grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_300px] gap-4 pb-20'

    return (
        <div className={gridClass}>
            {showLeftNav && (
                <aside className="rounded-xl border border-gray-800 bg-gray-900/70 p-3 h-fit xl:sticky xl:top-4">
                    <div className="text-xs uppercase tracking-wide text-gray-500 mb-2">Pipeline Stages</div>
                    <div className="space-y-1">
                        {stages.map((stage) => (
                            <button
                                key={stage.id}
                                type="button"
                                onClick={() => onStageClick?.(stage.id)}
                                className={`w-full text-left text-sm rounded px-2 py-2 border transition-colors ${
                                    activeStage === stage.id
                                        ? 'border-brand-500/40 bg-brand-500/10 text-brand-300'
                                        : 'border-gray-700 bg-gray-900 text-gray-300 hover:bg-gray-800'
                                }`}
                            >
                                {stage.label}
                            </button>
                        ))}
                    </div>
                    <div className="mt-3 pt-3 border-t border-gray-800">
                        <button
                            type="button"
                            onClick={onSwitchToWizard}
                            className="w-full text-xs rounded border border-gray-700 bg-gray-900 px-2 py-1.5 text-gray-200 hover:bg-gray-800"
                        >
                            Switch to Wizard Mode
                        </button>
                    </div>
                </aside>
            )}

            <section className="space-y-4 min-w-0">
                <div className="sticky top-2 z-30 rounded-xl border border-gray-800 bg-gray-900/90 backdrop-blur px-4 py-3 shadow-lg">
                    <div className="flex items-center justify-between gap-3">
                        <div>
                            <h2 className="text-xl font-bold text-white">{title}</h2>
                            {subtitle && <p className="text-sm text-gray-400">{subtitle}</p>}
                        </div>
                        <div className="flex items-center gap-2">
                            {!showLeftNav && (
                                <button
                                    type="button"
                                    onClick={onSwitchToWizard}
                                    className="px-3 py-2 rounded border border-gray-700 bg-gray-900 text-gray-200 text-xs hover:bg-gray-800"
                                >
                                    Wizard Mode
                                </button>
                            )}
                            <div>{headerActions}</div>
                        </div>
                    </div>
                </div>
                <div>{children}</div>
            </section>

            <aside className="rounded-xl border border-gray-800 bg-gray-900/70 p-3 h-fit xl:sticky xl:top-4">
                {rightPanel || (
                    <div className="space-y-2">
                        <div className="text-xs uppercase tracking-wide text-gray-500">Context Help</div>
                        <p className="text-xs text-gray-400">
                            Hover-focused help panel will be added in later phases.
                        </p>
                    </div>
                )}
            </aside>
        </div>
    )
}
