import { ReactNode } from 'react'

interface WizardStep {
    id: number
    label: string
}

interface WizardLayoutProps {
    title: string
    subtitle?: string
    steps: WizardStep[]
    currentStep: number
    children: ReactNode
    onSwitchToPro: () => void
}

export default function WizardLayout({
    title,
    subtitle,
    steps,
    currentStep,
    children,
    onSwitchToPro,
}: WizardLayoutProps) {
    return (
        <div className="max-w-4xl mx-auto space-y-4 pb-20">
            <div className="flex items-center justify-between border-b border-gray-800 pb-2">
                <div>
                    <h2 className="text-xl font-bold text-white">{title}</h2>
                    {subtitle && <p className="text-sm text-gray-400">{subtitle}</p>}
                </div>
                <button
                    type="button"
                    onClick={onSwitchToPro}
                    className="px-3 py-1.5 text-xs rounded border border-gray-700 bg-gray-900 text-gray-200 hover:bg-gray-800"
                    title="Switch to full dashboard controls"
                >
                    Switch to Pro Mode
                </button>
            </div>

            <div className="rounded-xl border border-gray-800 bg-gray-900/70 p-3">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    {steps.map((step) => {
                        const active = step.id === currentStep
                        const complete = step.id < currentStep
                        return (
                            <div
                                key={step.id}
                                className={`rounded-lg border px-3 py-2 text-xs ${
                                    active
                                        ? 'border-brand-500/50 bg-brand-500/10 text-brand-300'
                                        : complete
                                            ? 'border-green-500/30 bg-green-500/5 text-green-300'
                                            : 'border-gray-700 bg-gray-900 text-gray-400'
                                }`}
                            >
                                <div className="font-semibold">Step {step.id}</div>
                                <div className="truncate">{step.label}</div>
                            </div>
                        )
                    })}
                </div>
            </div>

            <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
                {children}
            </div>
        </div>
    )
}
