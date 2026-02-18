import type { ReactNode } from 'react'

interface SettingRowProps {
    label: string
    help?: string
    children: ReactNode
}

export default function SettingRow({ label, help, children }: SettingRowProps) {
    return (
        <div className="space-y-1">
            <div className="text-xs font-semibold text-gray-300">
                {label}
                {help ? <span className="ml-2 text-gray-500 font-normal">{help}</span> : null}
            </div>
            {children}
        </div>
    )
}

