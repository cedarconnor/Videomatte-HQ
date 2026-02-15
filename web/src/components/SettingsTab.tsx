import { useState, useEffect } from 'react'
import { Section } from './ui/Section'
import { Input } from './ui/Input'
import { Select } from './ui/Select'
import { Switch } from './ui/Switch'
import { FaSave, FaUndo } from 'react-icons/fa'

interface UserDefaults {
    outputDir: string
    device: string
    precision: string
}

interface UiPreferences {
    showAdvanced: boolean
}

const DEFAULT_USER_DEFAULTS: UserDefaults = {
    outputDir: "output",
    device: "cuda",
    precision: "fp16"
}

const DEFAULT_UI_PREFS: UiPreferences = {
    showAdvanced: true
}

export default function SettingsTab() {
    const [defaults, setDefaults] = useState<UserDefaults>(DEFAULT_USER_DEFAULTS)
    const [uiPrefs, setUiPrefs] = useState<UiPreferences>(DEFAULT_UI_PREFS)
    const [saved, setSaved] = useState(false)

    // Load from LocalStorage on mount
    useEffect(() => {
        const storedDefaults = localStorage.getItem('videomatte_defaults')
        if (storedDefaults) {
            try {
                setDefaults({ ...DEFAULT_USER_DEFAULTS, ...JSON.parse(storedDefaults) })
            } catch (e) {
                console.error("Failed to parse defaults", e)
            }
        }

        const storedPrefs = localStorage.getItem('videomatte_ui_prefs')
        if (storedPrefs) {
            try {
                setUiPrefs({ ...DEFAULT_UI_PREFS, ...JSON.parse(storedPrefs) })
            } catch (e) {
                console.error("Failed to parse prefs", e)
            }
        }
    }, [])

    const handleSave = () => {
        localStorage.setItem('videomatte_defaults', JSON.stringify(defaults))
        localStorage.setItem('videomatte_ui_prefs', JSON.stringify(uiPrefs))
        setSaved(true)
        setTimeout(() => setSaved(false), 2000)
    }

    const handleReset = () => {
        if (confirm("Reset all settings to default?")) {
            setDefaults(DEFAULT_USER_DEFAULTS)
            setUiPrefs(DEFAULT_UI_PREFS)
            localStorage.removeItem('videomatte_defaults')
            localStorage.removeItem('videomatte_ui_prefs')
        }
    }

    return (
        <div className="space-y-4 pb-20">
            <div className="border-b border-gray-800 pb-2 flex justify-between items-end">
                <div>
                    <h2 className="text-xl font-bold text-white">Settings</h2>
                    <p className="text-sm text-gray-400">Configure global application defaults and UI preferences.</p>
                </div>
                {saved && <span className="text-green-500 font-bold animate-pulse text-sm">Settings Saved!</span>}
            </div>

            <div className="space-y-2">
                {/* 1. Job Defaults */}
                <Section title="Job Defaults" defaultOpen={true} tooltip="Default values for new jobs.">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2">
                        <Input
                            label="Default Output Directory"
                            value={defaults.outputDir}
                            onChange={e => setDefaults({ ...defaults, outputDir: e.target.value })}
                            tooltip="Default path for job outputs."
                        />
                        <Select
                            label="Default Device"
                            value={defaults.device}
                            onChange={e => setDefaults({ ...defaults, device: e.target.value })}
                            options={[
                                { value: 'cuda', label: 'CUDA (NVIDIA)' },
                                { value: 'cpu', label: 'CPU' }
                            ]}
                            tooltip="Preferred hardware acceleration."
                        />
                        <Select
                            label="Default Precision"
                            value={defaults.precision}
                            onChange={e => setDefaults({ ...defaults, precision: e.target.value })}
                            options={[
                                { value: 'fp16', label: 'FP16 (Half)' },
                                { value: 'fp32', label: 'FP32 (Full)' }
                            ]}
                            tooltip="Preferred floating point precision."
                        />
                    </div>
                </Section>

                {/* 2. UI Preferences */}
                <Section title="UI Preferences" defaultOpen={true} tooltip="Customize the interface appearance.">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2">
                        <Switch
                            label="Show Advanced Options"
                            tooltip="Show/Hide advanced configuration sections in the Run Tab."
                            checked={uiPrefs.showAdvanced}
                            onChange={v => setUiPrefs({ ...uiPrefs, showAdvanced: v })}
                        />
                    </div>
                </Section>

                {/* 3. System Info */}
                <Section title="System Information" tooltip="Version and environment details.">
                    <div className="space-y-1 text-xs text-gray-400">
                        <div className="flex justify-between border-b border-gray-700/50 pb-1">
                            <span>Client Version</span>
                            <span className="text-white font-mono">v0.1.0</span>
                        </div>
                        <div className="flex justify-between border-b border-gray-700/50 pb-1 pt-1">
                            <span>API Status</span>
                            <span className="text-green-500 font-mono">Connected</span>
                            {/* In a real app we'd check this */}
                        </div>
                    </div>
                </Section>

                <div className="pt-4 sticky bottom-6 z-10 flex gap-4">
                    <button
                        onClick={handleSave}
                        className="flex-1 py-3 bg-brand-500 hover:bg-brand-600 text-white rounded-lg font-bold shadow-lg shadow-brand-500/20 flex items-center justify-center gap-2 transition-colors text-sm"
                    >
                        <FaSave /> Save Settings
                    </button>
                    <button
                        onClick={handleReset}
                        className="px-6 py-3 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-lg font-bold shadow-lg flex items-center justify-center gap-2 transition-colors text-sm"
                    >
                        <FaUndo /> Reset
                    </button>
                </div>
            </div>
        </div>
    )
}
