import { useState } from 'react'
import { FaPlay, FaSpinner, FaExclamationCircle } from 'react-icons/fa'

// API Types (mirrors backend)
interface Config {
    io: { input: string; output_alpha: string; alpha_format: string; shot_type: string }
    globals: { model: string; chunk_len: number; chunk_overlap: number }
    runtime: { device: string; precision: string }
}

export default function RunTab({ onSuccess }: { onSuccess: () => void }) {
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    // Default Config
    const [config, setConfig] = useState<Config>({
        io: { input: "", output_alpha: "out/alpha/%06d.png", alpha_format: "png16", shot_type: "locked_off" },
        globals: { model: "rvm", chunk_len: 24, chunk_overlap: 6 },
        runtime: { device: "cuda", precision: "fp16" }
    })

    async function handleSubmit(e: React.FormEvent) {
        e.preventDefault()
        setLoading(true)
        setError(null)

        try {
            const res = await fetch('/api/jobs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config })
            })

            if (!res.ok) throw new Error(await res.text())

            onSuccess()
        } catch (err: any) {
            setError(err.message)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="space-y-6">
            <div className="border-b border-gray-800 pb-4">
                <h2 className="text-2xl font-bold text-white">New Matting Job</h2>
                <p className="text-gray-400 mt-1">Configure and start a new video matting pipeline run.</p>
            </div>

            {error && (
                <div className="bg-red-500/10 border border-red-500/20 text-red-500 p-4 rounded-lg flex items-center gap-3">
                    <FaExclamationCircle />
                    {error}
                </div>
            )}

            <form onSubmit={handleSubmit} className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Access Inputs with nested state updater logic would go here. 
            For brevity in this artifact, I'm simplifying the form fields handling.
        */}

                {/* IO Section */}
                <Section title="Input / Output">
                    <Input label="Input File" value={config.io.input} onChange={(v: string) => setConfig({ ...config, io: { ...config.io, input: v } })} placeholder="Path to video or image sequence" />
                    <Input label="Output Pattern" value={config.io.output_alpha} onChange={(v: string) => setConfig({ ...config, io: { ...config.io, output_alpha: v } })} />
                    <Select label="Alpha Format" value={config.io.alpha_format} onChange={(v: string) => setConfig({ ...config, io: { ...config.io, alpha_format: v } })} options={['png16', 'exr_dwaa', 'exr_dwaa_hq', 'exr_lossless', 'exr_raw']} />
                </Section>

                {/* Model Section */}
                <Section title="Model Configuration">
                    <Select label="Global Model" value={config.globals.model} onChange={(v: string) => setConfig({ ...config, globals: { ...config.globals, model: v } })} options={['rvm']} />
                    <Select label="Shot Type" value={config.io.shot_type} onChange={(v: string) => setConfig({ ...config, io: { ...config.io, shot_type: v } })} options={['locked_off', 'handheld']} />
                </Section>

                {/* Runtime Section */}
                <Section title="Runtime">
                    <Select label="Device" value={config.runtime.device} onChange={(v: string) => setConfig({ ...config, runtime: { ...config.runtime, device: v } })} options={['cuda', 'cpu']} />
                    <Select label="Precision" value={config.runtime.precision} onChange={(v: string) => setConfig({ ...config, runtime: { ...config.runtime, precision: v } })} options={['fp16', 'fp32']} />
                </Section>

                <div className="md:col-span-2 pt-6">
                    <button
                        type="submit"
                        disabled={loading || !config.io.input}
                        className="w-full py-4 bg-brand-500 hover:bg-brand-600 text-white rounded-lg font-bold text-lg shadow-lg shadow-brand-500/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3 transition-colors"
                    >
                        {loading ? <FaSpinner className="animate-spin" /> : <FaPlay />}
                        Start Pipeline
                    </button>
                </div>
            </form >
        </div >
    )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
    return (
        <div className="bg-gray-800/30 p-6 rounded-xl border border-gray-700 space-y-4">
            <h3 className="text-lg font-semibold text-gray-200 mb-4">{title}</h3>
            {children}
        </div>
    )
}

interface InputProps {
    label: string
    value: string
    onChange: (val: string) => void
    placeholder?: string
}

function Input({ label, value, onChange, placeholder }: InputProps) {
    return (
        <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">{label}</label>
            <input
                type="text"
                value={value}
                onChange={e => onChange(e.target.value)}
                placeholder={placeholder}
                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500 transition-colors"
            />
        </div>
    )
}

interface SelectProps {
    label: string
    value: string
    onChange: (val: string) => void
    options: string[]
}

function Select({ label, value, onChange, options }: SelectProps) {
    return (
        <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">{label}</label>
            <select
                value={value}
                onChange={e => onChange(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500 transition-colors"
            >
                {options.map((opt: string) => <option key={opt} value={opt}>{opt}</option>)}
            </select>
        </div>
    )
}
