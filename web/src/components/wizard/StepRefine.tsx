import { Switch } from '../ui/Switch'

interface StepRefineProps {
    tightnessSliderValue: number
    softnessSliderValue: number
    despillEnabled: boolean
    rawPreviewUrl: string | null
    alphaPreviewUrl: string | null
    onTightnessChange: (value: number) => void
    onSoftnessChange: (value: number) => void
    onDespillChange: (value: boolean) => void
    onBack: () => void
    onNext: () => void
}

export default function StepRefine({
    tightnessSliderValue,
    softnessSliderValue,
    despillEnabled,
    rawPreviewUrl,
    alphaPreviewUrl,
    onTightnessChange,
    onSoftnessChange,
    onDespillChange,
    onBack,
    onNext,
}: StepRefineProps) {
    return (
        <div className="space-y-3">
            <h3 className="text-lg font-semibold text-white">Configure Global Edge Refinement</h3>
            <p className="text-xs text-gray-400">These settings apply to the final render. The preview below shows the anchor frame.</p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="rounded border border-gray-700 p-2 bg-gray-900">
                    <div className="text-xs uppercase tracking-wide text-gray-500 mb-1">Raw Video</div>
                    {rawPreviewUrl ? (
                        <img src={rawPreviewUrl} alt="Raw preview" className="w-full rounded border border-gray-800" />
                    ) : (
                        <div className="text-xs text-gray-500 py-10 text-center">Load and build a subject mask first.</div>
                    )}
                </div>
                <div className="rounded border border-gray-700 p-2 bg-gray-900">
                    <div className="text-xs uppercase tracking-wide text-gray-500 mb-1">Alpha Matte</div>
                    {alphaPreviewUrl ? (
                        <img src={alphaPreviewUrl} alt="Alpha preview" className="w-full rounded border border-gray-800" />
                    ) : (
                        <div className="text-xs text-gray-500 py-10 text-center">Alpha preview appears after building a mask.</div>
                    )}
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <label className="rounded border border-gray-700 p-3 bg-gray-900 space-y-2">
                    <div className="text-sm text-gray-200 font-semibold">Edge Tightness</div>
                    <input
                        type="range"
                        min={0}
                        max={100}
                        step={1}
                        value={tightnessSliderValue}
                        onChange={e => onTightnessChange(parseInt(e.target.value || "0", 10))}
                        className="w-full"
                    />
                    <div className="text-xs text-gray-400">Loose &lt;-&gt; Tight</div>
                </label>
                <label className="rounded border border-gray-700 p-3 bg-gray-900 space-y-2">
                    <div className="text-sm text-gray-200 font-semibold">Edge Softness</div>
                    <input
                        type="range"
                        min={0}
                        max={100}
                        step={1}
                        value={softnessSliderValue}
                        onChange={e => onSoftnessChange(parseInt(e.target.value || "0", 10))}
                        className="w-full"
                    />
                    <div className="text-xs text-gray-400">Hard &lt;-&gt; Soft</div>
                </label>
            </div>

            <Switch
                label="Enable De-Spill"
                checked={despillEnabled}
                onChange={onDespillChange}
                tooltip="Reduces background color contamination on subject edges."
            />

            <div className="flex justify-between">
                <button
                    type="button"
                    onClick={onBack}
                    className="px-4 py-2 rounded border border-gray-700 bg-gray-900 text-gray-200 hover:bg-gray-800"
                >
                    Back
                </button>
                <button
                    type="button"
                    onClick={onNext}
                    className="px-4 py-2 rounded bg-brand-500 hover:bg-brand-600 text-white font-semibold"
                >
                    Next: Render
                </button>
            </div>
        </div>
    )
}

