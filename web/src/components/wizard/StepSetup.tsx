import { Input } from '../ui/Input'
import { FaFileVideo } from 'react-icons/fa'

interface StepSetupProps {
    input: string
    outputDir: string
    frameStart: number
    frameEnd: number
    outputDirWarning: string | null
    onInputChange: (value: string) => void
    onOutputDirChange: (value: string) => void
    onFrameStartChange: (value: number) => void
    onFrameEndChange: (value: number) => void
    onNext: () => void
}

export default function StepSetup({
    input,
    outputDir,
    frameStart,
    frameEnd,
    outputDirWarning,
    onInputChange,
    onOutputDirChange,
    onFrameStartChange,
    onFrameEndChange,
    onNext,
}: StepSetupProps) {
    return (
        <div className="space-y-3">
            <h3 className="text-lg font-semibold text-white">Let's start a new matte.</h3>
            <div className="rounded-xl border-2 border-dashed border-gray-700 bg-gray-900/50 p-5 text-center">
                <FaFileVideo className="mx-auto text-2xl text-gray-500 mb-2" />
                <div className="text-sm text-gray-300">Drop Video Here</div>
                <div className="text-xs text-gray-500 mt-1">Or paste the file path below.</div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <Input
                    label="Input Video or Frame Sequence"
                    value={input}
                    onChange={e => onInputChange(e.target.value)}
                    placeholder="TestFiles\\6138680-uhd_3840_2160_24fps.mp4"
                    tooltip="Use a video path or frame pattern like frame_%05d.png."
                />
                <Input
                    label="Output Folder"
                    value={outputDir}
                    onChange={e => onOutputDirChange(e.target.value)}
                    tooltip="Where alpha frames and project files will be written."
                />
                <Input
                    label="Start Frame"
                    type="number"
                    value={frameStart}
                    onChange={e => onFrameStartChange(parseInt(e.target.value || "0", 10))}
                />
                <Input
                    label="End Frame (-1 = full clip)"
                    type="number"
                    value={frameEnd}
                    onChange={e => onFrameEndChange(parseInt(e.target.value || "-1", 10))}
                />
            </div>
            {outputDirWarning && (
                <div className="text-xs text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded px-3 py-2">
                    {outputDirWarning}
                </div>
            )}
            <div className="flex justify-end">
                <button
                    type="button"
                    disabled={!input}
                    onClick={onNext}
                    className="px-4 py-2 rounded bg-brand-500 hover:bg-brand-600 text-white font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    Next: Select Subject
                </button>
            </div>
        </div>
    )
}
