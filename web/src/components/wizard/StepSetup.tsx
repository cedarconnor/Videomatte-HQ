import { Input } from '../ui/Input'
import { FaFileVideo, FaFolderOpen } from 'react-icons/fa'

interface StepSetupProps {
    input: string
    outputDir: string
    frameStart: number
    frameEnd: number
    outputDirWarning: string | null
    browseBusy: boolean
    onInputChange: (value: string) => void
    onOutputDirChange: (value: string) => void
    onFrameStartChange: (value: number) => void
    onFrameEndChange: (value: number) => void
    onBrowseInput: () => void
    onBrowseInputDir: () => void
    onBrowseOutputDir: () => void
    onNext: () => void
}

export default function StepSetup({
    input,
    outputDir,
    frameStart,
    frameEnd,
    outputDirWarning,
    browseBusy,
    onInputChange,
    onOutputDirChange,
    onFrameStartChange,
    onFrameEndChange,
    onBrowseInput,
    onBrowseInputDir,
    onBrowseOutputDir,
    onNext,
}: StepSetupProps) {
    return (
        <div className="space-y-3">
            <h3 className="text-lg font-semibold text-white">Let's start a new matte.</h3>
            <div
                className="rounded-xl border-2 border-dashed border-gray-700 bg-gray-900/50 p-5 text-center transition-colors hover:border-gray-600 hover:bg-gray-800/50"
                onDragOver={(e) => e.preventDefault()}
                onDragEnter={(e) => e.preventDefault()}
                onDragLeave={(e) => e.preventDefault()}
                onDrop={(e) => {
                    e.preventDefault()
                    // Browsers block local file path access. We can only prevent the redirect.
                }}
            >
                <FaFileVideo className="mx-auto text-2xl text-gray-500 mb-2" />
                <div className="text-sm text-gray-300">Paste Path or Use Browse</div>
                <div className="text-xs text-gray-500 mt-1">
                    Drag-and-drop is not supported for local file paths due to browser security.
                </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="space-y-1">
                    <Input
                        label="Input Video or Frame Sequence"
                        value={input}
                        onChange={e => onInputChange(e.target.value)}
                        placeholder="TestFiles\\6138680-uhd_3840_2160_24fps.mp4"
                        tooltip="Use a video path or frame pattern like frame_%05d.png."
                    />
                    <button
                        type="button"
                        onClick={onBrowseInput}
                        disabled={browseBusy}
                        className="inline-flex items-center gap-2 px-3 py-1.5 rounded border border-gray-700 bg-gray-900 text-gray-200 text-xs hover:bg-gray-800 disabled:opacity-50"
                    >
                        <FaFolderOpen />
                        Browse Video File
                    </button>
                    <button
                        type="button"
                        onClick={onBrowseInputDir}
                        disabled={browseBusy}
                        className="ml-2 inline-flex items-center gap-2 px-3 py-1.5 rounded border border-gray-700 bg-gray-900 text-gray-200 text-xs hover:bg-gray-800 disabled:opacity-50"
                    >
                        <FaFolderOpen />
                        Browse Frame Folder
                    </button>
                </div>
                <div className="space-y-1">
                    <Input
                        label="Output Folder"
                        value={outputDir}
                        onChange={e => onOutputDirChange(e.target.value)}
                        tooltip="Where alpha frames and project files will be written."
                    />
                    <button
                        type="button"
                        onClick={onBrowseOutputDir}
                        disabled={browseBusy}
                        className="inline-flex items-center gap-2 px-3 py-1.5 rounded border border-gray-700 bg-gray-900 text-gray-200 text-xs hover:bg-gray-800 disabled:opacity-50"
                    >
                        <FaFolderOpen />
                        Browse Output Folder
                    </button>
                </div>
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
