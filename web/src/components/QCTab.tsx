import { useState } from 'react'
import WipeComparison from './WipeComparison'
import { FaRegImage } from 'react-icons/fa'

export default function QCTab() {
    // Mock data for now - in real app, we'd browse the /files directory
    const [frame, setFrame] = useState(0)
    // const [jobId, setJobId] = useState("")

    // Hardcoded paths for testing based on known structure
    // Input: TestFiles/6138680-uhd_3840_2160_24fps.mp4 -> This is a video, browser can't seek securely easily without backend support.
    // Actually, browsers can play MP4.
    // For images: out/alpha/%06d.png

    // Let's assume we have extracted frames for QC or just browse the output alpha.
    // For the wipe, we need matching RGB frames.
    // The backend doesn't currently export RGB frames from the video.
    // So we can only preview the alpha, OR the video player.

    // Implemented: Video Player for Input, Sequence Player for Alpha?
    // Or just a frame-by-frame stepper.

    const frameStr = frame.toString().padStart(6, '0')
    // We don't have extracted RGB frames, so we can't wipe against them easily unless we extract them.
    // Let's just build the UI to verify the component, assuming we have two images.
    // I'll point to the alpha for both for now (inverted? no).

    const inputUrl = `/files/out/alpha/${frameStr}.png` // Placeholder
    const outputUrl = `/files/out/alpha/${frameStr}.png`

    return (
        <div className="space-y-6">
            <div className="flex justify-between items-end border-b border-gray-800 pb-4">
                <div>
                    <h2 className="text-2xl font-bold text-white">Quality Control</h2>
                    <p className="text-gray-400 mt-1">Inspect matte quality with A/B wipe comparison.</p>
                </div>
                <div className="flex items-center gap-4 bg-gray-800 p-2 rounded-lg">
                    <label className="text-xs text-gray-400">Frame</label>
                    <input
                        type="number"
                        value={frame}
                        onChange={e => setFrame(parseInt(e.target.value) || 0)}
                        className="w-20 bg-gray-900 border border-gray-700 rounded px-2 py-1 text-right font-mono"
                    />
                </div>
            </div>

            <div className="bg-gray-800 rounded-xl overflow-hidden shadow-2xl border border-gray-700">
                <WipeComparison
                    leftImage={inputUrl}
                    rightImage={outputUrl}
                    leftLabel="Input (RGB)"
                    rightLabel="Output (Alpha)"
                />
                <div className="p-4 bg-gray-800 flex justify-center gap-4 text-sm text-gray-500">
                    <p>Drag the handle to compare.</p>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-6">
                <div className="bg-gray-800/50 p-4 rounded-xl border border-gray-700">
                    <h3 className="font-semibold text-gray-300 mb-2 flex items-center gap-2"><FaRegImage /> Input Source</h3>
                    <div className="aspect-video bg-black rounded flex items-center justify-center text-gray-600">
                        Source Video Player Placeholder
                    </div>
                </div>
                <div className="bg-gray-800/50 p-4 rounded-xl border border-gray-700">
                    <h3 className="font-semibold text-gray-300 mb-2 flex items-center gap-2"><FaRegImage /> Matte Output</h3>
                    <div className="aspect-video bg-black rounded flex items-center justify-center text-gray-600">
                        Alpha Sequence Player Placeholder
                    </div>
                </div>
            </div>
        </div>
    )
}
