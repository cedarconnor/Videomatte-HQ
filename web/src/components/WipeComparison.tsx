import { useState, useRef, useEffect } from 'react'
import { FaArrowsAltH } from 'react-icons/fa'

type CompositeMode = 'alpha' | 'checker' | 'white' | 'black'

interface WipeProps {
    leftImage: string
    rightImage: string
    leftLabel?: string
    rightLabel?: string
    compositeMode?: CompositeMode
}

export default function WipeComparison({ leftImage, rightImage, leftLabel = "RGB", rightLabel = "Alpha", compositeMode = 'alpha' }: WipeProps) {
    const [position, setPosition] = useState(50)
    const containerRef = useRef<HTMLDivElement>(null)
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const isDragging = useRef(false)
    const [rightLoaded, setRightLoaded] = useState(false)
    const rightImgRef = useRef<HTMLImageElement | null>(null)
    const leftImgRef = useRef<HTMLImageElement | null>(null)

    const handleMove = (clientX: number) => {
        if (!containerRef.current) return
        const rect = containerRef.current.getBoundingClientRect()
        const x = Math.max(0, Math.min(clientX - rect.left, rect.width))
        setPosition((x / rect.width) * 100)
    }

    const handleMouseDown = () => (isDragging.current = true)
    const handleMouseMove = (e: React.MouseEvent) => {
        if (isDragging.current) handleMove(e.clientX)
    }
    const handleTouchMove = (e: React.TouchEvent) => {
        handleMove(e.touches[0].clientX)
    }

    useEffect(() => {
        const up = () => (isDragging.current = false)
        window.addEventListener('mouseup', up)
        return () => window.removeEventListener('mouseup', up)
    }, [])

    // Load right image for canvas compositing
    useEffect(() => {
        if (compositeMode === 'alpha') {
            setRightLoaded(false)
            return
        }
        const img = new Image()
        img.crossOrigin = 'anonymous'
        img.onload = () => {
            rightImgRef.current = img
            setRightLoaded(true)
        }
        img.onerror = () => setRightLoaded(false)
        img.src = rightImage
    }, [rightImage, compositeMode])

    // Load left image for compositing modes that need RGB
    useEffect(() => {
        if (compositeMode === 'alpha') return
        const img = new Image()
        img.crossOrigin = 'anonymous'
        img.onload = () => { leftImgRef.current = img }
        img.src = leftImage
    }, [leftImage, compositeMode])

    // Render composite to canvas
    useEffect(() => {
        if (compositeMode === 'alpha' || !rightLoaded || !canvasRef.current || !rightImgRef.current) return
        const canvas = canvasRef.current
        const alphaImg = rightImgRef.current
        const rgbImg = leftImgRef.current

        canvas.width = alphaImg.naturalWidth
        canvas.height = alphaImg.naturalHeight
        const ctx = canvas.getContext('2d')!

        // Draw alpha to get pixel data
        ctx.drawImage(alphaImg, 0, 0)
        const alphaData = ctx.getImageData(0, 0, canvas.width, canvas.height)

        // Get RGB data if available
        let rgbData: ImageData | null = null
        if (rgbImg && (compositeMode === 'checker' || compositeMode === 'white' || compositeMode === 'black')) {
            const tmpCanvas = document.createElement('canvas')
            tmpCanvas.width = canvas.width
            tmpCanvas.height = canvas.height
            const tmpCtx = tmpCanvas.getContext('2d')!
            tmpCtx.drawImage(rgbImg, 0, 0, canvas.width, canvas.height)
            rgbData = tmpCtx.getImageData(0, 0, canvas.width, canvas.height)
        }

        const out = ctx.createImageData(canvas.width, canvas.height)
        const checkerSize = 16

        for (let i = 0; i < alphaData.data.length; i += 4) {
            // Alpha is stored as grayscale in the alpha image (R=G=B=alpha value)
            const a = alphaData.data[i] / 255

            let bgR: number, bgG: number, bgB: number
            if (compositeMode === 'white') {
                bgR = bgG = bgB = 255
            } else if (compositeMode === 'black') {
                bgR = bgG = bgB = 0
            } else {
                // Checker
                const px = (i / 4) % canvas.width
                const py = Math.floor((i / 4) / canvas.width)
                const isLight = ((Math.floor(px / checkerSize) + Math.floor(py / checkerSize)) % 2) === 0
                bgR = bgG = bgB = isLight ? 200 : 140
            }

            // Composite: fg * alpha + bg * (1 - alpha)
            const fgR = rgbData ? rgbData.data[i] : 255
            const fgG = rgbData ? rgbData.data[i + 1] : 255
            const fgB = rgbData ? rgbData.data[i + 2] : 255

            out.data[i] = fgR * a + bgR * (1 - a)
            out.data[i + 1] = fgG * a + bgG * (1 - a)
            out.data[i + 2] = fgB * a + bgB * (1 - a)
            out.data[i + 3] = 255
        }

        ctx.putImageData(out, 0, 0)
    }, [compositeMode, rightLoaded, rightImage, leftImage])

    const useCanvas = compositeMode !== 'alpha' && rightLoaded

    const modeLabel = compositeMode === 'alpha' ? rightLabel
        : compositeMode === 'checker' ? 'Checker Composite'
        : compositeMode === 'white' ? 'White BG Composite'
        : 'Black BG Composite'

    return (
        <div
            ref={containerRef}
            className="relative w-full aspect-video bg-black select-none overflow-hidden cursor-ew-resize group"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onTouchMove={handleTouchMove}
        >
            {/* Right Image (Background) */}
            {useCanvas ? (
                <canvas
                    ref={canvasRef}
                    className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                />
            ) : (
                <img
                    src={rightImage}
                    alt={rightLabel}
                    className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                    draggable={false}
                />
            )}

            {/* Left Image (Foreground - RGB) - Clipped */}
            <img
                src={leftImage}
                alt={leftLabel}
                className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                style={{ clipPath: `polygon(0 0, ${position}% 0, ${position}% 100%, 0 100%)` }}
                draggable={false}
            />

            {/* Divider Line */}
            <div
                className="absolute top-0 bottom-0 w-0.5 bg-brand-500 shadow-[0_0_10px_rgba(0,0,0,0.5)] pointer-events-none"
                style={{ left: `${position}%` }}
            />

            {/* Handle */}
            <div
                className="absolute top-1/2 -translate-y-1/2 w-8 h-8 bg-brand-500 rounded-full shadow-lg flex items-center justify-center text-white text-xs transform -translate-x-1/2 pointer-events-none"
                style={{ left: `${position}%` }}
            >
                <FaArrowsAltH />
            </div>

            {/* Labels */}
            <div className="absolute top-4 left-4 bg-black/50 text-white px-2 py-1 rounded text-xs font-mono pointer-events-none border border-white/20">
                {leftLabel}
            </div>
            <div className="absolute top-4 right-4 bg-black/50 text-white px-2 py-1 rounded text-xs font-mono pointer-events-none border border-white/20">
                {modeLabel}
            </div>
        </div>
    )
}
