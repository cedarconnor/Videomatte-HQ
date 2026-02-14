import { useState, useRef, useEffect } from 'react'
import { FaArrowsAltH } from 'react-icons/fa'

interface WipeProps {
    leftImage: string
    rightImage: string
    leftLabel?: string
    rightLabel?: string
}

export default function WipeComparison({ leftImage, rightImage, leftLabel = "RGB", rightLabel = "Alpha" }: WipeProps) {
    const [position, setPosition] = useState(50)
    const containerRef = useRef<HTMLDivElement>(null)
    const isDragging = useRef(false)

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

    // Global mouse up to catch drag release outside container
    useEffect(() => {
        const up = () => (isDragging.current = false)
        window.addEventListener('mouseup', up)
        return () => window.removeEventListener('mouseup', up)
    }, [])

    return (
        <div
            ref={containerRef}
            className="relative w-full aspect-video bg-black select-none overflow-hidden cursor-ew-resize group"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onTouchMove={handleTouchMove}
        >
            {/* Right Image (Background - Alpha) */}
            <img
                src={rightImage}
                alt={rightLabel}
                className="absolute inset-0 w-full h-full object-contain"
                draggable={false}
            />

            {/* Left Image (Foreground - RGB) - Clipped */}
            <div
                className="absolute inset-0 overflow-hidden border-r-2 border-brand-500 bg-black"
                style={{ width: `${position}%` }}
            >
                <img
                    src={leftImage}
                    alt={leftLabel}
                    className="absolute top-0 left-0 max-w-none h-full object-contain"
                    // We need to set the width of this img to match the container width 
                    // but we can't easily do that in CSS if object-fit is contain.
                    // For now, let's assume images are 16:9 and fit the container.
                    // A better approach uses background-image or structured layout.
                    style={{ width: containerRef.current?.getBoundingClientRect().width || '100%' }}
                    draggable={false}
                />
            </div>

            {/* Handle */}
            <div
                className="absolute top-0 bottom-0 w-1 bg-brand-500 shadow-[0_0_10px_rgba(0,0,0,0.5)] flex items-center justify-center transform -translate-x-1/2 pointer-events-none"
                style={{ left: `${position}%` }}
            >
                <div className="w-8 h-8 bg-brand-500 rounded-full shadow-lg flex items-center justify-center text-white text-xs">
                    <FaArrowsAltH />
                </div>
            </div>

            {/* Labels */}
            <div className="absolute top-4 left-4 bg-black/50 text-white px-2 py-1 rounded text-xs font-mono pointer-events-none border border-white/20">
                {leftLabel}
            </div>
            <div className="absolute top-4 right-4 bg-black/50 text-white px-2 py-1 rounded text-xs font-mono pointer-events-none border border-white/20">
                {rightLabel}
            </div>
        </div>
    )
}
