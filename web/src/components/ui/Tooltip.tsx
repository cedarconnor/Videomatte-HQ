import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { FaQuestionCircle } from 'react-icons/fa';


interface TooltipProps {
    content: string;
}

export const Tooltip: React.FC<TooltipProps> = ({ content }) => {
    const [isVisible, setIsVisible] = useState(false);
    const [position, setPosition] = useState({ top: 0, left: 0 });
    const triggerRef = useRef<HTMLDivElement>(null);

    const updatePosition = () => {
        if (triggerRef.current) {
            const rect = triggerRef.current.getBoundingClientRect();
            setPosition({
                top: rect.top - 8, // 8px buffer above
                left: rect.left + (rect.width / 2),
            });
        }
    };

    const handleMouseEnter = () => {
        updatePosition();
        setIsVisible(true);
        window.dispatchEvent(new CustomEvent('vmhq-help-show', { detail: { content } }));
    };

    const handleMouseLeave = () => {
        setIsVisible(false);
        window.dispatchEvent(new CustomEvent('vmhq-help-clear'));
    };

    // Update position on scroll/resize while visible
    useEffect(() => {
        if (!isVisible) return;

        window.addEventListener('scroll', updatePosition, true);
        window.addEventListener('resize', updatePosition);

        return () => {
            window.removeEventListener('scroll', updatePosition, true);
            window.removeEventListener('resize', updatePosition);
        };
    }, [isVisible]);

    return (
        <>
            <div
                ref={triggerRef}
                className="relative inline-block ml-1 cursor-help text-gray-500 hover:text-brand-400 transition-colors"
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
            >
                <FaQuestionCircle className="text-[10px]" />
            </div>

            {isVisible && createPortal(
                <div
                    className="fixed z-50 px-3 py-2 text-xs font-medium text-white bg-gray-900 rounded-md shadow-xl border border-gray-700 pointer-events-none max-w-xs -translate-x-1/2 -translate-y-full"
                    style={{
                        top: position.top,
                        left: position.left,
                    }}
                >
                    {content}
                    {/* Arrow */}
                    <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-gray-900" />
                </div>,
                document.body
            )}
        </>
    );
};
