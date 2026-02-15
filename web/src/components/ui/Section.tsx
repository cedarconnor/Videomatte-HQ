import React, { useState } from 'react';
import { FaChevronRight } from 'react-icons/fa';
import { clsx } from 'clsx';
import { Tooltip } from './Tooltip';

interface SectionProps {
    title: string;
    children: React.ReactNode;
    defaultOpen?: boolean;
    description?: string;
    tooltip?: string;
}

export const Section: React.FC<SectionProps> = ({ title, children, defaultOpen = false, description, tooltip }) => {
    const [isOpen, setIsOpen] = useState(defaultOpen);

    return (
        <div className="bg-gray-800/20 rounded-lg border border-gray-700/50 overflow-hidden transition-all duration-200">
            <button
                type="button"
                onClick={() => setIsOpen(!isOpen)}
                className="w-full px-3 py-2 flex items-center justify-between hover:bg-gray-800/40 transition-colors text-left group"
            >
                <div className="flex items-center gap-2">
                    <span className={clsx("text-xs text-gray-400 transition-transform duration-200", isOpen && "rotate-90")}>
                        <FaChevronRight />
                    </span>
                    <div>
                        <div className="flex items-center gap-2">
                            <h3 className="text-sm font-bold text-gray-200">{title}</h3>
                            {tooltip && <Tooltip content={tooltip} />}
                        </div>
                        {description && <p className="text-[10px] text-gray-500 leading-none mt-0.5">{description}</p>}
                    </div>
                </div>
            </button>

            <div
                className={clsx(
                    "transition-all duration-300 ease-in-out overflow-hidden bg-gray-900/10",
                    isOpen ? "max-h-[2000px] opacity-100" : "max-h-0 opacity-0"
                )}
            >
                <div className="p-3 border-t border-gray-700/30">
                    {children}
                </div>
            </div>
        </div>
    );
};
