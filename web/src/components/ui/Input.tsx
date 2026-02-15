import React from 'react';
import { twMerge } from 'tailwind-merge';
import { Tooltip } from './Tooltip';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
    label: string;
    description?: string; // Kept for backward compatibility, but tooltip preferred
    tooltip?: string;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(({ className, label, description, tooltip, ...props }, ref) => {
    return (
        <div className="space-y-0.5">
            <div className="flex items-center mb-0.5">
                <label className="block text-xs font-semibold text-gray-300">{label}</label>
                {tooltip && <Tooltip content={tooltip} />}
            </div>
            <input
                ref={ref}
                className={twMerge(
                    "w-full bg-gray-900/50 border border-gray-700/50 rounded px-2.5 py-1.5 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed",
                    className
                )}
                {...props}
            />
            {description && <p className="text-[10px] text-gray-500 leading-tight mt-0.5">{description}</p>}
        </div>
    );
});
Input.displayName = 'Input';
