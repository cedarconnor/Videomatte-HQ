import React from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { Tooltip } from './Tooltip';

interface SwitchProps {
    label: string;
    checked: boolean;
    onChange: (checked: boolean) => void;
    description?: string;
    tooltip?: string;
    className?: string;
}

export const Switch: React.FC<SwitchProps> = ({ label, checked, onChange, description, tooltip, className }) => {
    return (
        <div className={twMerge("flex items-center justify-between py-0.5 group", className)}>
            <div className="space-y-0 max-w-[85%] flex items-center">
                <div className="flex flex-col">
                    <label
                        className="text-xs font-semibold text-gray-300 cursor-pointer select-none"
                        onClick={() => onChange(!checked)}
                    >
                        {label}
                    </label>
                    {description && <p className="text-[10px] text-gray-500 leading-tight pointer-events-none select-none">{description}</p>}
                </div>
                {tooltip && <Tooltip content={tooltip} />}
            </div>
            <button
                type="button"
                role="switch"
                aria-checked={checked}
                onClick={() => onChange(!checked)}
                className={clsx(
                    "relative inline-flex h-5 w-9 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-brand-500 focus:ring-offset-1 focus:ring-offset-gray-900",
                    checked ? "bg-brand-600" : "bg-gray-700"
                )}
            >
                <span className="sr-only">Use setting</span>
                <span
                    aria-hidden="true"
                    className={clsx(
                        "pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out",
                        checked ? "translate-x-4" : "translate-x-0"
                    )}
                />
            </button>
        </div>
    );
};
