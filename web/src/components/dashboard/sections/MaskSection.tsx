import type { ReactNode } from 'react'
import { Section } from '../../ui/Section'

interface MaskSectionProps {
    children: ReactNode
}

export default function MaskSection({ children }: MaskSectionProps) {
    return (
        <div id="run-step-assignment" className="scroll-mt-28">
            <Section title="Subject Mask Setup" defaultOpen={true} tooltip="Create subject masks from your video. Importing external masks is optional.">
                {children}
            </Section>
        </div>
    )
}

