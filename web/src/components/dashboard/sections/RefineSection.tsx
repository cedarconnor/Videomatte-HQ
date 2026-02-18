import type { ReactNode } from 'react'
import { Section } from '../../ui/Section'

interface RefineSectionProps {
    children: ReactNode
}

export default function RefineSection({ children }: RefineSectionProps) {
    return (
        <div id="run-step-refine" className="scroll-mt-28">
            <Section title="Edge Detail Refinement" tooltip="Boundary-focused refinement using MEMatte at full-resolution tiles.">
                {children}
            </Section>
        </div>
    )
}

