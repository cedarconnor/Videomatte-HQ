import type { ReactNode } from 'react'
import { Section } from '../../ui/Section'

interface IOSectionProps {
    children: ReactNode
}

export default function IOSection({ children }: IOSectionProps) {
    return (
        <div id="run-step-io" className="scroll-mt-28">
            <Section title="Video Input and Output" defaultOpen={true} tooltip="Set source media, output folder, frame range, and alpha format.">
                {children}
            </Section>
        </div>
    )
}

