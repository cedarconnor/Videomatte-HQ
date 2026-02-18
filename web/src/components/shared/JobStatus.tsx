import JobProgress from '../JobProgress'

interface JobStatusProps {
    jobId: string | null
}

export default function JobStatus({ jobId }: JobStatusProps) {
    return <JobProgress jobId={jobId} />
}

