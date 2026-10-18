export type JobStatus = 'processing' | 'done' | 'failed'
export type Verdict = 'violation' | 'no_violation' | 'requires_review' | 'requires_review'

export interface NormativeRule {
  title?: string
  document?: string
  section?: string
  clause?: string
  url?: string
}

export interface EvidenceFrame {
  timestamp_sec: number
  image_base64: string
}

export interface Violation {
  type: string
  timecode_sec: number
  description: string
  evidence_frames: EvidenceFrame[]
  rule?: NormativeRule | string
  normative_source?: NormativeRule | string
}

export interface AnalysisJob {
  job_id: string
  status: JobStatus
  verdict?: Verdict
  violations?: Violation[]
  error?: string
}

export interface CompletedScan {
  id: string
  name: string
  timestamp: Date
  elapsedSec: number
  size: number
  job: AnalysisJob
  demo: boolean
  file?: File
}
