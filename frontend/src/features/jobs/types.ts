export type IngestionJobType = 'text' | 'pdf' | 'web' | 'reindex'
export type IngestionJobStatus =
  'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface IngestionJob {
  id: string
  requested_by: string | null
  document_id: string | null
  job_type: IngestionJobType
  status: IngestionJobStatus
  progress: number
  attempt_count: number
  error_code: string | null
  error_message: string | null
  started_at: string | null
  finished_at: string | null
  created_at: string
  updated_at: string
}

export interface IngestionJobPage {
  items: IngestionJob[]
  total: number
  offset: number
  limit: number
}

export interface JobFilters {
  status?: IngestionJobStatus
  jobType?: IngestionJobType
  documentId?: string
  offset: number
  limit: number
}
