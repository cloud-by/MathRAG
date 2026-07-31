import { apiRequest } from '../../api/client'
import type { IngestionJob, IngestionJobPage, JobFilters } from './types'

function collectionPath(filters: JobFilters): `/api/${string}` {
  const query = new URLSearchParams({
    offset: String(filters.offset),
    limit: String(filters.limit),
  })
  if (filters.status) query.set('status', filters.status)
  if (filters.jobType) query.set('job_type', filters.jobType)
  if (filters.documentId) query.set('document_id', filters.documentId)
  return `/api/v1/ingestion-jobs?${query.toString()}`
}

export interface JobsApi {
  list(filters: JobFilters, signal?: AbortSignal): Promise<IngestionJobPage>
  get(id: string): Promise<IngestionJob>
  cancel(id: string): Promise<IngestionJob>
  retry(id: string): Promise<IngestionJob>
}

export const jobsApi: JobsApi = {
  list(filters, signal) {
    return apiRequest<IngestionJobPage>(collectionPath(filters), { signal })
  },
  get(id) {
    return apiRequest<IngestionJob>(
      `/api/v1/ingestion-jobs/${encodeURIComponent(id)}`,
    )
  },
  cancel(id) {
    return apiRequest<IngestionJob>(
      `/api/v1/ingestion-jobs/${encodeURIComponent(id)}/cancel`,
      { method: 'POST' },
    )
  },
  retry(id) {
    return apiRequest<IngestionJob>(
      `/api/v1/ingestion-jobs/${encodeURIComponent(id)}/retry`,
      { method: 'POST' },
    )
  },
}
