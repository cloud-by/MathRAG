import type { IngestionJob } from '../jobs/types'

export type DocumentStatus =
  'pending' | 'processing' | 'ready' | 'failed' | 'archived'

export interface KnowledgeDocument {
  id: string
  owner_id: string | null
  original_name: string
  mime_type: string
  size_bytes: number
  sha256: string
  status: DocumentStatus
  created_at: string
  updated_at: string
}

export interface DocumentPage {
  items: KnowledgeDocument[]
  page: number
  page_size: number
  total: number
}

export interface DocumentAccepted {
  document: KnowledgeDocument
  job: IngestionJob
}

export interface DocumentFilters {
  status?: DocumentStatus
  page: number
  pageSize: number
}
