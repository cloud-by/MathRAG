import { apiRequest } from '../../api/client'
import type { DocumentAccepted, DocumentFilters, DocumentPage } from './types'

function collectionPath(filters: DocumentFilters): `/api/${string}` {
  const query = new URLSearchParams({
    page: String(filters.page),
    page_size: String(filters.pageSize),
  })
  if (filters.status) query.set('status', filters.status)
  return `/api/v1/documents?${query.toString()}`
}

export interface DocumentsApi {
  list(filters: DocumentFilters, signal?: AbortSignal): Promise<DocumentPage>
  upload(file: File, category?: string): Promise<DocumentAccepted>
}

export const documentsApi: DocumentsApi = {
  list(filters, signal) {
    return apiRequest<DocumentPage>(collectionPath(filters), { signal })
  },
  upload(file, category) {
    const body = new FormData()
    body.append('file', file)
    if (category?.trim()) body.append('category', category.trim())
    return apiRequest<DocumentAccepted>('/api/v1/documents', {
      method: 'POST',
      body,
    })
  },
}
