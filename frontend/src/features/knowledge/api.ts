import { apiRequest } from '../../api/client'
import type {
  KnowledgeCreate,
  KnowledgeFilters,
  KnowledgeItem,
  KnowledgePage,
  KnowledgeUpdate,
} from './types'

export interface KnowledgeApi {
  list(filters: KnowledgeFilters, signal?: AbortSignal): Promise<KnowledgePage>
  get(id: string, signal?: AbortSignal): Promise<KnowledgeItem>
  create(values: KnowledgeCreate): Promise<KnowledgeItem>
  update(id: string, values: KnowledgeUpdate): Promise<KnowledgeItem>
  archive(id: string, revision: number): Promise<void>
}

export const knowledgeApi: KnowledgeApi = {
  list(filters, signal) {
    const params = new URLSearchParams({
      page: String(filters.page),
      page_size: String(filters.pageSize),
    })
    if (filters.status) params.set('status', filters.status)
    if (filters.visibility) params.set('visibility', filters.visibility)
    if (filters.category) params.set('category', filters.category)
    return apiRequest<KnowledgePage>(
      `/api/v1/knowledge-items?${params.toString()}`,
      { signal },
    )
  },
  get(id, signal) {
    return apiRequest<KnowledgeItem>(
      `/api/v1/knowledge-items/${encodeURIComponent(id)}`,
      { signal },
    )
  },
  create(values) {
    return apiRequest<KnowledgeItem, KnowledgeCreate>(
      '/api/v1/knowledge-items',
      { method: 'POST', body: values },
    )
  },
  update(id, values) {
    return apiRequest<KnowledgeItem, KnowledgeUpdate>(
      `/api/v1/knowledge-items/${encodeURIComponent(id)}`,
      { method: 'PATCH', body: values },
    )
  },
  archive(id, revision) {
    return apiRequest<void>(
      `/api/v1/knowledge-items/${encodeURIComponent(id)}?revision=${revision}`,
      { method: 'DELETE' },
    )
  },
}
