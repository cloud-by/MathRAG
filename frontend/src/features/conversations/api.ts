import { apiRequest } from '../../api/client'
import type {
  Conversation,
  ConversationPage,
  ConversationQuery,
  ConversationUpdate,
  MessagePage,
} from './types'

export interface ConversationsApi {
  list(
    query: ConversationQuery,
    signal?: AbortSignal,
  ): Promise<ConversationPage>
  get(id: string, signal?: AbortSignal): Promise<Conversation>
  create(title: string): Promise<Conversation>
  update(id: string, values: ConversationUpdate): Promise<Conversation>
  archive(id: string): Promise<void>
  listMessages(
    id: string,
    page: number,
    pageSize: number,
    signal?: AbortSignal,
  ): Promise<MessagePage>
}

export const conversationsApi: ConversationsApi = {
  list(query, signal) {
    const params = new URLSearchParams({
      status: query.status,
      page: String(query.page),
      page_size: String(query.pageSize),
    })
    return apiRequest<ConversationPage>(
      `/api/v1/conversations?${params.toString()}`,
      { signal },
    )
  },
  get(id, signal) {
    return apiRequest<Conversation>(
      `/api/v1/conversations/${encodeURIComponent(id)}`,
      { signal },
    )
  },
  create(title) {
    return apiRequest<Conversation, { title: string }>(
      '/api/v1/conversations',
      {
        method: 'POST',
        body: { title },
      },
    )
  },
  update(id, values) {
    return apiRequest<Conversation, ConversationUpdate>(
      `/api/v1/conversations/${encodeURIComponent(id)}`,
      { method: 'PATCH', body: values },
    )
  },
  archive(id) {
    return apiRequest<void>(`/api/v1/conversations/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    })
  },
  listMessages(id, page, pageSize, signal) {
    const params = new URLSearchParams({
      page: String(page),
      page_size: String(pageSize),
    })
    return apiRequest<MessagePage>(
      `/api/v1/conversations/${encodeURIComponent(id)}/messages?${params.toString()}`,
      { signal },
    )
  },
}
