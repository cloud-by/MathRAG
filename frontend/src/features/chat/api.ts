import { apiRequest } from '../../api/client'
import type { ChatRequest, ChatResponse } from './types'

export interface ChatApi {
  answer(request: ChatRequest, signal: AbortSignal): Promise<ChatResponse>
}

export const chatApi: ChatApi = {
  answer(request, signal) {
    return apiRequest<ChatResponse, ChatRequest>('/api/v1/chat', {
      method: 'POST',
      body: request,
      signal,
      requestId: request.client_request_id,
    })
  },
}
