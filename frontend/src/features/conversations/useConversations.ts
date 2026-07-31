import { onBeforeUnmount, shallowRef, type Ref } from 'vue'

import { ApiError } from '../../api/errors'
import { conversationsApi, type ConversationsApi } from './api'
import type {
  ConversationHistory,
  ConversationPage,
  ConversationQuery,
  QueryState,
} from './types'

function requestError(error: unknown): ApiError {
  if (error instanceof ApiError) return error
  return new ApiError({
    code: 'NETWORK_ERROR',
    message: '请求失败，请稍后重试。',
    requestId: 'unavailable',
    status: 0,
    details: null,
  })
}

function wasAborted(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

export interface ConversationListController {
  state: Ref<QueryState<ConversationPage>>
  load(query: ConversationQuery): Promise<void>
  refresh(): Promise<void>
}

export function useConversationList(
  api: ConversationsApi = conversationsApi,
): ConversationListController {
  const state = shallowRef<QueryState<ConversationPage>>({
    status: 'idle',
    data: null,
  })
  let currentQuery: ConversationQuery | null = null
  let controller: AbortController | null = null
  let requestSequence = 0

  async function load(query: ConversationQuery): Promise<void> {
    currentQuery = query
    controller?.abort()
    controller = new AbortController()
    const sequence = ++requestSequence
    const previous = state.value.data
    state.value = { status: 'loading', data: previous }

    try {
      const page = await api.list(query, controller.signal)
      if (sequence === requestSequence) {
        state.value = { status: 'success', data: page }
      }
    } catch (error) {
      if (sequence !== requestSequence || wasAborted(error)) return
      state.value = {
        status: 'error',
        data: previous,
        error: requestError(error),
      }
    }
  }

  function refresh(): Promise<void> {
    return currentQuery ? load(currentQuery) : Promise.resolve()
  }

  onBeforeUnmount(() => controller?.abort())
  return { state, load, refresh }
}

export interface ConversationHistoryController {
  state: Ref<QueryState<ConversationHistory>>
  load(id: string, page: number, pageSize: number): Promise<void>
  refresh(): Promise<void>
}

export function useConversationHistory(
  api: ConversationsApi = conversationsApi,
): ConversationHistoryController {
  const state = shallowRef<QueryState<ConversationHistory>>({
    status: 'idle',
    data: null,
  })
  let current: { id: string; page: number; pageSize: number } | null = null
  let controller: AbortController | null = null
  let requestSequence = 0

  async function load(
    id: string,
    page: number,
    pageSize: number,
  ): Promise<void> {
    current = { id, page, pageSize }
    controller?.abort()
    controller = new AbortController()
    const sequence = ++requestSequence
    const previous = state.value.data
    state.value = { status: 'loading', data: previous }

    try {
      const [conversation, messages] = await Promise.all([
        api.get(id, controller.signal),
        api.listMessages(id, page, pageSize, controller.signal),
      ])
      if (sequence === requestSequence) {
        state.value = {
          status: 'success',
          data: { conversation, messages },
        }
      }
    } catch (error) {
      if (sequence !== requestSequence || wasAborted(error)) return
      state.value = {
        status: 'error',
        data: previous,
        error: requestError(error),
      }
    }
  }

  function refresh(): Promise<void> {
    return current
      ? load(current.id, current.page, current.pageSize)
      : Promise.resolve()
  }

  onBeforeUnmount(() => controller?.abort())
  return { state, load, refresh }
}
