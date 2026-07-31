import { onBeforeUnmount, shallowRef } from 'vue'

import { ApiError } from '../../api/errors'
import { knowledgeApi, type KnowledgeApi } from './api'
import type {
  KnowledgeCreate,
  KnowledgeFilters,
  KnowledgeItem,
  KnowledgeMutationResult,
  KnowledgePage,
  KnowledgeQueryState,
  KnowledgeUpdate,
} from './types'

function asApiError(error: unknown): ApiError {
  if (error instanceof ApiError) return error
  return new ApiError({
    code: 'NETWORK_ERROR',
    message: '请求失败，请稍后重试。',
    requestId: 'unavailable',
    status: 0,
    details: null,
  })
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

export function useKnowledgeList(api: KnowledgeApi = knowledgeApi) {
  const state = shallowRef<KnowledgeQueryState<KnowledgePage>>({
    status: 'idle',
    data: null,
  })
  let current: KnowledgeFilters | null = null
  let controller: AbortController | null = null
  let sequence = 0

  async function load(filters: KnowledgeFilters): Promise<void> {
    current = filters
    controller?.abort()
    controller = new AbortController()
    const request = ++sequence
    const previous = state.value.data
    state.value = { status: 'loading', data: previous }
    try {
      const data = await api.list(filters, controller.signal)
      if (request === sequence) state.value = { status: 'success', data }
    } catch (error) {
      if (request !== sequence || isAbort(error)) return
      state.value = {
        status: 'error',
        data: previous,
        error: asApiError(error),
      }
    }
  }

  function refresh(): Promise<void> {
    return current ? load(current) : Promise.resolve()
  }

  onBeforeUnmount(() => controller?.abort())
  return { state, load, refresh }
}

export function useKnowledgeEditor(api: KnowledgeApi = knowledgeApi) {
  const state = shallowRef<KnowledgeQueryState<KnowledgeItem>>({
    status: 'idle',
    data: null,
  })
  const conflict = shallowRef<KnowledgeItem | null>(null)
  let controller: AbortController | null = null
  let sequence = 0

  async function load(id: string): Promise<KnowledgeItem | null> {
    controller?.abort()
    controller = new AbortController()
    const request = ++sequence
    const previous = state.value.data
    state.value = { status: 'loading', data: previous }
    try {
      const item = await api.get(id, controller.signal)
      if (request === sequence) state.value = { status: 'success', data: item }
      return request === sequence ? item : null
    } catch (error) {
      if (request !== sequence || isAbort(error)) return null
      state.value = {
        status: 'error',
        data: previous,
        error: asApiError(error),
      }
      return null
    }
  }

  async function create(
    values: KnowledgeCreate,
  ): Promise<KnowledgeMutationResult> {
    try {
      const item = await api.create(values)
      conflict.value = null
      state.value = { status: 'success', data: item }
      return { kind: 'saved', item }
    } catch (error) {
      return { kind: 'error', error: asApiError(error) }
    }
  }

  async function update(
    id: string,
    values: KnowledgeUpdate,
  ): Promise<KnowledgeMutationResult> {
    try {
      const item = await api.update(id, values)
      conflict.value = null
      state.value = { status: 'success', data: item }
      return { kind: 'saved', item }
    } catch (error) {
      const apiError = asApiError(error)
      if (apiError.code !== 'KNOWLEDGE_REVISION_CONFLICT') {
        return { kind: 'error', error: apiError }
      }
      try {
        const server = await api.get(id)
        conflict.value = server
        return { kind: 'conflict', server, error: apiError }
      } catch (reloadError) {
        return { kind: 'error', error: asApiError(reloadError) }
      }
    }
  }

  function clearConflict(): void {
    conflict.value = null
  }

  onBeforeUnmount(() => controller?.abort())
  return { state, conflict, load, create, update, clearConflict }
}

export function fieldErrorsFrom(error: ApiError): Record<string, string> {
  if (error.status !== 422 || !Array.isArray(error.details)) return {}
  const fields: Record<string, string> = {}
  for (const detail of error.details) {
    if (typeof detail !== 'object' || detail === null) continue
    const location =
      'loc' in detail && Array.isArray(detail.loc) ? detail.loc : []
    const field = location.at(-1)
    const message = 'msg' in detail ? detail.msg : null
    if (typeof field === 'string' && typeof message === 'string') {
      fields[field] = message
    }
  }
  return fields
}
