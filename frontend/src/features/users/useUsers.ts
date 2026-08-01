import { onBeforeUnmount, readonly, shallowRef } from 'vue'

import { ApiError } from '../../api/errors'
import { usersApi, type UsersApi } from './api'
import type { UserFilters, UserPage } from './types'

export type UserListState =
  | { status: 'idle'; data: null; error: null }
  | { status: 'loading'; data: UserPage | null; error: null }
  | { status: 'ready'; data: UserPage; error: null }
  | { status: 'error'; data: UserPage | null; error: ApiError }

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
  return (
    typeof error === 'object' &&
    error !== null &&
    'name' in error &&
    error.name === 'AbortError'
  )
}

export function createUserListController(api: UsersApi = usersApi) {
  const state = shallowRef<UserListState>({
    status: 'idle',
    data: null,
    error: null,
  })
  let currentFilters: UserFilters | null = null
  let controller: AbortController | null = null
  let sequence = 0

  async function load(filters: UserFilters): Promise<void> {
    currentFilters = filters
    controller?.abort()
    controller = new AbortController()
    const request = ++sequence
    const previous = state.value.data
    state.value = { status: 'loading', data: previous, error: null }
    try {
      const data = await api.list(filters, controller.signal)
      if (request === sequence) {
        state.value = { status: 'ready', data, error: null }
      }
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
    return currentFilters ? load(currentFilters) : Promise.resolve()
  }

  function dispose(): void {
    controller?.abort()
  }

  return { state: readonly(state), load, refresh, dispose }
}

export function useUserList(api: UsersApi = usersApi) {
  const controller = createUserListController(api)
  onBeforeUnmount(controller.dispose)
  return controller
}
