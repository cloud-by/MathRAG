import { apiRequest } from '../../api/client'
import type {
  ManagedUser,
  UserCreate,
  UserFilters,
  UserPage,
  UserPasswordReset,
  UserUpdate,
} from './types'

function collectionPath(filters: UserFilters): `/api/${string}` {
  const query = new URLSearchParams({
    page: String(filters.page),
    page_size: String(filters.pageSize),
  })
  if (filters.query) query.set('q', filters.query)
  if (filters.role) query.set('role', filters.role)
  if (filters.status) query.set('status', filters.status)
  return `/api/v1/users?${query.toString()}`
}

function userPath(id: string): `/api/${string}` {
  return `/api/v1/users/${encodeURIComponent(id)}`
}

export interface UsersApi {
  list(filters: UserFilters, signal?: AbortSignal): Promise<UserPage>
  get(id: string, signal?: AbortSignal): Promise<ManagedUser>
  create(values: UserCreate): Promise<ManagedUser>
  update(id: string, values: UserUpdate): Promise<ManagedUser>
  resetPassword(id: string, values: UserPasswordReset): Promise<void>
}

export const usersApi: UsersApi = {
  list(filters, signal) {
    return apiRequest<UserPage>(collectionPath(filters), { signal })
  },
  get(id, signal) {
    return apiRequest<ManagedUser>(userPath(id), { signal })
  },
  create(values) {
    return apiRequest<ManagedUser, UserCreate>('/api/v1/users', {
      method: 'POST',
      body: values,
    })
  },
  update(id, values) {
    return apiRequest<ManagedUser, UserUpdate>(userPath(id), {
      method: 'PATCH',
      body: values,
    })
  },
  resetPassword(id, values) {
    return apiRequest<void, UserPasswordReset>(
      `${userPath(id)}/reset-password`,
      { method: 'POST', body: values },
    )
  },
}
