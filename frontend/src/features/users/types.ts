import type { components } from '../../api/schema'

export type ManagedUser = components['schemas']['ManagedUserRead']
export type UserPage = components['schemas']['UserPage']
export type UserCreate = components['schemas']['UserCreate']
export type UserUpdate = components['schemas']['UserUpdate']
export type UserPasswordReset = components['schemas']['UserPasswordReset']
export type UserRole = ManagedUser['role']
export type UserStatus = ManagedUser['status']

export interface UserFilters {
  query?: string
  role?: UserRole
  status?: UserStatus
  page: number
  pageSize: number
}
