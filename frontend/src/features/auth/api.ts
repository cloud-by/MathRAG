import { apiRequest } from '../../api/client'
import type { components } from '../../api/schema'

export type AuthUser = components['schemas']['AuthUserRead']
export type LoginCredentials = components['schemas']['LoginRequest']

export interface AuthApi {
  getCurrentUser(): Promise<AuthUser>
  login(credentials: LoginCredentials): Promise<AuthUser>
  logout(): Promise<void>
}

export const authApi: AuthApi = {
  getCurrentUser() {
    return apiRequest<AuthUser>('/api/v1/auth/me')
  },
  login(credentials) {
    return apiRequest<AuthUser, LoginCredentials>('/api/v1/auth/login', {
      method: 'POST',
      body: credentials,
    })
  },
  logout() {
    return apiRequest<void>('/api/v1/auth/logout', { method: 'POST' })
  },
}
