import { inject, readonly, shallowRef, type InjectionKey, type Ref } from 'vue'

import { ApiError } from '../../api/errors'
import {
  authApi,
  type AuthApi,
  type AuthUser,
  type ChangePasswordRequest,
  type LoginCredentials,
} from './api'

export type AuthState =
  | { status: 'unknown'; user: null }
  | { status: 'anonymous'; user: null }
  | { status: 'authenticated'; user: AuthUser }

export interface AuthController {
  readonly state: Readonly<Ref<AuthState>>
  bootstrap(): Promise<AuthState>
  invalidate(): void
  login(credentials: LoginCredentials): Promise<AuthUser>
  logout(): Promise<void>
  changePassword(values: ChangePasswordRequest): Promise<void>
}

function isUnauthorized(error: unknown): error is ApiError {
  return error instanceof ApiError && error.status === 401
}

export function createAuthController(api: AuthApi = authApi): AuthController {
  const state = shallowRef<AuthState>({ status: 'unknown', user: null })
  let bootstrapPromise: Promise<AuthState> | null = null
  let generation = 0

  function setAuthenticated(user: AuthUser): AuthState {
    const nextState: AuthState = { status: 'authenticated', user }
    state.value = nextState
    return nextState
  }

  function setAnonymous(): AuthState {
    const nextState: AuthState = { status: 'anonymous', user: null }
    state.value = nextState
    return nextState
  }

  function bootstrap(): Promise<AuthState> {
    if (state.value.status !== 'unknown') {
      return Promise.resolve(state.value)
    }
    if (bootstrapPromise === null) {
      const operation = ++generation
      const request = api
        .getCurrentUser()
        .then((user) => {
          if (operation === generation) {
            setAuthenticated(user)
          }
          return state.value
        })
        .catch((error: unknown) => {
          if (operation !== generation) {
            return state.value
          }
          if (isUnauthorized(error)) {
            return setAnonymous()
          }
          throw error
        })
      bootstrapPromise = request.finally(() => {
        bootstrapPromise = null
      })
    }
    return bootstrapPromise
  }

  function invalidate(): void {
    generation += 1
    setAnonymous()
  }

  async function login(credentials: LoginCredentials): Promise<AuthUser> {
    const operation = ++generation
    try {
      await api.login(credentials)
      const user = await api.getCurrentUser()
      if (operation === generation) {
        setAuthenticated(user)
      }
      return user
    } catch (error) {
      if (operation === generation && isUnauthorized(error)) {
        setAnonymous()
      }
      throw error
    }
  }

  async function logout(): Promise<void> {
    const operation = ++generation
    try {
      await api.logout()
      if (operation === generation) {
        setAnonymous()
      }
    } catch (error) {
      if (operation === generation && isUnauthorized(error)) {
        setAnonymous()
      }
      throw error
    }
  }

  async function changePassword(values: ChangePasswordRequest): Promise<void> {
    const operation = ++generation
    await api.changePassword(values)
    if (operation === generation) {
      invalidate()
    }
  }

  return {
    state: readonly(state),
    bootstrap,
    invalidate,
    login,
    logout,
    changePassword,
  }
}

export const authController = createAuthController()
export const authKey: InjectionKey<AuthController> = Symbol('mathrag-auth')

export function useAuth(): AuthController {
  return inject(authKey, authController)
}
