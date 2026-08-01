import { readonly, ref } from 'vue'
import { createMemoryHistory } from 'vue-router'
import { describe, expect, it, vi } from 'vitest'

import type { AuthController, AuthState } from '../features/auth/useAuth'
import type { AuthUser } from '../features/auth/api'
import { ApiError } from '../api/errors'
import type { UnauthorizedEvent } from '../api/client'
import {
  createAppRouter,
  safeNextPath,
  type UnauthorizedSubscriber,
} from './index'

const USER: AuthUser = {
  id: '11111111-1111-4111-8111-111111111111',
  username: 'learner',
  email: 'learner@example.com',
  role: 'student',
  status: 'active',
  must_change_password: false,
}

const ADMIN: AuthUser = { ...USER, role: 'admin', username: 'admin' }
const TEACHER: AuthUser = { ...USER, role: 'teacher', username: 'teacher' }

function fakeAuth(initial: AuthState) {
  const mutableState = ref<AuthState>(initial)
  const controller: AuthController = {
    state: readonly(mutableState),
    bootstrap: vi.fn(async () => mutableState.value),
    invalidate: vi.fn(() => {
      mutableState.value = { status: 'anonymous', user: null }
    }),
    login: vi.fn(async () => USER),
    logout: vi.fn(async () => undefined),
    changePassword: vi.fn(async () => undefined),
  }
  return { controller, mutableState }
}

async function navigate(initial: AuthState, path: string) {
  const auth = fakeAuth(initial)
  const router = createAppRouter({
    auth: auth.controller,
    history: createMemoryHistory(),
  })
  await router.push(path)
  await router.isReady()
  return { ...auth, router }
}

describe('application router guards', () => {
  it('redirects anonymous users to login with an internal next path', async () => {
    const { router } = await navigate(
      { status: 'anonymous', user: null },
      '/conversations/123?tab=history',
    )

    expect(router.currentRoute.value.path).toBe('/login')
    expect(router.currentRoute.value.query.next).toBe(
      '/conversations/123?tab=history',
    )
  })

  it('routes the root by authentication state', async () => {
    const anonymous = await navigate({ status: 'anonymous', user: null }, '/')
    expect(anonymous.router.currentRoute.value.path).toBe('/login')

    const authenticated = await navigate(
      { status: 'authenticated', user: USER },
      '/',
    )
    expect(authenticated.router.currentRoute.value.path).toBe('/chat')
  })

  it('forces a temporary-password user to change password', async () => {
    const temporaryUser: AuthUser = {
      ...USER,
      must_change_password: true,
    }

    const { router } = await navigate(
      { status: 'authenticated', user: temporaryUser },
      '/chat',
    )

    expect(router.currentRoute.value.path).toBe('/change-password')
  })

  it('keeps password-ready users out of the change-password route', async () => {
    const { router } = await navigate(
      { status: 'authenticated', user: USER },
      '/change-password',
    )

    expect(router.currentRoute.value.path).toBe('/chat')
  })

  it('rejects ordinary users from administrator routes', async () => {
    for (const path of ['/knowledge', '/documents', '/jobs']) {
      const { router } = await navigate(
        { status: 'authenticated', user: USER },
        path,
      )
      expect(router.currentRoute.value.path).toBe('/chat')
    }

    const administrator = await navigate(
      { status: 'authenticated', user: ADMIN },
      '/knowledge',
    )
    expect(administrator.router.currentRoute.value.path).toBe('/knowledge')
  })

  it('allows teachers and administrators to manage users', async () => {
    const student = await navigate(
      { status: 'authenticated', user: USER },
      '/users',
    )
    expect(student.router.currentRoute.value.path).toBe('/chat')

    for (const user of [TEACHER, ADMIN]) {
      for (const path of ['/users', '/users/new', `/users/${USER.id}`]) {
        const { router } = await navigate(
          { status: 'authenticated', user },
          path,
        )
        expect(router.currentRoute.value.path).toBe(path)
      }
    }
  })

  it('keeps teachers out of administrator-only tools', async () => {
    for (const path of ['/knowledge', '/documents', '/jobs']) {
      const { router } = await navigate(
        { status: 'authenticated', user: TEACHER },
        path,
      )
      expect(router.currentRoute.value.path).toBe('/chat')
    }
  })

  it('applies forced password change before user management permissions', async () => {
    const temporaryTeacher: AuthUser = {
      ...TEACHER,
      must_change_password: true,
    }
    const { router } = await navigate(
      { status: 'authenticated', user: temporaryTeacher },
      '/users',
    )

    expect(router.currentRoute.value.path).toBe('/change-password')
  })

  it('accepts only safe same-origin absolute next paths', async () => {
    expect(safeNextPath('/conversations?view=recent')).toBe(
      '/conversations?view=recent',
    )
    for (const value of [
      'https://evil.example/path',
      '//evil.example/path',
      '\\evil.example/path',
      '/%5Cevil.example/path',
      '/%2F%2Fevil.example/path',
    ]) {
      expect(safeNextPath(value)).toBeNull()
    }

    const { router } = await navigate(
      { status: 'authenticated', user: USER },
      '/login?next=https://evil.example/path',
    )
    expect(router.currentRoute.value.path).toBe('/chat')
  })

  it('waits for bootstrap before completing protected navigation', async () => {
    let finishBootstrap!: () => void
    const { controller, mutableState } = fakeAuth({
      status: 'unknown',
      user: null,
    })
    controller.bootstrap = vi.fn(
      () =>
        new Promise<AuthState>((resolve) => {
          finishBootstrap = () => {
            mutableState.value = { status: 'authenticated', user: USER }
            resolve(mutableState.value)
          }
        }),
    )
    const router = createAppRouter({
      auth: controller,
      history: createMemoryHistory(),
    })

    const navigation = router.push('/chat')
    await vi.waitFor(() => {
      expect(controller.bootstrap).toHaveBeenCalledTimes(1)
    })
    expect(router.currentRoute.value.matched).toHaveLength(0)

    finishBootstrap()
    await navigation
    expect(router.currentRoute.value.path).toBe('/chat')
  })

  it('falls back to a usable login page when bootstrap fails', async () => {
    const { controller } = fakeAuth({ status: 'unknown', user: null })
    controller.bootstrap = vi.fn(async () => {
      throw new Error('service unavailable')
    })
    const router = createAppRouter({
      auth: controller,
      history: createMemoryHistory(),
    })

    await router.push('/conversations?view=recent')

    expect(router.currentRoute.value.path).toBe('/login')
    expect(router.currentRoute.value.query).toEqual({
      auth_error: '1',
      next: '/conversations?view=recent',
    })
  })

  it('invalidates an expired session and preserves the current route as next', async () => {
    let unauthorizedListener: ((event: UnauthorizedEvent) => void) | undefined
    const subscribe: UnauthorizedSubscriber = (listener) => {
      unauthorizedListener = listener
      return () => {
        unauthorizedListener = undefined
      }
    }
    const { controller } = fakeAuth({
      status: 'authenticated',
      user: USER,
    })
    const router = createAppRouter({
      auth: controller,
      history: createMemoryHistory(),
      subscribeUnauthorized: subscribe,
    })
    await router.push('/conversations/123?tab=history')

    unauthorizedListener?.({
      path: '/api/v1/conversations/123/messages',
      error: new ApiError({
        code: 'AUTH_REQUIRED',
        message: '登录状态已失效。',
        requestId: 'request-expired',
        status: 401,
        details: null,
      }),
    })

    await vi.waitFor(() => {
      expect(router.currentRoute.value.path).toBe('/login')
    })
    expect(controller.invalidate).toHaveBeenCalledTimes(1)
    expect(router.currentRoute.value.query.next).toBe(
      '/conversations/123?tab=history',
    )
  })
})
