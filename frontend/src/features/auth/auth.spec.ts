import { fireEvent, render, screen, waitFor } from '@testing-library/vue'
import { createMemoryHistory, createRouter } from 'vue-router'
import { describe, expect, it, vi } from 'vitest'

import { ApiError } from '../../api/errors'
import LoginPage from './LoginPage.vue'
import type { AuthApi, AuthUser } from './api'
import { authKey, createAuthController } from './useAuth'

const USER: AuthUser = {
  id: '11111111-1111-4111-8111-111111111111',
  username: 'learner',
  email: 'learner@example.com',
  role: 'user',
  status: 'active',
}

const ADMIN: AuthUser = {
  ...USER,
  id: '22222222-2222-4222-8222-222222222222',
  username: 'admin',
  role: 'admin',
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, reject, resolve }
}

function unauthorized(message = '登录状态已失效。') {
  return new ApiError({
    code: 'AUTH_REQUIRED',
    message,
    requestId: 'request-auth-401',
    status: 401,
    details: null,
  })
}

function createApi(overrides: Partial<AuthApi> = {}): AuthApi {
  return {
    getCurrentUser: vi.fn(async () => USER),
    login: vi.fn(async () => USER),
    logout: vi.fn(async () => undefined),
    ...overrides,
  }
}

describe('authentication state', () => {
  it('shares one bootstrap promise across concurrent consumers', async () => {
    const currentUser = deferred<AuthUser>()
    const api = createApi({
      getCurrentUser: vi.fn(() => currentUser.promise),
    })
    const auth = createAuthController(api)

    const first = auth.bootstrap()
    const second = auth.bootstrap()

    expect(first).toBe(second)
    expect(api.getCurrentUser).toHaveBeenCalledTimes(1)
    expect(auth.state.value).toEqual({ status: 'unknown', user: null })

    currentUser.resolve(USER)
    await expect(first).resolves.toEqual({
      status: 'authenticated',
      user: USER,
    })
    expect(auth.state.value.user).toEqual(USER)

    await auth.bootstrap()
    expect(api.getCurrentUser).toHaveBeenCalledTimes(1)
  })

  it('uses /auth/me after login instead of trusting the login response', async () => {
    const api = createApi({
      login: vi.fn(async () => ADMIN),
      getCurrentUser: vi.fn(async () => USER),
    })
    const auth = createAuthController(api)

    await expect(
      auth.login({
        username: 'learner@example.com',
        password: 'correct-password',
      }),
    ).resolves.toEqual(USER)

    expect(api.login).toHaveBeenCalledWith({
      username: 'learner@example.com',
      password: 'correct-password',
    })
    expect(api.getCurrentUser).toHaveBeenCalledTimes(1)
    expect(auth.state.value).toEqual({
      status: 'authenticated',
      user: USER,
    })
  })

  it('turns a bootstrap 401 into anonymous state', async () => {
    const api = createApi({
      getCurrentUser: vi.fn(async () => {
        throw unauthorized()
      }),
    })
    const auth = createAuthController(api)

    await expect(auth.bootstrap()).resolves.toEqual({
      status: 'anonymous',
      user: null,
    })
    expect(auth.state.value).toEqual({ status: 'anonymous', user: null })
  })

  it('clears authenticated state after logout and after a logout 401', async () => {
    const api = createApi()
    const auth = createAuthController(api)
    await auth.bootstrap()

    await auth.logout()
    expect(auth.state.value).toEqual({ status: 'anonymous', user: null })

    const expiredApi = createApi({
      logout: vi.fn(async () => {
        throw unauthorized()
      }),
    })
    const expiredAuth = createAuthController(expiredApi)
    await expiredAuth.bootstrap()

    await expect(expiredAuth.logout()).rejects.toMatchObject({ status: 401 })
    expect(expiredAuth.state.value).toEqual({
      status: 'anonymous',
      user: null,
    })
  })

  it('does not let a stale bootstrap 401 overwrite a completed login', async () => {
    const oldBootstrap = deferred<AuthUser>()
    const api = createApi({
      getCurrentUser: vi
        .fn<() => Promise<AuthUser>>()
        .mockReturnValueOnce(oldBootstrap.promise)
        .mockResolvedValueOnce(USER),
    })
    const auth = createAuthController(api)

    const bootstrap = auth.bootstrap()
    await auth.login({ username: 'learner', password: 'correct-password' })
    oldBootstrap.reject(unauthorized())

    await bootstrap
    expect(auth.state.value).toEqual({
      status: 'authenticated',
      user: USER,
    })
  })

  it('does not let a stale bootstrap restore a logged-out user', async () => {
    const oldBootstrap = deferred<AuthUser>()
    const api = createApi({
      getCurrentUser: vi.fn(() => oldBootstrap.promise),
    })
    const auth = createAuthController(api)

    const bootstrap = auth.bootstrap()
    await auth.logout()
    oldBootstrap.resolve(USER)

    await bootstrap
    expect(auth.state.value).toEqual({ status: 'anonymous', user: null })
  })
})

describe('LoginPage', () => {
  function renderLogin(
    api: AuthApi,
    initialPath = '/login?next=/conversations',
  ) {
    const auth = createAuthController(api)
    const router = createRouter({
      history: createMemoryHistory(),
      routes: [
        { path: '/login', component: LoginPage },
        {
          path: '/conversations',
          component: { template: '<main>会话记录</main>' },
        },
      ],
    })
    return router.push(initialPath).then(async () => {
      await router.isReady()
      render(LoginPage, {
        global: {
          plugins: [router],
          provide: { [authKey as symbol]: auth },
        },
      })
      return { auth, router }
    })
  }

  it('validates required fields before calling the API', async () => {
    const api = createApi()
    await renderLogin(api)

    await fireEvent.click(screen.getByRole('button', { name: '登录' }))

    expect(screen.getByText('请输入邮箱或用户名。')).toBeTruthy()
    expect(screen.getByText('请输入密码。')).toBeTruthy()
    expect(api.login).not.toHaveBeenCalled()
  })

  it('disables inputs while submitting and presents a server error', async () => {
    const loginRequest = deferred<AuthUser>()
    const api = createApi({
      login: vi.fn(() => loginRequest.promise),
    })
    await renderLogin(api)

    const username = screen.getByLabelText('邮箱或用户名')
    const password = screen.getByLabelText('密码')
    await fireEvent.update(username, 'learner@example.com')
    await fireEvent.update(password, 'wrong-password')
    await fireEvent.click(screen.getByRole('button', { name: '登录' }))

    expect(username).toHaveProperty('disabled', true)
    expect(password).toHaveProperty('disabled', true)
    expect(screen.getByRole('button', { name: '正在登录' })).toHaveProperty(
      'disabled',
      true,
    )

    loginRequest.reject(unauthorized('邮箱或密码不正确。'))
    expect((await screen.findByRole('alert')).textContent).toContain(
      '邮箱或密码不正确。',
    )
    await waitFor(() => expect(username).toHaveProperty('disabled', false))
  })

  it('shows a recoverable message when session bootstrap fails', async () => {
    await renderLogin(createApi(), '/login?auth_error=1&next=/chat')

    expect(screen.getByRole('alert').textContent).toContain(
      '暂时无法确认登录状态，请重新登录。',
    )
    expect(screen.getByRole('button', { name: '登录' })).toBeTruthy()
  })
})
