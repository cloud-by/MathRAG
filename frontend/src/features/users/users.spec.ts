import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/vue'
import { defineComponent, readonly, ref } from 'vue'
import { createMemoryHistory, createRouter, RouterView } from 'vue-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ApiError } from '../../api/errors'
import type { AuthUser } from '../auth/api'
import { authKey, type AuthController, type AuthState } from '../auth/useAuth'
import { usersApi, type UsersApi } from './api'
import UserEditorPage from './UserEditorPage.vue'
import UserListPage from './UserListPage.vue'
import type { ManagedUser, UserPage } from './types'
import { createUserListController } from './useUsers'

const STUDENT: ManagedUser = {
  id: '11111111-1111-4111-8111-111111111111',
  username: 'student-a',
  email: 'student-a@example.com',
  role: 'student',
  status: 'active',
  must_change_password: true,
  created_by_user_id: '22222222-2222-4222-8222-222222222222',
  created_by_username: 'teacher-a',
  created_at: '2026-08-01T08:00:00Z',
  updated_at: '2026-08-01T08:00:00Z',
}

const AUTH_BASE = {
  email: null,
  status: 'active' as const,
  must_change_password: false,
}
const TEACHER: AuthUser = {
  ...AUTH_BASE,
  id: '22222222-2222-4222-8222-222222222222',
  username: 'teacher-a',
  role: 'teacher',
}
const ADMIN: AuthUser = {
  ...AUTH_BASE,
  id: '33333333-3333-4333-8333-333333333333',
  username: 'admin',
  role: 'admin',
}
const SELF_ADMIN: ManagedUser = {
  ...STUDENT,
  id: ADMIN.id,
  username: ADMIN.username,
  email: ADMIN.email,
  role: 'admin',
  created_by_user_id: null,
  created_by_username: null,
}

function page(overrides: Partial<UserPage> = {}): UserPage {
  return {
    items: [STUDENT],
    page: 1,
    page_size: 20,
    total: 1,
    ...overrides,
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((finish, fail) => {
    resolve = finish
    reject = fail
  })
  return { promise, reject, resolve }
}

function fakeAuth(user: AuthUser): AuthController {
  const state = ref<AuthState>({ status: 'authenticated', user })
  return {
    state: readonly(state),
    bootstrap: vi.fn(async () => state.value),
    invalidate: vi.fn(),
    login: vi.fn(async () => user),
    logout: vi.fn(async () => undefined),
    changePassword: vi.fn(async () => undefined),
  }
}

async function renderUsers(user: AuthUser, path: string) {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/users', name: 'users', component: UserListPage },
      { path: '/users/new', name: 'user-new', component: UserEditorPage },
      { path: '/users/:id', name: 'user-detail', component: UserEditorPage },
    ],
  })
  await router.push(path)
  await router.isReady()
  const Host = defineComponent({
    components: { RouterView },
    template: '<RouterView />',
  })
  render(Host, {
    global: {
      plugins: [router],
      provide: { [authKey as symbol]: fakeAuth(user) },
    },
  })
  return router
}

beforeEach(() => {
  vi.spyOn(usersApi, 'list').mockResolvedValue(page())
  vi.spyOn(usersApi, 'get').mockResolvedValue(STUDENT)
  vi.spyOn(usersApi, 'create').mockResolvedValue(STUDENT)
  vi.spyOn(usersApi, 'update').mockResolvedValue(STUDENT)
  vi.spyOn(usersApi, 'resetPassword').mockResolvedValue(undefined)
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('users api', () => {
  it('sends the five user management requests with encoded paths and bodies', async () => {
    vi.mocked(usersApi.list).mockRestore()
    vi.mocked(usersApi.get).mockRestore()
    vi.mocked(usersApi.create).mockRestore()
    vi.mocked(usersApi.update).mockRestore()
    vi.mocked(usersApi.resetPassword).mockRestore()
    const fetchMock = vi.fn(async (path: string, _options: RequestInit) => {
      void _options
      if (path.endsWith('/reset-password')) {
        return new Response(null, { status: 204 })
      }
      return new Response(JSON.stringify(STUDENT), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    })
    vi.stubGlobal('fetch', fetchMock)

    await usersApi.list({
      query: 'A B',
      role: 'student',
      status: 'active',
      page: 2,
      pageSize: 10,
    })
    await usersApi.get('id/part')
    await usersApi.create({
      username: 'student-b',
      email: null,
      password: 'temporary-123',
      role: 'student',
    })
    await usersApi.update('id/part', { email: null })
    await usersApi.resetPassword('id/part', { password: 'temporary-456' })

    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      '/api/v1/users?page=2&page_size=10&q=A+B&role=student&status=active',
    )
    expect(fetchMock.mock.calls[1]?.[0]).toBe('/api/v1/users/id%2Fpart')
    expect(fetchMock.mock.calls[2]?.[0]).toBe('/api/v1/users')
    expect(fetchMock.mock.calls[2]?.[1]).toMatchObject({
      method: 'POST',
      body: JSON.stringify({
        username: 'student-b',
        email: null,
        password: 'temporary-123',
        role: 'student',
      }),
    })
    expect(fetchMock.mock.calls[3]?.[0]).toBe('/api/v1/users/id%2Fpart')
    expect(fetchMock.mock.calls[3]?.[1]).toMatchObject({
      method: 'PATCH',
      body: JSON.stringify({ email: null }),
    })
    expect(fetchMock.mock.calls[4]?.[0]).toBe(
      '/api/v1/users/id%2Fpart/reset-password',
    )
    expect(fetchMock.mock.calls[4]?.[1]).toMatchObject({
      method: 'POST',
      body: JSON.stringify({ password: 'temporary-456' }),
    })
  })
})

describe('user list controller', () => {
  it('keeps only the latest request result and aborts the previous request', async () => {
    let firstSignal: AbortSignal | undefined
    let resolveFirst!: (value: UserPage) => void
    const api: UsersApi = {
      list: vi
        .fn<
          (
            filters: Parameters<UsersApi['list']>[0],
            signal?: AbortSignal,
          ) => Promise<UserPage>
        >()
        .mockImplementationOnce((_filters, signal) => {
          firstSignal = signal
          return new Promise((resolve) => {
            resolveFirst = resolve
          })
        })
        .mockResolvedValueOnce(page({ items: [], total: 0 })),
      get: vi.fn(),
      create: vi.fn(),
      update: vi.fn(),
      resetPassword: vi.fn(),
    }
    const controller = createUserListController(api)

    const first = controller.load({ query: 'first', page: 1, pageSize: 20 })
    const second = controller.load({ query: 'second', page: 1, pageSize: 20 })
    await second
    resolveFirst(page())
    await first

    expect(firstSignal?.aborted).toBe(true)
    expect(controller.state.value).toMatchObject({
      status: 'ready',
      data: { items: [], total: 0 },
    })
  })
})

describe('user list page', () => {
  it('shows a loading state while the first page is pending', async () => {
    let resolve!: (value: UserPage) => void
    vi.mocked(usersApi.list).mockReturnValue(
      new Promise((finish) => {
        resolve = finish
      }),
    )
    await renderUsers(ADMIN, '/users')

    expect(screen.getByRole('status').textContent).toContain('正在加载用户列表')
    resolve(page())
    expect(await screen.findByText('student-a')).toBeTruthy()
  })

  it('loads administrator filters from the URL and supports pagination', async () => {
    vi.mocked(usersApi.list).mockResolvedValue(page({ page: 2, total: 41 }))
    const router = await renderUsers(
      ADMIN,
      '/users?q=student&role=student&status=active&page=2',
    )

    expect(await screen.findByText('student-a')).toBeTruthy()
    expect(usersApi.list).toHaveBeenCalledWith(
      {
        query: 'student',
        role: 'student',
        status: 'active',
        page: 2,
        pageSize: 20,
      },
      expect.any(AbortSignal),
    )
    expect(screen.getByLabelText('角色')).toHaveProperty('value', 'student')
    await fireEvent.click(screen.getByRole('button', { name: '下一页' }))
    await waitFor(() => expect(router.currentRoute.value.query.page).toBe('3'))
  })

  it('hides role filtering from teachers and renders an empty result', async () => {
    vi.mocked(usersApi.list).mockResolvedValue(page({ items: [], total: 0 }))
    await renderUsers(TEACHER, '/users?role=admin')

    expect(await screen.findByText('没有符合条件的账号')).toBeTruthy()
    expect(screen.queryByLabelText('角色')).toBeNull()
    expect(usersApi.list).toHaveBeenCalledWith(
      {
        query: undefined,
        role: undefined,
        status: undefined,
        page: 1,
        pageSize: 20,
      },
      expect.any(AbortSignal),
    )
  })

  it('applies filter controls to the URL and resets pagination', async () => {
    const router = await renderUsers(ADMIN, '/users?page=4')
    await screen.findByText('student-a')

    await fireEvent.change(screen.getByLabelText('角色'), {
      target: { value: 'teacher' },
    })
    await waitFor(() =>
      expect(router.currentRoute.value.query.role).toBe('teacher'),
    )
    await fireEvent.change(screen.getByLabelText('状态'), {
      target: { value: 'disabled' },
    })
    await waitFor(() =>
      expect(router.currentRoute.value.query.status).toBe('disabled'),
    )
    await fireEvent.update(screen.getByLabelText('搜索'), 'student')
    await fireEvent.click(screen.getByRole('button', { name: '应用筛选' }))

    await waitFor(() => {
      expect(router.currentRoute.value.query).toEqual({
        q: 'student',
        role: 'teacher',
        status: 'disabled',
        page: '1',
      })
    })
    await waitFor(() => {
      expect(usersApi.list).toHaveBeenLastCalledWith(
        {
          query: 'student',
          role: 'teacher',
          status: 'disabled',
          page: 1,
          pageSize: 20,
        },
        expect.any(AbortSignal),
      )
    })
  })

  it('renders mobile row labels for every account field', async () => {
    await renderUsers(ADMIN, '/users')
    await screen.findByText('student-a')

    for (const label of [
      '用户名',
      '邮箱',
      '角色',
      '状态',
      '创建者',
      '创建时间',
      '操作',
    ]) {
      expect(document.querySelector(`td[data-label="${label}"]`)).toBeTruthy()
    }
  })

  it('shows a structured loading error and retries the request', async () => {
    vi.mocked(usersApi.list).mockRejectedValueOnce(
      new ApiError({
        code: 'USER_LIST_FAILED',
        message: '用户列表暂时不可用。',
        requestId: 'request-users',
        status: 503,
        details: null,
      }),
    )
    await renderUsers(ADMIN, '/users')

    expect((await screen.findByRole('alert')).textContent).toContain(
      '用户列表暂时不可用。',
    )
    await fireEvent.click(screen.getByRole('button', { name: '重试' }))
    expect(await screen.findByText('student-a')).toBeTruthy()
    expect(usersApi.list).toHaveBeenCalledTimes(2)
  })
})

describe('user editor', () => {
  it('shows only a retryable error when account details cannot load', async () => {
    vi.mocked(usersApi.get).mockRejectedValue(
      new ApiError({
        code: 'USER_NOT_FOUND',
        message: '账号不存在。',
        requestId: 'request-user-detail',
        status: 404,
        details: null,
      }),
    )
    await renderUsers(ADMIN, `/users/${STUDENT.id}`)

    const alert = await screen.findByRole('alert')
    expect(alert.textContent).toContain('账号不存在。')
    expect(alert.textContent).toContain('request-user-detail')
    expect(screen.queryByRole('button', { name: '保存更改' })).toBeNull()
    expect(screen.queryByRole('button', { name: '重置密码' })).toBeNull()
    expect(screen.getByRole('button', { name: '重试' })).toBeTruthy()
  })

  it('does not let a stale detail request fill the creation form', async () => {
    const pending = deferred<ManagedUser>()
    vi.mocked(usersApi.get).mockReturnValue(pending.promise)
    const router = await renderUsers(ADMIN, `/users/${STUDENT.id}`)
    expect(screen.getByRole('status').textContent).toContain('正在加载账号信息')

    await router.push('/users/new')
    pending.resolve(STUDENT)
    await Promise.resolve()

    expect(screen.getByRole('heading', { name: '创建账号' })).toBeTruthy()
    expect(screen.getByLabelText('用户名')).toHaveProperty('value', '')
  })

  it('fixes the teacher creation role to student', async () => {
    await renderUsers(TEACHER, '/users/new')
    expect(screen.queryByLabelText('角色')).toBeNull()

    await fireEvent.update(screen.getByLabelText('用户名'), 'student-b')
    await fireEvent.update(
      screen.getByLabelText('邮箱'),
      'student-b@example.com',
    )
    await fireEvent.update(screen.getByLabelText('临时密码'), 'temporary-123')
    await fireEvent.update(
      screen.getByLabelText('确认临时密码'),
      'temporary-123',
    )
    await fireEvent.click(screen.getByRole('button', { name: '创建账号' }))

    await waitFor(() => {
      expect(usersApi.create).toHaveBeenCalledWith({
        username: 'student-b',
        email: 'student-b@example.com',
        password: 'temporary-123',
        role: 'student',
      })
    })
  })

  it('lets administrators choose all three account roles', async () => {
    await renderUsers(ADMIN, '/users/new')
    const role = screen.getByLabelText('角色')

    expect(role.querySelectorAll('option')).toHaveLength(3)
    expect(role.textContent).toContain('学生')
    expect(role.textContent).toContain('教师')
    expect(role.textContent).toContain('管理员')

    await fireEvent.update(role, 'teacher')
    await fireEvent.update(screen.getByLabelText('用户名'), 'teacher-b')
    await fireEvent.update(screen.getByLabelText('临时密码'), 'temporary-123')
    await fireEvent.update(
      screen.getByLabelText('确认临时密码'),
      'temporary-123',
    )
    await fireEvent.click(screen.getByRole('button', { name: '创建账号' }))
    await waitFor(() => {
      expect(usersApi.create).toHaveBeenCalledWith(
        expect.objectContaining({ username: 'teacher-b', role: 'teacher' }),
      )
    })
  })

  it('validates temporary password confirmation', async () => {
    await renderUsers(ADMIN, '/users/new')
    await fireEvent.update(screen.getByLabelText('用户名'), 'teacher-b')
    await fireEvent.update(screen.getByLabelText('临时密码'), 'temporary-123')
    await fireEvent.update(
      screen.getByLabelText('确认临时密码'),
      'different-123',
    )
    await fireEvent.click(screen.getByRole('button', { name: '创建账号' }))

    expect(screen.getByText('两次输入的临时密码不一致。')).toBeTruthy()
    expect(usersApi.create).not.toHaveBeenCalled()
  })

  it('requires confirmation before disabling an account', async () => {
    await renderUsers(ADMIN, `/users/${STUDENT.id}`)
    await screen.findByDisplayValue('student-a')

    await fireEvent.click(screen.getByRole('switch', { name: '账号已启用' }))

    expect(screen.getByRole('alertdialog', { name: '禁用账号' })).toBeTruthy()
    expect(usersApi.update).not.toHaveBeenCalled()

    await fireEvent.click(screen.getByRole('button', { name: '确认禁用' }))
    expect(usersApi.update).not.toHaveBeenCalled()
    await fireEvent.click(screen.getByRole('button', { name: '保存更改' }))
    await waitFor(() => {
      expect(usersApi.update).toHaveBeenCalledWith(STUDENT.id, {
        status: 'disabled',
      })
    })
  })

  it('patches only changed account fields', async () => {
    vi.mocked(usersApi.update).mockResolvedValue({
      ...STUDENT,
      username: 'student-b',
    })
    await renderUsers(ADMIN, `/users/${STUDENT.id}`)
    const username = await screen.findByDisplayValue('student-a')

    await fireEvent.update(username, 'student-b')
    await fireEvent.click(screen.getByRole('button', { name: '保存更改' }))

    await waitFor(() => {
      expect(usersApi.update).toHaveBeenCalledWith(STUDENT.id, {
        username: 'student-b',
      })
    })
  })

  it('hides role controls from teachers editing a student', async () => {
    await renderUsers(TEACHER, `/users/${STUDENT.id}`)
    await screen.findByDisplayValue('student-a')
    expect(screen.queryByLabelText('角色')).toBeNull()
  })

  it('locks role and status controls for an administrator editing self', async () => {
    vi.mocked(usersApi.get).mockResolvedValue(SELF_ADMIN)
    await renderUsers(ADMIN, `/users/${ADMIN.id}`)
    await screen.findByDisplayValue('admin')
    expect(screen.getByLabelText('角色')).toHaveProperty('disabled', true)
    expect(screen.getByRole('switch', { name: '账号已启用' })).toHaveProperty(
      'disabled',
      true,
    )
  })

  it('resets a temporary password after matching confirmation', async () => {
    await renderUsers(ADMIN, `/users/${STUDENT.id}`)
    await screen.findByDisplayValue('student-a')
    await fireEvent.click(screen.getByRole('button', { name: '重置密码' }))
    const dialog = screen.getByRole('dialog', { name: '重置临时密码' })

    await fireEvent.update(
      within(dialog).getByLabelText('重置临时密码'),
      'new-temporary-123',
    )
    await fireEvent.update(
      within(dialog).getByLabelText('确认重置临时密码'),
      'new-temporary-123',
    )
    await fireEvent.click(
      within(dialog).getByRole('button', { name: '确认重置' }),
    )

    await waitFor(() => {
      expect(usersApi.resetPassword).toHaveBeenCalledWith(STUDENT.id, {
        password: 'new-temporary-123',
      })
    })
    expect((await screen.findByRole('status')).textContent).toContain(
      '临时密码已重置。',
    )
  })

  it('does not reset a temporary password when confirmation differs', async () => {
    await renderUsers(ADMIN, `/users/${STUDENT.id}`)
    await screen.findByDisplayValue('student-a')
    await fireEvent.click(screen.getByRole('button', { name: '重置密码' }))
    const dialog = screen.getByRole('dialog', { name: '重置临时密码' })

    await fireEvent.update(
      within(dialog).getByLabelText('重置临时密码'),
      'new-temporary-123',
    )
    await fireEvent.update(
      within(dialog).getByLabelText('确认重置临时密码'),
      'different-value',
    )
    await fireEvent.click(
      within(dialog).getByRole('button', { name: '确认重置' }),
    )

    expect(within(dialog).getByText('两次输入的临时密码不一致。')).toBeTruthy()
    expect(usersApi.resetPassword).not.toHaveBeenCalled()
  })

  it('clears a reset failure after the dialog closes', async () => {
    vi.mocked(usersApi.resetPassword).mockRejectedValue(
      new ApiError({
        code: 'USER_LAST_ADMIN_PROTECTED',
        message: '不能重置该账号。',
        requestId: 'request-reset',
        status: 409,
        details: null,
      }),
    )
    await renderUsers(ADMIN, `/users/${STUDENT.id}`)
    await screen.findByDisplayValue('student-a')
    await fireEvent.click(screen.getByRole('button', { name: '重置密码' }))
    let dialog = screen.getByRole('dialog', { name: '重置临时密码' })
    await fireEvent.update(
      within(dialog).getByLabelText('重置临时密码'),
      'new-temporary-123',
    )
    await fireEvent.update(
      within(dialog).getByLabelText('确认重置临时密码'),
      'new-temporary-123',
    )
    await fireEvent.click(
      within(dialog).getByRole('button', { name: '确认重置' }),
    )

    const alert = await within(dialog).findByRole('alert')
    expect(alert.textContent).toContain('不能重置该账号。')
    expect(alert.textContent).toContain('request-reset')
    await fireEvent.click(within(dialog).getByRole('button', { name: '取消' }))
    await waitFor(() =>
      expect(screen.queryByRole('dialog', { name: '重置临时密码' })).toBeNull(),
    )

    await fireEvent.click(screen.getByRole('button', { name: '重置密码' }))
    dialog = screen.getByRole('dialog', { name: '重置临时密码' })
    expect(within(dialog).queryByRole('alert')).toBeNull()
  })

  it('shows a conflict returned by the server', async () => {
    vi.mocked(usersApi.create).mockRejectedValue(
      new ApiError({
        code: 'USER_USERNAME_CONFLICT',
        message: '用户名已存在。',
        requestId: 'request-conflict',
        status: 409,
        details: null,
      }),
    )
    await renderUsers(ADMIN, '/users/new')
    await fireEvent.update(screen.getByLabelText('用户名'), 'student-a')
    await fireEvent.update(screen.getByLabelText('临时密码'), 'temporary-123')
    await fireEvent.update(
      screen.getByLabelText('确认临时密码'),
      'temporary-123',
    )
    await fireEvent.click(screen.getByRole('button', { name: '创建账号' }))

    expect((await screen.findByRole('alert')).textContent).toContain(
      '用户名已存在。',
    )
  })
})
