import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/vue'
import { defineComponent, readonly, ref } from 'vue'
import { createMemoryHistory, createRouter } from 'vue-router'
import { describe, expect, it, vi } from 'vitest'

import type { AuthUser } from '../features/auth/api'
import {
  authKey,
  type AuthController,
  type AuthState,
} from '../features/auth/useAuth'
import ConfirmDialog from '../components/ConfirmDialog.vue'
import IconButton from '../components/IconButton.vue'
import AppShell from './AppShell.vue'

const USER: AuthUser = {
  id: '11111111-1111-4111-8111-111111111111',
  username: 'learner',
  email: 'learner@example.com',
  role: 'user',
  status: 'active',
}

const ADMIN: AuthUser = { ...USER, role: 'admin', username: 'admin' }

function fakeAuth(user: AuthUser) {
  const mutableState = ref<AuthState>({ status: 'authenticated', user })
  const controller: AuthController = {
    state: readonly(mutableState),
    bootstrap: vi.fn(async () => mutableState.value),
    invalidate: vi.fn(() => {
      mutableState.value = { status: 'anonymous', user: null }
    }),
    login: vi.fn(async () => user),
    logout: vi.fn(async () => {
      mutableState.value = { status: 'anonymous', user: null }
    }),
  }
  return controller
}

async function renderShell(user: AuthUser, path = '/chat') {
  const auth = fakeAuth(user)
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      {
        path: '/chat',
        component: { template: '<main>聊天内容</main>' },
        meta: { requiresAuth: true, title: '新建问答' },
      },
      {
        path: '/conversations',
        component: { template: '<main>会话记录</main>' },
        meta: { requiresAuth: true, title: '会话记录' },
      },
      {
        path: '/knowledge',
        component: { template: '<main>知识库</main>' },
        meta: { requiresAdmin: true, requiresAuth: true, title: '知识库' },
      },
      { path: '/login', component: { template: '<main>登录</main>' } },
    ],
  })
  await router.push(path)
  await router.isReady()
  render(AppShell, {
    global: {
      plugins: [router],
      provide: { [authKey as symbol]: auth },
    },
  })
  return { auth, router }
}

describe('AppShell', () => {
  it('shows the current student navigation without administrator commands', async () => {
    await renderShell(USER)

    expect(screen.getByRole('heading', { name: '新建问答' })).toBeTruthy()
    expect(screen.getAllByRole('link', { name: '新建问答' })).not.toHaveLength(
      0,
    )
    expect(screen.getAllByRole('link', { name: '会话记录' })).not.toHaveLength(
      0,
    )
    expect(screen.queryByRole('link', { name: '知识库' })).toBeNull()
    expect(screen.queryByRole('link', { name: '文档管理' })).toBeNull()
    expect(screen.queryByRole('link', { name: '摄取任务' })).toBeNull()
    expect(
      screen
        .getAllByRole('link', { name: '新建问答' })
        .some((link) => link.getAttribute('aria-current') === 'page'),
    ).toBe(true)
  })

  it('shows administrator navigation only to administrators', async () => {
    await renderShell(ADMIN, '/knowledge')

    expect(screen.getAllByRole('link', { name: '知识库' })).not.toHaveLength(0)
    expect(screen.getAllByRole('link', { name: '文档管理' })).not.toHaveLength(
      0,
    )
    expect(screen.getAllByRole('link', { name: '摄取任务' })).not.toHaveLength(
      0,
    )
  })

  it('closes the mobile drawer with Escape and restores trigger focus', async () => {
    await renderShell(USER)
    const trigger = screen.getByRole('button', { name: '打开导航' })

    await fireEvent.click(trigger)
    const drawer = screen.getByRole('dialog', { name: '主导航' })
    expect(drawer).toBeTruthy()

    const close = within(drawer).getByRole('button', { name: '关闭导航' })
    const links = within(drawer).getAllByRole('link')
    close.focus()
    await fireEvent.keyDown(drawer, { key: 'Tab', shiftKey: true })
    expect(document.activeElement).toBe(links.at(-1))
    await fireEvent.keyDown(drawer, { key: 'Tab' })
    expect(document.activeElement).toBe(close)

    await fireEvent.keyDown(drawer, { key: 'Escape' })

    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    expect(document.activeElement).toBe(trigger)
  })

  it('exposes logout from the user menu', async () => {
    const { auth, router } = await renderShell(USER)

    await fireEvent.click(
      screen.getByRole('button', { name: 'learner，打开用户菜单' }),
    )
    await fireEvent.click(screen.getByRole('menuitem', { name: '退出登录' }))

    await waitFor(() => expect(auth.logout).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(router.currentRoute.value.path).toBe('/login'))
  })
})

describe('shared controls', () => {
  it('gives icon-only buttons an accessible name and native tooltip', () => {
    render(IconButton, {
      props: { label: '刷新' },
      slots: { default: '<span aria-hidden="true">R</span>' },
    })

    const button = screen.getByRole('button', { name: '刷新' })
    expect(button.getAttribute('title')).toBe('刷新')
    expect(button.classList.contains('icon-button')).toBe(true)
  })

  it('traps focus and returns it after cancelling a destructive dialog', async () => {
    const Host = defineComponent({
      components: { ConfirmDialog },
      setup() {
        const open = ref(false)
        return { open }
      },
      template: `
        <button type="button" @click="open = true">删除条目</button>
        <ConfirmDialog
          :open="open"
          title="删除知识条目"
          object-name="二次函数判别式"
          confirm-label="确认删除"
          danger
          @cancel="open = false"
          @confirm="open = false"
        />
      `,
    })
    render(Host)
    const trigger = screen.getByRole('button', { name: '删除条目' })

    trigger.focus()
    await fireEvent.click(trigger)
    const dialog = screen.getByRole('alertdialog', { name: '删除知识条目' })
    expect(dialog.textContent).toContain('二次函数判别式')

    const cancel = screen.getByRole('button', { name: '取消' })
    const confirm = screen.getByRole('button', { name: '确认删除' })
    cancel.focus()
    await fireEvent.keyDown(dialog, { key: 'Tab', shiftKey: true })
    expect(document.activeElement).toBe(confirm)
    await fireEvent.keyDown(dialog, { key: 'Tab' })
    expect(document.activeElement).toBe(cancel)

    await fireEvent.keyDown(dialog, { key: 'Escape' })
    await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull())
    await waitFor(() => expect(document.activeElement).toBe(trigger))
  })
})
