import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/vue'
import { createMemoryHistory, createRouter } from 'vue-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ApiError } from '../../api/errors'
import { conversationsApi } from './api'
import ConversationHistoryPage from './ConversationHistoryPage.vue'
import ConversationListPage from './ConversationListPage.vue'
import type {
  Conversation,
  ConversationPage,
  Message,
  MessagePage,
} from './types'

const CONVERSATION_ID = '11111111-1111-4111-8111-111111111111'

const CONVERSATION: Conversation = {
  id: CONVERSATION_ID,
  title: '二次函数复习',
  status: 'active',
  created_at: '2026-07-30T08:00:00Z',
  updated_at: '2026-07-31T10:30:00Z',
}

function conversationPage(
  items: Conversation[] = [CONVERSATION],
  overrides: Partial<ConversationPage> = {},
): ConversationPage {
  return {
    items,
    page: 1,
    page_size: 20,
    total: items.length,
    ...overrides,
  }
}

function message(
  id: string,
  role: Message['role'],
  content: string,
  modelMetadata: Message['model_metadata'] = {},
): Message {
  return {
    id,
    conversation_id: CONVERSATION_ID,
    role,
    content,
    status: 'completed',
    model_metadata: modelMetadata,
    created_at: '2026-07-31T10:30:00Z',
  }
}

function messagePage(items: Message[]): MessagePage {
  return { items, page: 1, page_size: 50, total: items.length }
}

function apiError(status: number, message: string): ApiError {
  return new ApiError({
    code: status === 409 ? 'CONVERSATION_CONFLICT' : 'CONVERSATION_NOT_FOUND',
    message,
    requestId: `request-${status}`,
    status,
    details: null,
  })
}

async function renderPage(
  component: typeof ConversationListPage | typeof ConversationHistoryPage,
  path: string,
) {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/chat', name: 'chat', component: { template: '<main />' } },
      {
        path: '/conversations',
        name: 'conversations',
        component: ConversationListPage,
      },
      {
        path: '/conversations/:id',
        name: 'conversation',
        component: ConversationHistoryPage,
      },
    ],
  })
  await router.push(path)
  await router.isReady()
  render(component, { global: { plugins: [router] } })
  return router
}

beforeEach(() => {
  vi.spyOn(conversationsApi, 'list').mockResolvedValue(conversationPage())
  vi.spyOn(conversationsApi, 'get').mockResolvedValue(CONVERSATION)
  vi.spyOn(conversationsApi, 'create').mockResolvedValue(CONVERSATION)
  vi.spyOn(conversationsApi, 'update').mockResolvedValue(CONVERSATION)
  vi.spyOn(conversationsApi, 'archive').mockResolvedValue(undefined)
  vi.spyOn(conversationsApi, 'listMessages').mockResolvedValue(messagePage([]))
})

afterEach(() => vi.restoreAllMocks())

describe('ConversationListPage', () => {
  it('loads active conversations and exposes complete timestamps', async () => {
    await renderPage(ConversationListPage, '/conversations')

    expect(
      await screen.findByRole('link', { name: '二次函数复习' }),
    ).toBeTruthy()
    expect(conversationsApi.list).toHaveBeenCalledWith(
      { page: 1, pageSize: 20, status: 'active' },
      expect.any(AbortSignal),
    )
    expect(screen.getByText('2026/7/31').getAttribute('title')).toBe(
      '2026-07-31T10:30:00.000Z',
    )
  })

  it('keeps status and pagination in the URL', async () => {
    vi.mocked(conversationsApi.list).mockResolvedValue(
      conversationPage([{ ...CONVERSATION, status: 'archived' }], {
        page: 2,
        total: 25,
      }),
    )
    const router = await renderPage(
      ConversationListPage,
      '/conversations?status=archived&page=2',
    )

    expect(await screen.findByText('二次函数复习')).toBeTruthy()
    expect(conversationsApi.list).toHaveBeenCalledWith(
      { page: 2, pageSize: 20, status: 'archived' },
      expect.any(AbortSignal),
    )

    await fireEvent.click(screen.getByRole('button', { name: '活跃会话' }))
    await waitFor(() =>
      expect(router.currentRoute.value.query).toEqual({
        page: '1',
        status: 'active',
      }),
    )
  })

  it('does not let an old response overwrite a new filter', async () => {
    let resolveActive: ((page: ConversationPage) => void) | undefined
    vi.mocked(conversationsApi.list)
      .mockReturnValueOnce(
        new Promise((resolve) => {
          resolveActive = resolve
        }),
      )
      .mockResolvedValueOnce(
        conversationPage([
          { ...CONVERSATION, title: '已归档会话', status: 'archived' },
        ]),
      )

    await renderPage(ConversationListPage, '/conversations')
    await fireEvent.click(screen.getByRole('button', { name: '已归档' }))
    expect(await screen.findByText('已归档会话')).toBeTruthy()

    resolveActive?.(conversationPage([{ ...CONVERSATION, title: '旧响应' }]))
    await Promise.resolve()
    expect(screen.queryByText('旧响应')).toBeNull()
  })

  it('shows an empty state and can retry a failed request', async () => {
    vi.mocked(conversationsApi.list)
      .mockRejectedValueOnce(apiError(403, '没有权限读取会话。'))
      .mockResolvedValueOnce(conversationPage([]))

    await renderPage(ConversationListPage, '/conversations')
    expect((await screen.findByRole('alert')).textContent).toContain(
      '没有权限读取会话。',
    )
    expect(screen.getByRole('alert').textContent).toContain('request-403')

    await fireEvent.click(screen.getByRole('button', { name: '重试' }))
    expect(
      await screen.findByRole('heading', { name: '还没有会话' }),
    ).toBeTruthy()
  })

  it('renames a conversation and surfaces a conflict without losing the row', async () => {
    vi.mocked(conversationsApi.update).mockRejectedValueOnce(
      apiError(409, '会话已被其他操作更新。'),
    )
    await renderPage(ConversationListPage, '/conversations')
    await screen.findByText('二次函数复习')

    await fireEvent.click(
      screen.getByRole('button', { name: '重命名“二次函数复习”' }),
    )
    const input = screen.getByRole('textbox', { name: '会话标题' })
    await fireEvent.update(input, '函数专题')
    await fireEvent.click(screen.getByRole('button', { name: '保存' }))

    expect(conversationsApi.update).toHaveBeenCalledWith(CONVERSATION_ID, {
      title: '函数专题',
    })
    expect((await screen.findByRole('alert')).textContent).toContain(
      '会话已被其他操作更新。',
    )
    expect(screen.getByText('二次函数复习')).toBeTruthy()
  })

  it('archives only after confirmation and reloads the server list', async () => {
    await renderPage(ConversationListPage, '/conversations')
    await screen.findByText('二次函数复习')

    await fireEvent.click(
      screen.getByRole('button', { name: '归档“二次函数复习”' }),
    )
    const dialog = screen.getByRole('alertdialog', { name: '归档会话' })
    expect(within(dialog).getByText('二次函数复习')).toBeTruthy()
    expect(conversationsApi.archive).not.toHaveBeenCalled()

    await fireEvent.click(
      within(dialog).getByRole('button', { name: '确认归档' }),
    )
    await waitFor(() =>
      expect(conversationsApi.archive).toHaveBeenCalledWith(CONVERSATION_ID),
    )
    await waitFor(() => expect(conversationsApi.list).toHaveBeenCalledTimes(2))
  })

  it('creates a conversation and opens its history', async () => {
    const router = await renderPage(ConversationListPage, '/conversations')
    await screen.findByText('二次函数复习')

    await fireEvent.click(screen.getByRole('button', { name: '新建会话' }))
    await waitFor(() =>
      expect(conversationsApi.create).toHaveBeenCalledWith('新对话'),
    )
    await waitFor(() =>
      expect(router.currentRoute.value.fullPath).toBe(
        `/conversations/${CONVERSATION_ID}`,
      ),
    )
  })
})

describe('ConversationHistoryPage', () => {
  it('keeps server message order and restores structured answer details', async () => {
    vi.mocked(conversationsApi.listMessages).mockResolvedValue(
      messagePage([
        message('question', 'user', '如何求解 $x^2=4$？'),
        message('answer', 'assistant', '服务端纯文本回答', {
          response: {
            answer: '由 $x^2=4$ 得到 $x=\\pm2$。',
            steps: ['两边开平方'],
            used_knowledge: ['平方根'],
          },
        }),
      ]),
    )

    await renderPage(
      ConversationHistoryPage,
      `/conversations/${CONVERSATION_ID}`,
    )
    expect(
      await screen.findByRole('heading', { name: '二次函数复习' }),
    ).toBeTruthy()

    const messages = screen.getAllByTestId('history-message')
    expect(messages).toHaveLength(2)
    expect(messages[0]?.textContent).toContain('如何求解')
    expect(messages[1]?.textContent).toContain('由')
    expect(messages[1]?.textContent).toContain('两边开平方')
    expect(messages[1]?.textContent).not.toContain('服务端纯文本回答')
    expect(screen.getByRole('button', { name: '刷新历史' })).toBeTruthy()
  })

  it('falls back to message content for invalid response metadata', async () => {
    vi.mocked(conversationsApi.listMessages).mockResolvedValue(
      messagePage([
        message('answer', 'assistant', '可读的降级回答', {
          response: { answer: 42 },
        }),
      ]),
    )
    await renderPage(
      ConversationHistoryPage,
      `/conversations/${CONVERSATION_ID}`,
    )

    expect(await screen.findByText('可读的降级回答')).toBeTruthy()
    expect(screen.queryByRole('heading', { name: '回答' })).toBeNull()
  })

  it('shows not-found errors and can refresh the history', async () => {
    vi.mocked(conversationsApi.get)
      .mockRejectedValueOnce(apiError(404, '会话不存在。'))
      .mockResolvedValueOnce(CONVERSATION)

    await renderPage(
      ConversationHistoryPage,
      `/conversations/${CONVERSATION_ID}`,
    )
    expect((await screen.findByRole('alert')).textContent).toContain(
      '会话不存在。',
    )

    await fireEvent.click(screen.getByRole('button', { name: '重试' }))
    expect(
      await screen.findByRole('heading', { name: '二次函数复习' }),
    ).toBeTruthy()
  })
})
