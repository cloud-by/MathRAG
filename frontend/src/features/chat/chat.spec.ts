import { fireEvent, render, screen, waitFor } from '@testing-library/vue'
import { createMemoryHistory, createRouter } from 'vue-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ApiError } from '../../api/errors'
import { conversationsApi } from '../conversations/api'
import type { Conversation, MessagePage } from '../conversations/types'
import { chatApi, type ChatApi } from './api'
import ChatComposer from './ChatComposer.vue'
import ChatPage from './ChatPage.vue'
import type { ChatResponse } from './types'
import { createChatController } from './useChat'

const CONVERSATION_ID = '11111111-1111-4111-8111-111111111111'

const CONVERSATION: Conversation = {
  id: CONVERSATION_ID,
  title: '二次函数',
  status: 'active',
  created_at: '2026-07-31T10:00:00Z',
  updated_at: '2026-07-31T10:00:00Z',
}

const EMPTY_MESSAGES: MessagePage = {
  items: [],
  page: 1,
  page_size: 50,
  total: 0,
}

function response(overrides: Partial<ChatResponse> = {}): ChatResponse {
  return {
    conversation_id: CONVERSATION_ID,
    question_message_id: '22222222-2222-4222-8222-222222222222',
    answer_message_id: '33333333-3333-4333-8333-333333333333',
    rag_run_id: '44444444-4444-4444-8444-444444444444',
    client_request_id: '55555555-5555-4555-8555-555555555555',
    question: '如何求解？',
    answer: '使用求根公式。',
    references: [],
    related_questions: [],
    steps: [],
    used_knowledge: [],
    ...overrides,
  }
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

function ragError(code: string, message = '请求状态冲突。'): ApiError {
  return new ApiError({
    code,
    message,
    requestId: `request-${code}`,
    status: 409,
    details: null,
  })
}

function createApiMock(): ChatApi {
  return { answer: vi.fn(async () => response()) }
}

beforeEach(() => {
  vi.spyOn(conversationsApi, 'create').mockResolvedValue(CONVERSATION)
  vi.spyOn(conversationsApi, 'get').mockResolvedValue(CONVERSATION)
  vi.spyOn(conversationsApi, 'listMessages').mockResolvedValue(EMPTY_MESSAGES)
  vi.spyOn(chatApi, 'answer').mockResolvedValue(response())
})

afterEach(() => vi.restoreAllMocks())

describe('chat controller', () => {
  it('creates a conversation before the first answer request', async () => {
    const api = createApiMock()
    const createConversation = vi.fn(async () => CONVERSATION)
    const created = vi.fn(async () => undefined)
    const controller = createChatController(api, { create: createConversation })

    await controller.submit({
      conversationId: null,
      question: '  如何求解？  ',
      topK: 3,
      onConversationCreated: created,
    })

    expect(createConversation).toHaveBeenCalledWith('如何求解？')
    expect(created).toHaveBeenCalledWith(CONVERSATION_ID)
    expect(api.answer).toHaveBeenCalledWith(
      expect.objectContaining({
        conversation_id: CONVERSATION_ID,
        question: '如何求解？',
        top_k: 3,
      }),
      expect.any(AbortSignal),
    )
    expect(createConversation.mock.invocationCallOrder[0]).toBeLessThan(
      vi.mocked(api.answer).mock.invocationCallOrder[0]!,
    )
    expect(controller.state.value.status).toBe('success')
  })

  it('reuses an existing conversation and ignores duplicate submits', async () => {
    const pending = deferred<ChatResponse>()
    const api: ChatApi = { answer: vi.fn(() => pending.promise) }
    const createConversation = vi.fn(async () => CONVERSATION)
    const controller = createChatController(api, { create: createConversation })
    const options = {
      conversationId: CONVERSATION_ID,
      question: '判别式是什么？',
      topK: 4,
    }

    const first = controller.submit(options)
    const duplicate = controller.submit(options)
    await Promise.resolve()

    expect(createConversation).not.toHaveBeenCalled()
    expect(api.answer).toHaveBeenCalledTimes(1)
    expect(controller.state.value.status).toBe('submitting')

    pending.resolve(response({ question: options.question }))
    await Promise.all([first, duplicate])
  })

  it('cancels the active request without presenting a network error', async () => {
    const api: ChatApi = {
      answer: vi.fn((_request, signal) => {
        return new Promise<ChatResponse>((_resolve, reject) => {
          signal.addEventListener('abort', () =>
            reject(new DOMException('Aborted', 'AbortError')),
          )
        })
      }),
    }
    const controller = createChatController(api, {
      create: vi.fn(async () => CONVERSATION),
    })
    const submitting = controller.submit({
      conversationId: CONVERSATION_ID,
      question: '停止这个请求',
      topK: 3,
    })
    await waitFor(() =>
      expect(controller.state.value.status).toBe('submitting'),
    )

    controller.cancel()
    await submitting

    expect(controller.state.value.status).toBe('cancelled')
    expect(controller.pending.value).toBeNull()
  })

  it.each([
    ['transport error', new TypeError('Failed to fetch')],
    ['in-progress conflict', ragError('RAG_REQUEST_IN_PROGRESS')],
  ])(
    'retries %s with the original client request id',
    async (_label, error) => {
      const api = createApiMock()
      vi.mocked(api.answer)
        .mockRejectedValueOnce(error)
        .mockResolvedValueOnce(response())
      const controller = createChatController(api, {
        create: vi.fn(async () => CONVERSATION),
      })

      await controller.submit({
        conversationId: CONVERSATION_ID,
        question: '重试问题',
        topK: 3,
      })
      expect(controller.state.value.status).toBe('error')
      if (controller.state.value.status === 'error') {
        expect(controller.state.value.retryable).toBe(true)
      }
      const firstId = vi.mocked(api.answer).mock.calls[0]?.[0].client_request_id

      await controller.retry()
      const secondId = vi.mocked(api.answer).mock.calls[1]?.[0]
        .client_request_id
      expect(secondId).toBe(firstId)
      expect(controller.state.value.status).toBe('success')
    },
  )

  it('generates a new id after RAG_CANCELLED', async () => {
    const api = createApiMock()
    vi.mocked(api.answer)
      .mockRejectedValueOnce(ragError('RAG_CANCELLED', '请求已取消。'))
      .mockResolvedValueOnce(response())
    const controller = createChatController(api, {
      create: vi.fn(async () => CONVERSATION),
    })
    const options = {
      conversationId: CONVERSATION_ID,
      question: '重新提问',
      topK: 3,
    }

    await controller.submit(options)
    const cancelledId = vi.mocked(api.answer).mock.calls[0]?.[0]
      .client_request_id
    expect(controller.pending.value).toBeNull()

    await controller.submit(options)
    const nextId = vi.mocked(api.answer).mock.calls[1]?.[0].client_request_id
    expect(nextId).not.toBe(cancelledId)
  })

  it('does not append the same successful server message twice', async () => {
    const duplicate = response()
    const api = createApiMock()
    vi.mocked(api.answer).mockResolvedValue(duplicate)
    const controller = createChatController(api, {
      create: vi.fn(async () => CONVERSATION),
    })

    await controller.submit({
      conversationId: CONVERSATION_ID,
      question: '第一次',
      topK: 3,
    })
    await controller.submit({
      conversationId: CONVERSATION_ID,
      question: '第二次',
      topK: 3,
    })

    expect(controller.turns.value).toHaveLength(1)
  })
})

describe('chat workspace', () => {
  async function renderChat(path: string) {
    const router = createRouter({
      history: createMemoryHistory(),
      routes: [
        { path: '/chat', name: 'chat', component: ChatPage },
        {
          path: '/conversations/:id',
          name: 'conversation',
          component: ChatPage,
        },
        {
          path: '/conversations',
          name: 'conversations',
          component: { template: '<main />' },
        },
      ],
    })
    await router.push(path)
    await router.isReady()
    render(ChatPage, { global: { plugins: [router] } })
    return router
  }

  it('replaces the new-chat URL and shows the server response', async () => {
    const router = await renderChat('/chat')
    expect(
      screen.getByRole('spinbutton', { name: '检索知识条数' }),
    ).toHaveProperty('value', '3')

    await fireEvent.update(
      screen.getByRole('textbox', { name: '数学问题' }),
      '如何求解？',
    )
    await fireEvent.click(screen.getByRole('button', { name: '发送问题' }))

    await waitFor(() =>
      expect(router.currentRoute.value.path).toBe(
        `/conversations/${CONVERSATION_ID}`,
      ),
    )
    expect(await screen.findByText('使用求根公式。')).toBeTruthy()
    expect(conversationsApi.create).toHaveBeenCalledTimes(1)
    expect(chatApi.answer).toHaveBeenCalledTimes(1)
  })

  it('fills a related question without sending it', async () => {
    vi.mocked(chatApi.answer).mockResolvedValue(
      response({ related_questions: ['根与系数有什么关系？'] }),
    )
    await renderChat(`/conversations/${CONVERSATION_ID}`)
    const textarea = await screen.findByRole('textbox', { name: '数学问题' })
    await fireEvent.update(textarea, '先问一个问题')
    await fireEvent.click(screen.getByRole('button', { name: '发送问题' }))
    await fireEvent.click(
      await screen.findByRole('button', { name: '根与系数有什么关系？' }),
    )

    expect((textarea as HTMLTextAreaElement).value).toBe('根与系数有什么关系？')
    expect(chatApi.answer).toHaveBeenCalledTimes(1)
  })
})

describe('ChatComposer', () => {
  it('enforces top-k bounds and does not submit during IME composition', async () => {
    const { emitted, rerender } = render(ChatComposer, {
      props: { modelValue: '测试问题', topK: 1, state: 'idle' },
    })
    expect(screen.getByRole('button', { name: '减少检索数量' })).toHaveProperty(
      'disabled',
      true,
    )
    await fireEvent.click(screen.getByRole('button', { name: '增加检索数量' }))
    expect(emitted()['update:topK']?.[0]).toEqual([2])
    await rerender({ modelValue: '测试问题', topK: 10, state: 'idle' })
    expect(screen.getByRole('button', { name: '增加检索数量' })).toHaveProperty(
      'disabled',
      true,
    )

    const textarea = screen.getByRole('textbox', { name: '数学问题' })
    await fireEvent.compositionStart(textarea)
    await fireEvent.keyDown(textarea, { key: 'Enter' })
    expect(emitted().submit).toBeUndefined()
    await fireEvent.compositionEnd(textarea)
    await fireEvent.keyDown(textarea, { key: 'Enter' })
    expect(emitted().submit).toHaveLength(1)
  })
})
