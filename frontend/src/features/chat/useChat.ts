import { onBeforeUnmount, shallowRef, type Ref } from 'vue'

import { ApiError } from '../../api/errors'
import type { ConversationsApi } from '../conversations/api'
import { conversationsApi } from '../conversations/api'
import { chatApi, type ChatApi } from './api'
import type {
  ChatRequest,
  ChatResponse,
  ChatStatus,
  ChatTurn,
  PendingTurn,
} from './types'

export type ChatState =
  | { status: 'idle' }
  | { status: 'submitting' }
  | { status: 'success'; response: ChatResponse }
  | { status: 'cancelled' }
  | { status: 'error'; error: ApiError; retryable: boolean }

export interface SubmitChatOptions {
  conversationId: string | null
  question: string
  topK: number
  onConversationCreated?: (id: string) => Promise<void> | void
}

export interface ChatController {
  readonly state: Readonly<Ref<ChatState>>
  readonly status: Readonly<Ref<ChatStatus>>
  readonly pending: Readonly<Ref<PendingTurn | null>>
  readonly turns: Readonly<Ref<ChatTurn[]>>
  submit(options: SubmitChatOptions): Promise<ChatResponse | undefined>
  retry(): Promise<ChatResponse | undefined>
  cancel(): void
  clearTurns(): void
  dispose(): void
}

type ConversationCreator = Pick<ConversationsApi, 'create'>

function normalizedError(error: unknown): ApiError {
  if (error instanceof ApiError) return error
  return new ApiError({
    code: 'NETWORK_ERROR',
    message: '网络连接失败，请检查连接后重试。',
    requestId: 'unavailable',
    status: 0,
    details: null,
  })
}

function validationError(message: string): ApiError {
  return new ApiError({
    code: 'CHAT_VALIDATION_ERROR',
    message,
    requestId: 'not-sent',
    status: 0,
    details: null,
  })
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError'
}

function canRetry(error: ApiError): boolean {
  return (
    error.code === 'RAG_REQUEST_IN_PROGRESS' ||
    error.status === 0 ||
    error.status >= 500
  )
}

function conversationTitle(question: string): string {
  const characters = Array.from(question)
  return characters.length <= 80
    ? question
    : `${characters.slice(0, 79).join('')}…`
}

export function createChatController(
  api: ChatApi = chatApi,
  conversations: ConversationCreator = conversationsApi,
): ChatController {
  const state = shallowRef<ChatState>({ status: 'idle' })
  const status = shallowRef<ChatStatus>('idle')
  const pending = shallowRef<PendingTurn | null>(null)
  const turns = shallowRef<ChatTurn[]>([])
  let operation = 0

  function setState(next: ChatState): void {
    state.value = next
    status.value = next.status
  }

  function addResponse(response: ChatResponse): void {
    const duplicate = turns.value.some(
      (turn) =>
        turn.response.answer_message_id === response.answer_message_id ||
        turn.response.client_request_id === response.client_request_id,
    )
    if (!duplicate) turns.value = [...turns.value, { response }]
  }

  async function execute(turn: PendingTurn): Promise<ChatResponse | undefined> {
    const currentOperation = ++operation
    pending.value = turn
    setState({ status: 'submitting' })
    const request: ChatRequest = {
      conversation_id: turn.conversationId,
      client_request_id: turn.clientRequestId,
      question: turn.question,
      top_k: turn.topK,
    }

    try {
      const response = await api.answer(request, turn.controller.signal)
      if (currentOperation !== operation) return undefined
      addResponse(response)
      pending.value = null
      setState({ status: 'success', response })
      return response
    } catch (error) {
      if (currentOperation !== operation) return undefined
      if (isAbort(error)) {
        pending.value = null
        setState({ status: 'cancelled' })
        return undefined
      }

      const apiError = normalizedError(error)
      if (apiError.code === 'RAG_CANCELLED') {
        pending.value = null
        setState({ status: 'cancelled' })
        return undefined
      }
      const retryable = canRetry(apiError)
      if (!retryable) pending.value = null
      setState({ status: 'error', error: apiError, retryable })
      return undefined
    }
  }

  async function submit(
    options: SubmitChatOptions,
  ): Promise<ChatResponse | undefined> {
    if (state.value.status === 'submitting') return undefined

    const question = options.question.trim()
    if (!question || question.length > 8000) {
      pending.value = null
      setState({
        status: 'error',
        error: validationError('问题长度必须在 1 到 8000 个字符之间。'),
        retryable: false,
      })
      return undefined
    }
    if (
      !Number.isInteger(options.topK) ||
      options.topK < 1 ||
      options.topK > 10
    ) {
      pending.value = null
      setState({
        status: 'error',
        error: validationError('检索数量必须是 1 到 10 的整数。'),
        retryable: false,
      })
      return undefined
    }

    const startOperation = ++operation
    setState({ status: 'submitting' })
    let conversationId = options.conversationId
    try {
      if (!conversationId) {
        const conversation = await conversations.create(
          conversationTitle(question),
        )
        if (startOperation !== operation) return undefined
        conversationId = conversation.id
        await options.onConversationCreated?.(conversationId)
        if (startOperation !== operation) return undefined
      }
    } catch (error) {
      if (startOperation !== operation) return undefined
      const apiError = normalizedError(error)
      setState({ status: 'error', error: apiError, retryable: false })
      return undefined
    }

    return execute({
      conversationId,
      clientRequestId: crypto.randomUUID(),
      question,
      topK: options.topK,
      controller: new AbortController(),
    })
  }

  function retry(): Promise<ChatResponse | undefined> {
    if (
      state.value.status !== 'error' ||
      !state.value.retryable ||
      !pending.value
    ) {
      return Promise.resolve(undefined)
    }
    return execute({
      ...pending.value,
      controller: new AbortController(),
    })
  }

  function cancel(): void {
    if (state.value.status !== 'submitting') return
    operation += 1
    pending.value?.controller.abort()
    pending.value = null
    setState({ status: 'cancelled' })
  }

  function clearTurns(): void {
    turns.value = []
    if (state.value.status !== 'submitting') setState({ status: 'idle' })
  }

  function dispose(): void {
    cancel()
  }

  return {
    state,
    status,
    pending,
    turns,
    submit,
    retry,
    cancel,
    clearTurns,
    dispose,
  }
}

export function useChat(): ChatController {
  const controller = createChatController()
  onBeforeUnmount(controller.dispose)
  return controller
}
