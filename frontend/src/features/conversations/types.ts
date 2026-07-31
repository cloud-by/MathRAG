import type { components } from '../../api/schema'
import type { ApiError } from '../../api/errors'
import type { AnswerContent, ReferenceItem } from '../chat/types'

export type Conversation = components['schemas']['ConversationRead']
export type ConversationPage = components['schemas']['ConversationPage']
export type ConversationStatus = Conversation['status']
export type ConversationUpdate = components['schemas']['ConversationUpdate']
export type Message = components['schemas']['MessageRead']
export type MessagePage = components['schemas']['MessagePage']

export interface ConversationQuery {
  page: number
  pageSize: number
  status: ConversationStatus
}

export interface ConversationHistory {
  conversation: Conversation
  messages: MessagePage
}

export type QueryState<T> =
  | { status: 'idle'; data: null }
  | { status: 'loading'; data: T | null }
  | { status: 'success'; data: T }
  | { status: 'error'; data: T | null; error: ApiError }

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function stringArray(value: unknown): string[] | undefined {
  return Array.isArray(value) && value.every((item) => typeof item === 'string')
    ? value
    : undefined
}

function isReference(value: unknown): value is ReferenceItem {
  if (!isRecord(value)) return false
  return (
    typeof value.answer_context === 'string' &&
    typeof value.category === 'string' &&
    typeof value.chunk_id === 'string' &&
    typeof value.content === 'string' &&
    typeof value.difficulty === 'string' &&
    typeof value.example === 'string' &&
    typeof value.rank === 'number' &&
    typeof value.retrieval_text === 'string' &&
    typeof value.score === 'number' &&
    typeof value.source_id === 'string' &&
    typeof value.title === 'string'
  )
}

function references(value: unknown): ReferenceItem[] | undefined {
  return Array.isArray(value) && value.every(isReference) ? value : undefined
}

function agenticPlan(
  value: unknown,
): AnswerContent['agentic_plan'] | undefined {
  if (!isRecord(value) || typeof value.strategy !== 'string') return undefined
  const retrievalQueries = stringArray(value.retrieval_queries)
  if (value.retrieval_queries !== undefined && !retrievalQueries)
    return undefined
  return {
    strategy: value.strategy,
    retrieval_queries: retrievalQueries ?? [],
  }
}

export function answerFromMessage(message: Message): AnswerContent | null {
  const response = message.model_metadata.response
  if (!isRecord(response) || typeof response.answer !== 'string') return null

  return {
    answer: response.answer,
    agentic_plan: agenticPlan(response.agentic_plan),
    reasoning_content:
      typeof response.reasoning_content === 'string'
        ? response.reasoning_content
        : undefined,
    references: references(response.references),
    related_questions: stringArray(response.related_questions),
    steps: stringArray(response.steps),
    used_knowledge: stringArray(response.used_knowledge),
  }
}
