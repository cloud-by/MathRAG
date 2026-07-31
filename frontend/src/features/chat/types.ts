import type { components } from '../../api/schema'

export type ChatRequest = components['schemas']['ChatV1Request']
export type ChatResponse = components['schemas']['ChatV1Response']

export type AnswerContent = Pick<
  ChatResponse,
  | 'agentic_plan'
  | 'answer'
  | 'reasoning_content'
  | 'references'
  | 'related_questions'
  | 'steps'
  | 'used_knowledge'
>

export type ReferenceItem = components['schemas']['ReferenceItem']

export interface PendingTurn {
  conversationId: string
  clientRequestId: string
  question: string
  topK: number
  controller: AbortController
}

export interface ChatTurn {
  response: ChatResponse
}

export type ChatStatus =
  'idle' | 'submitting' | 'success' | 'error' | 'cancelled'
