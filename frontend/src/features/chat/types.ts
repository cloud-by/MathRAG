import type { components } from '../../api/schema'

type ChatResponse = components['schemas']['ChatV1Response']

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
