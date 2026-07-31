import type { components } from '../../api/schema'
import type { ApiError } from '../../api/errors'

export type KnowledgeItem = components['schemas']['KnowledgeItemRead']
export type KnowledgePage = components['schemas']['KnowledgeItemPage']
export type KnowledgeCreate = components['schemas']['KnowledgeItemCreate']
export type KnowledgeUpdate = components['schemas']['KnowledgeItemUpdate']
export type KnowledgeStatus = KnowledgeItem['status']
export type KnowledgeVisibility = KnowledgeItem['visibility']

export interface KnowledgeFilters {
  status?: KnowledgeStatus
  visibility?: KnowledgeVisibility
  category?: string
  page: number
  pageSize: number
}

export type KnowledgeDraft = KnowledgeCreate

export type KnowledgeQueryState<T> =
  | { status: 'idle'; data: null }
  | { status: 'loading'; data: T | null }
  | { status: 'success'; data: T }
  | { status: 'error'; data: T | null; error: ApiError }

export type KnowledgeMutationResult =
  | { kind: 'saved'; item: KnowledgeItem }
  | { kind: 'conflict'; server: KnowledgeItem; error: ApiError }
  | { kind: 'error'; error: ApiError }

export function emptyKnowledgeDraft(): KnowledgeDraft {
  return {
    category: '',
    title: '',
    keywords: [],
    content: '',
    example: '',
    steps: [''],
    difficulty: 'medium',
    visibility: 'public',
  }
}

export function draftFromItem(item: KnowledgeItem): KnowledgeDraft {
  return {
    category: item.category,
    title: item.title,
    keywords: [...item.keywords],
    content: item.content,
    example: item.example,
    steps: [...item.steps],
    difficulty: item.difficulty,
    visibility: item.visibility,
  }
}

export function normalizedDraft(draft: KnowledgeDraft): KnowledgeDraft {
  return {
    category: draft.category.trim(),
    title: draft.title.trim(),
    keywords: [
      ...new Set(draft.keywords.map((value) => value.trim()).filter(Boolean)),
    ],
    content: draft.content.trim(),
    example: draft.example.trim(),
    steps: draft.steps.map((value) => value.trim()).filter(Boolean),
    difficulty: draft.difficulty,
    visibility: draft.visibility,
  }
}
