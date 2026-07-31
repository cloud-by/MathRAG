<script setup lang="ts">
import { computed } from 'vue'

import MathContent from '../../components/MathContent.vue'
import type { Message } from '../conversations/types'
import { answerFromMessage } from '../conversations/types'
import AnswerView from './AnswerView.vue'
import type { AnswerContent, ChatTurn, PendingTurn } from './types'

type DisplayEntry =
  | { id: string; role: 'user'; content: string; status: Message['status'] }
  | {
      id: string
      role: 'assistant'
      content: string
      answer: AnswerContent | null
      status: Message['status']
    }
  | { id: string; role: 'system'; content: string; status: Message['status'] }

const props = withDefaults(
  defineProps<{
    messages?: Message[]
    pending?: PendingTurn | null
    turns?: ChatTurn[]
  }>(),
  { messages: () => [], pending: null, turns: () => [] },
)

const emit = defineEmits<{
  selectRelated: [question: string]
}>()

function statusLabel(status: Message['status']): string {
  if (status === 'pending') return '正在处理'
  if (status === 'failed') return '未完成'
  return ''
}

const entries = computed<DisplayEntry[]>(() => {
  const result: DisplayEntry[] = props.messages.map((message) => {
    if (message.role === 'assistant') {
      return {
        id: message.id,
        role: message.role,
        content: message.content,
        answer: answerFromMessage(message),
        status: message.status,
      }
    }
    return {
      id: message.id,
      role: message.role,
      content: message.content,
      status: message.status,
    }
  })
  const ids = new Set(result.map((entry) => entry.id))

  for (const turn of props.turns) {
    if (!ids.has(turn.response.question_message_id)) {
      result.push({
        id: turn.response.question_message_id,
        role: 'user',
        content: turn.response.question,
        status: 'completed',
      })
      ids.add(turn.response.question_message_id)
    }
    if (!ids.has(turn.response.answer_message_id)) {
      result.push({
        id: turn.response.answer_message_id,
        role: 'assistant',
        content: turn.response.answer,
        answer: turn.response,
        status: 'completed',
      })
      ids.add(turn.response.answer_message_id)
    }
  }

  if (props.pending) {
    const alreadyCompleted = props.turns.some(
      (turn) =>
        turn.response.client_request_id === props.pending?.clientRequestId,
    )
    const alreadyPersisted = props.messages.some(
      (message) =>
        message.role === 'user' && message.content === props.pending?.question,
    )
    if (!alreadyCompleted && !alreadyPersisted) {
      result.push({
        id: `pending-${props.pending.clientRequestId}`,
        role: 'user',
        content: props.pending.question,
        status: 'pending',
      })
    }
  }
  return result
})
</script>

<template>
  <section v-if="entries.length" class="message-list" aria-label="会话消息">
    <article
      v-for="entry in entries"
      :key="entry.id"
      class="chat-message"
      :class="`chat-message--${entry.role}`"
      data-testid="history-message"
    >
      <header>
        <h2>
          {{
            entry.role === 'user'
              ? '你'
              : entry.role === 'assistant'
                ? 'MathRAG'
                : '系统'
          }}
        </h2>
        <span v-if="entry.status !== 'completed'">
          {{ statusLabel(entry.status) }}
        </span>
      </header>
      <AnswerView
        v-if="entry.role === 'assistant' && entry.answer"
        :answer="entry.answer"
        @select-related="emit('selectRelated', $event)"
      />
      <MathContent v-else :content="entry.content" />
    </article>
  </section>
</template>

<style scoped>
.message-list {
  border-top: 1px solid var(--color-border);
}

.chat-message {
  padding: var(--space-5) 0;
  border-bottom: 1px solid var(--color-border);
}

.chat-message > header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3);
  margin-bottom: var(--space-3);
}

.chat-message > header h2 {
  margin: 0;
  color: var(--color-text-secondary);
  font-size: 13px;
  letter-spacing: 0;
}

.chat-message > header span {
  color: var(--color-warning);
  font-size: 12px;
}

.chat-message--user {
  padding-left: var(--space-5);
  border-left: 3px solid var(--color-neutral-200);
}

.chat-message :deep(.answer-view) {
  max-width: none;
}

@media (max-width: 640px) {
  .chat-message--user {
    padding-left: var(--space-3);
  }
}
</style>
