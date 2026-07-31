<script setup lang="ts">
import { ArrowLeft, MessageSquarePlus, RefreshCw } from '@lucide/vue'
import { computed, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'

import EmptyState from '../../components/EmptyState.vue'
import IconButton from '../../components/IconButton.vue'
import InlineAlert from '../../components/InlineAlert.vue'
import LoadingState from '../../components/LoadingState.vue'
import MathContent from '../../components/MathContent.vue'
import PaginationControls from '../../components/PaginationControls.vue'
import AnswerView from '../chat/AnswerView.vue'
import type { Message } from './types'
import { answerFromMessage } from './types'
import { useConversationHistory } from './useConversations'

const PAGE_SIZE = 50
const route = useRoute()
const router = useRouter()
const query = useConversationHistory()

const conversationId = computed(() => String(route.params.id ?? ''))
const page = computed(() => {
  const value = Number(route.query.message_page)
  return Number.isInteger(value) && value >= 1 ? value : 1
})

function roleLabel(role: Message['role']): string {
  if (role === 'user') return '你'
  if (role === 'assistant') return 'MathRAG'
  return '系统'
}

function statusLabel(status: Message['status']): string {
  if (status === 'pending') return '正在处理'
  if (status === 'failed') return '未完成'
  return ''
}

function setOffset(offset: number): void {
  void router.push({
    name: 'conversation',
    params: { id: conversationId.value },
    query: { message_page: String(Math.floor(offset / PAGE_SIZE) + 1) },
  })
}

watch(
  [conversationId, page],
  ([id, nextPage]) => {
    if (id) void query.load(id, nextPage, PAGE_SIZE)
  },
  { immediate: true },
)
</script>

<template>
  <main class="history-page">
    <InlineAlert
      v-if="query.state.value.status === 'error'"
      tone="error"
      title="无法读取会话"
    >
      <p>{{ query.state.value.error.message }}</p>
      <small>请求编号：{{ query.state.value.error.requestId }}</small>
      <button class="inline-command" type="button" @click="query.refresh">
        重试
      </button>
    </InlineAlert>
    <LoadingState
      v-if="query.state.value.status === 'loading' && !query.state.value.data"
      label="正在恢复会话"
    />
    <template v-else-if="query.state.value.data">
      <header class="history-page__header">
        <div>
          <RouterLink class="history-page__back" to="/conversations">
            <ArrowLeft :size="16" aria-hidden="true" />
            返回会话列表
          </RouterLink>
          <h2>{{ query.state.value.data.conversation.title }}</h2>
        </div>
        <div class="history-page__commands">
          <IconButton label="刷新历史" @click="query.refresh">
            <RefreshCw :size="18" aria-hidden="true" />
          </IconButton>
          <RouterLink
            class="primary-command"
            :to="`/chat?conversation_id=${conversationId}`"
          >
            <MessageSquarePlus :size="18" aria-hidden="true" />
            继续对话
          </RouterLink>
        </div>
      </header>

      <EmptyState
        v-if="!query.state.value.data.messages.items.length"
        title="这个会话还没有消息"
        description="继续对话后，问题和回答会保存在这里。"
      />
      <section v-else class="history-feed" aria-label="历史消息">
        <article
          v-for="item in query.state.value.data.messages.items"
          :key="item.id"
          class="history-message"
          :class="`history-message--${item.role}`"
          data-testid="history-message"
        >
          <header>
            <h2>{{ roleLabel(item.role) }}</h2>
            <span v-if="item.status !== 'completed'">
              {{ statusLabel(item.status) }}
            </span>
          </header>
          <AnswerView
            v-if="item.role === 'assistant' && answerFromMessage(item)"
            :answer="answerFromMessage(item)!"
          />
          <MathContent v-else :content="item.content" />
        </article>
      </section>

      <PaginationControls
        v-if="query.state.value.data.messages.total > PAGE_SIZE"
        :limit="PAGE_SIZE"
        :offset="(page - 1) * PAGE_SIZE"
        :total="query.state.value.data.messages.total"
        @update:offset="setOffset"
      />
    </template>
  </main>
</template>

<style scoped>
.history-page {
  width: min(100%, 980px);
  margin: 0 auto;
  padding: var(--space-6);
}

.history-page__header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: var(--space-5);
  margin-bottom: var(--space-6);
}

.history-page__header h2 {
  margin: var(--space-2) 0 0;
  overflow-wrap: anywhere;
  font-size: 24px;
  letter-spacing: 0;
}

.history-page__back,
.history-page__commands,
.primary-command {
  display: flex;
  align-items: center;
}

.history-page__back {
  width: fit-content;
  gap: var(--space-1);
  color: var(--color-text-secondary);
  font-size: 13px;
  text-decoration: none;
}

.history-page__commands {
  flex: 0 0 auto;
  gap: var(--space-2);
}

.primary-command {
  min-height: 38px;
  gap: var(--space-2);
  padding: 0 var(--space-4);
  border-radius: var(--radius-md);
  color: var(--color-neutral-0);
  background: var(--color-action);
  font-weight: 650;
  text-decoration: none;
}

.history-feed {
  border-top: 1px solid var(--color-border);
}

.history-message {
  padding: var(--space-5) 0;
  border-bottom: 1px solid var(--color-border);
}

.history-message > header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3);
  margin-bottom: var(--space-3);
}

.history-message > header h2 {
  margin: 0;
  color: var(--color-text-secondary);
  font-size: 13px;
  letter-spacing: 0;
}

.history-message > header span {
  color: var(--color-warning);
  font-size: 12px;
}

.history-message--user {
  padding-left: var(--space-5);
  border-left: 3px solid var(--color-neutral-200);
}

.history-message :deep(.answer-view) {
  max-width: none;
}

.inline-command {
  margin-top: var(--space-3);
  padding: 0;
  border: 0;
  color: var(--color-action);
  background: transparent;
  font-weight: 650;
}

small {
  display: block;
  margin-top: var(--space-1);
  color: var(--color-text-secondary);
}

@media (max-width: 640px) {
  .history-page {
    padding: var(--space-4);
  }

  .history-page__header {
    align-items: stretch;
    flex-direction: column;
  }

  .history-page__commands {
    justify-content: flex-end;
  }

  .history-message--user {
    padding-left: var(--space-3);
  }
}
</style>
