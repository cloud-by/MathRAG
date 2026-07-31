<script setup lang="ts">
import { ArrowLeft, RefreshCw } from '@lucide/vue'
import { computed, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'

import EmptyState from '../../components/EmptyState.vue'
import IconButton from '../../components/IconButton.vue'
import InlineAlert from '../../components/InlineAlert.vue'
import LoadingState from '../../components/LoadingState.vue'
import PaginationControls from '../../components/PaginationControls.vue'
import MessageList from '../chat/MessageList.vue'
import type { ChatTurn, PendingTurn } from '../chat/types'
import { useConversationHistory } from './useConversations'

withDefaults(
  defineProps<{
    pending?: PendingTurn | null
    turns?: ChatTurn[]
  }>(),
  { pending: null, turns: () => [] },
)

const emit = defineEmits<{
  selectRelated: [question: string]
}>()

const PAGE_SIZE = 50
const route = useRoute()
const router = useRouter()
const query = useConversationHistory()

const conversationId = computed(() => String(route.params.id ?? ''))
const page = computed(() => {
  const value = Number(route.query.message_page)
  return Number.isInteger(value) && value >= 1 ? value : 1
})

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

defineExpose({ refresh: query.refresh })
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
        <IconButton label="刷新历史" @click="query.refresh">
          <RefreshCw :size="18" aria-hidden="true" />
        </IconButton>
      </header>

      <div class="history-page__messages">
        <EmptyState
          v-if="
            !query.state.value.data.messages.items.length &&
            !turns.length &&
            !pending
          "
          title="这个会话还没有消息"
          description="继续对话后，问题和回答会保存在这里。"
        />
        <MessageList
          v-else
          :messages="query.state.value.data.messages.items"
          :turns="turns"
          :pending="pending"
          @select-related="emit('selectRelated', $event)"
        />

        <PaginationControls
          v-if="query.state.value.data.messages.total > PAGE_SIZE"
          :limit="PAGE_SIZE"
          :offset="(page - 1) * PAGE_SIZE"
          :total="query.state.value.data.messages.total"
          @update:offset="setOffset"
        />
      </div>
      <slot name="composer" />
    </template>
  </main>
</template>

<style scoped>
.history-page {
  display: grid;
  height: calc(100vh - 58px);
  width: min(100%, 980px);
  grid-template-rows: auto minmax(0, 1fr) auto;
  margin: 0 auto;
  padding: var(--space-6);
  overflow: hidden;
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

.history-page__back {
  display: flex;
  align-items: center;
}

.history-page__messages {
  min-height: 0;
  overflow-y: auto;
  overscroll-behavior: contain;
}

.history-page__back {
  width: fit-content;
  gap: var(--space-1);
  color: var(--color-text-secondary);
  font-size: 13px;
  text-decoration: none;
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
}
</style>
