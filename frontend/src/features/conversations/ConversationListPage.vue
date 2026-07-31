<script setup lang="ts">
import { Plus, RefreshCw } from '@lucide/vue'
import { computed, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import ConfirmDialog from '../../components/ConfirmDialog.vue'
import EmptyState from '../../components/EmptyState.vue'
import IconButton from '../../components/IconButton.vue'
import InlineAlert from '../../components/InlineAlert.vue'
import LoadingState from '../../components/LoadingState.vue'
import PaginationControls from '../../components/PaginationControls.vue'
import { ApiError } from '../../api/errors'
import { conversationsApi } from './api'
import ConversationRow from './ConversationRow.vue'
import type { Conversation, ConversationStatus } from './types'
import { useConversationList } from './useConversations'

const PAGE_SIZE = 20
const route = useRoute()
const router = useRouter()
const query = useConversationList()
const mutationError = ref<ApiError | null>(null)
const busyId = ref<string | null>(null)
const creating = ref(false)
const archiveTarget = ref<Conversation | null>(null)

const status = computed<ConversationStatus>(() =>
  route.query.status === 'archived' ? 'archived' : 'active',
)
const page = computed(() => {
  const value = Number(route.query.page)
  return Number.isInteger(value) && value >= 1 ? value : 1
})

function setStatus(nextStatus: ConversationStatus): void {
  void router.push({
    name: 'conversations',
    query: { status: nextStatus, page: '1' },
  })
}

function setOffset(offset: number): void {
  void router.push({
    name: 'conversations',
    query: {
      status: status.value,
      page: String(Math.floor(offset / PAGE_SIZE) + 1),
    },
  })
}

function asApiError(error: unknown): ApiError {
  return error instanceof ApiError
    ? error
    : new ApiError({
        code: 'NETWORK_ERROR',
        message: '操作失败，请稍后重试。',
        requestId: 'unavailable',
        status: 0,
        details: null,
      })
}

async function createConversation(): Promise<void> {
  creating.value = true
  mutationError.value = null
  try {
    const conversation = await conversationsApi.create('新对话')
    await router.push(`/conversations/${conversation.id}`)
  } catch (error) {
    mutationError.value = asApiError(error)
  } finally {
    creating.value = false
  }
}

async function renameConversation(
  conversation: Conversation,
  title: string,
): Promise<void> {
  if (title === conversation.title) return
  busyId.value = conversation.id
  mutationError.value = null
  try {
    await conversationsApi.update(conversation.id, { title })
    await query.refresh()
  } catch (error) {
    mutationError.value = asApiError(error)
  } finally {
    busyId.value = null
  }
}

async function archiveConversation(): Promise<void> {
  const target = archiveTarget.value
  if (!target) return
  busyId.value = target.id
  mutationError.value = null
  try {
    await conversationsApi.archive(target.id)
    archiveTarget.value = null
    await query.refresh()
  } catch (error) {
    mutationError.value = asApiError(error)
  } finally {
    busyId.value = null
  }
}

watch(
  [status, page],
  ([nextStatus, nextPage]) => {
    void query.load({ status: nextStatus, page: nextPage, pageSize: PAGE_SIZE })
  },
  { immediate: true },
)
</script>

<template>
  <main class="conversation-page">
    <header class="conversation-page__toolbar">
      <div class="segmented-control" aria-label="会话状态">
        <button
          type="button"
          :aria-pressed="status === 'active'"
          @click="setStatus('active')"
        >
          活跃会话
        </button>
        <button
          type="button"
          :aria-pressed="status === 'archived'"
          @click="setStatus('archived')"
        >
          已归档
        </button>
      </div>
      <div class="conversation-page__commands">
        <IconButton label="刷新会话" @click="query.refresh">
          <RefreshCw :size="18" aria-hidden="true" />
        </IconButton>
        <button
          class="primary-command"
          type="button"
          :disabled="creating"
          @click="createConversation"
        >
          <Plus :size="18" aria-hidden="true" />
          {{ creating ? '正在创建' : '新建会话' }}
        </button>
      </div>
    </header>

    <InlineAlert v-if="mutationError" tone="error" title="操作未完成">
      <p>{{ mutationError.message }}</p>
      <small>请求编号：{{ mutationError.requestId }}</small>
    </InlineAlert>

    <InlineAlert
      v-if="query.state.value.status === 'error'"
      tone="error"
      title="无法加载会话"
    >
      <p>{{ query.state.value.error.message }}</p>
      <small>请求编号：{{ query.state.value.error.requestId }}</small>
      <button class="inline-command" type="button" @click="query.refresh">
        重试
      </button>
    </InlineAlert>
    <LoadingState
      v-if="query.state.value.status === 'loading' && !query.state.value.data"
      label="正在加载会话"
    />
    <template v-else-if="query.state.value.data">
      <EmptyState
        v-if="!query.state.value.data.items.length"
        title="还没有会话"
        :description="
          status === 'active'
            ? '新建一个会话，开始整理你的数学问题。'
            : '归档后的会话会显示在这里。'
        "
      >
        <template v-if="status === 'active'" #action>
          <button
            class="primary-command"
            type="button"
            @click="createConversation"
          >
            <Plus :size="18" aria-hidden="true" />
            新建会话
          </button>
        </template>
      </EmptyState>
      <section v-else class="conversation-list" aria-label="会话列表">
        <ul>
          <ConversationRow
            v-for="conversation in query.state.value.data.items"
            :key="conversation.id"
            :conversation="conversation"
            :busy="busyId === conversation.id"
            @archive="archiveTarget = $event"
            @rename="renameConversation"
          />
        </ul>
        <PaginationControls
          :limit="PAGE_SIZE"
          :offset="(page - 1) * PAGE_SIZE"
          :total="query.state.value.data.total"
          @update:offset="setOffset"
        />
      </section>
    </template>

    <ConfirmDialog
      :open="archiveTarget !== null"
      :busy="archiveTarget !== null && busyId === archiveTarget.id"
      title="归档会话"
      :object-name="archiveTarget?.title ?? ''"
      confirm-label="确认归档"
      danger
      @cancel="archiveTarget = null"
      @confirm="archiveConversation"
    />
  </main>
</template>

<style scoped>
.conversation-page {
  width: min(100%, 980px);
  margin: 0 auto;
  padding: var(--space-6);
}

.conversation-page__toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-4);
  margin-bottom: var(--space-5);
}

.conversation-page__commands,
.segmented-control,
.primary-command {
  display: flex;
  align-items: center;
}

.conversation-page__commands {
  gap: var(--space-2);
}

.segmented-control {
  padding: 2px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-neutral-0);
}

.segmented-control button {
  min-height: 34px;
  padding: 0 var(--space-3);
  border: 0;
  border-radius: var(--radius-sm);
  color: var(--color-text-secondary);
  background: transparent;
  font-size: 13px;
}

.segmented-control button[aria-pressed='true'] {
  color: var(--color-text-primary);
  background: var(--color-neutral-100);
  font-weight: 650;
}

.primary-command {
  min-height: 38px;
  justify-content: center;
  gap: var(--space-2);
  padding: 0 var(--space-4);
  border: 1px solid var(--color-action);
  border-radius: var(--radius-md);
  color: var(--color-neutral-0);
  background: var(--color-action);
  font-weight: 650;
}

.primary-command:hover {
  background: var(--color-action-hover);
}

.primary-command:disabled {
  opacity: 0.6;
}

.conversation-list {
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: var(--color-neutral-0);
}

.conversation-list ul {
  margin: 0;
  padding: 0;
  list-style: none;
}

.conversation-list :deep(.pagination) {
  padding: var(--space-2) var(--space-3);
  border-top: 1px solid var(--color-border);
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
  .conversation-page {
    padding: var(--space-4);
  }

  .conversation-page__toolbar {
    align-items: stretch;
    flex-direction: column-reverse;
  }

  .conversation-page__commands {
    justify-content: flex-end;
  }

  .segmented-control button {
    flex: 1;
  }
}
</style>
