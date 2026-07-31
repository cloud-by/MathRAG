<script setup lang="ts">
import { Archive, Pencil, Plus, RefreshCw } from '@lucide/vue'
import { computed, ref, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'

import ConfirmDialog from '../../components/ConfirmDialog.vue'
import EmptyState from '../../components/EmptyState.vue'
import IconButton from '../../components/IconButton.vue'
import InlineAlert from '../../components/InlineAlert.vue'
import LoadingState from '../../components/LoadingState.vue'
import PaginationControls from '../../components/PaginationControls.vue'
import { ApiError } from '../../api/errors'
import { knowledgeApi } from './api'
import type {
  KnowledgeItem,
  KnowledgeStatus,
  KnowledgeVisibility,
} from './types'
import { useKnowledgeList } from './useKnowledge'

const PAGE_SIZE = 20
const STATUSES = new Set<KnowledgeStatus>([
  'draft',
  'indexing',
  'ready',
  'failed',
  'archived',
])
const route = useRoute()
const router = useRouter()
const query = useKnowledgeList()
const categoryInput = ref(
  typeof route.query.category === 'string' ? route.query.category : '',
)
const archiveTarget = ref<KnowledgeItem | null>(null)
const archiving = ref(false)
const mutationError = ref<ApiError | null>(null)

const status = computed<KnowledgeStatus | undefined>(() => {
  const value = route.query.status
  return typeof value === 'string' && STATUSES.has(value as KnowledgeStatus)
    ? (value as KnowledgeStatus)
    : undefined
})
const visibility = computed<KnowledgeVisibility | undefined>(() =>
  route.query.visibility === 'public' || route.query.visibility === 'private'
    ? route.query.visibility
    : undefined,
)
const category = computed(() =>
  typeof route.query.category === 'string' && route.query.category.trim()
    ? route.query.category.trim()
    : undefined,
)
const page = computed(() => {
  const value = Number(route.query.page)
  return Number.isInteger(value) && value >= 1 ? value : 1
})

function updateFilters(values: {
  status?: KnowledgeStatus
  visibility?: KnowledgeVisibility
  category?: string
  page?: number
}): void {
  void router.push({
    name: 'knowledge',
    query: {
      ...(values.status ? { status: values.status } : {}),
      ...(values.visibility ? { visibility: values.visibility } : {}),
      ...(values.category ? { category: values.category } : {}),
      page: String(values.page ?? 1),
    },
  })
}

function applyCategory(): void {
  updateFilters({
    status: status.value,
    visibility: visibility.value,
    category: categoryInput.value.trim() || undefined,
  })
}

function setStatus(event: Event): void {
  const value = (event.target as HTMLSelectElement).value
  updateFilters({
    status: value ? (value as KnowledgeStatus) : undefined,
    visibility: visibility.value,
    category: category.value,
  })
}

function setVisibility(event: Event): void {
  const value = (event.target as HTMLSelectElement).value
  updateFilters({
    status: status.value,
    visibility: value ? (value as KnowledgeVisibility) : undefined,
    category: category.value,
  })
}

function setOffset(offset: number): void {
  updateFilters({
    status: status.value,
    visibility: visibility.value,
    category: category.value,
    page: Math.floor(offset / PAGE_SIZE) + 1,
  })
}

function statusLabel(value: KnowledgeStatus): string {
  return {
    draft: '草稿',
    indexing: '索引中',
    ready: '可用',
    failed: '失败',
    archived: '已归档',
  }[value]
}

function difficultyLabel(value: KnowledgeItem['difficulty']): string {
  return { easy: '简单', medium: '中等', hard: '困难' }[value]
}

async function archiveItem(): Promise<void> {
  const target = archiveTarget.value
  if (!target) return
  archiving.value = true
  mutationError.value = null
  try {
    await knowledgeApi.archive(target.id, target.revision)
    archiveTarget.value = null
    await query.refresh()
  } catch (error) {
    mutationError.value =
      error instanceof ApiError
        ? error
        : new ApiError({
            code: 'NETWORK_ERROR',
            message: '归档失败，请稍后重试。',
            requestId: 'unavailable',
            status: 0,
            details: null,
          })
    if (mutationError.value.code === 'KNOWLEDGE_REVISION_CONFLICT') {
      archiveTarget.value = null
      await query.refresh()
    }
  } finally {
    archiving.value = false
  }
}

watch(
  [status, visibility, category, page],
  ([nextStatus, nextVisibility, nextCategory, nextPage]) => {
    categoryInput.value = nextCategory ?? ''
    void query.load({
      status: nextStatus,
      visibility: nextVisibility,
      category: nextCategory,
      page: nextPage,
      pageSize: PAGE_SIZE,
    })
  },
  { immediate: true },
)
</script>

<template>
  <main class="knowledge-page">
    <header class="knowledge-page__toolbar">
      <form class="knowledge-filters" @submit.prevent="applyCategory">
        <label>
          <span>状态</span>
          <select :value="status ?? ''" @change="setStatus">
            <option value="">所有状态</option>
            <option value="draft">草稿</option>
            <option value="indexing">索引中</option>
            <option value="ready">可用</option>
            <option value="failed">失败</option>
            <option value="archived">已归档</option>
          </select>
        </label>
        <label>
          <span>可见性</span>
          <select :value="visibility ?? ''" @change="setVisibility">
            <option value="">全部</option>
            <option value="public">公开</option>
            <option value="private">私有</option>
          </select>
        </label>
        <label class="knowledge-filters__category">
          <span>类别</span>
          <input v-model="categoryInput" type="search" placeholder="精确类别" />
        </label>
        <button class="secondary-command" type="submit">应用筛选</button>
      </form>
      <div class="knowledge-page__commands">
        <IconButton label="刷新知识库" @click="query.refresh">
          <RefreshCw :size="18" aria-hidden="true" />
        </IconButton>
        <RouterLink class="primary-command" to="/knowledge/new">
          <Plus :size="18" aria-hidden="true" />
          新建条目
        </RouterLink>
      </div>
    </header>

    <InlineAlert v-if="mutationError" tone="error" title="操作未完成">
      <p>{{ mutationError.message }}</p>
      <small>请求编号：{{ mutationError.requestId }}</small>
    </InlineAlert>
    <InlineAlert
      v-if="query.state.value.status === 'error'"
      tone="error"
      title="无法加载知识库"
    >
      <p>{{ query.state.value.error.message }}</p>
      <button class="inline-command" type="button" @click="query.refresh">
        重试
      </button>
    </InlineAlert>
    <LoadingState
      v-if="query.state.value.status === 'loading' && !query.state.value.data"
      label="正在加载知识条目"
    />
    <template v-else-if="query.state.value.data">
      <EmptyState
        v-if="!query.state.value.data.items.length"
        title="没有符合条件的知识条目"
      />
      <section v-else class="knowledge-table-wrap" aria-label="知识条目列表">
        <table class="knowledge-table">
          <thead>
            <tr>
              <th>标题</th>
              <th>类别</th>
              <th>难度</th>
              <th>可见性</th>
              <th>状态</th>
              <th>修订</th>
              <th aria-label="操作"></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="item in query.state.value.data.items" :key="item.id">
              <td>
                <RouterLink :to="`/knowledge/${item.id}`">{{
                  item.title
                }}</RouterLink>
              </td>
              <td>{{ item.category }}</td>
              <td>{{ difficultyLabel(item.difficulty) }}</td>
              <td>{{ item.visibility === 'public' ? '公开' : '私有' }}</td>
              <td>{{ statusLabel(item.status) }}</td>
              <td>{{ item.revision }}</td>
              <td class="knowledge-table__actions">
                <IconButton
                  :label="`编辑“${item.title}”`"
                  @click="router.push(`/knowledge/${item.id}`)"
                >
                  <Pencil :size="17" aria-hidden="true" />
                </IconButton>
                <IconButton
                  v-if="item.status !== 'archived'"
                  :label="`归档“${item.title}”`"
                  @click="archiveTarget = item"
                >
                  <Archive :size="17" aria-hidden="true" />
                </IconButton>
              </td>
            </tr>
          </tbody>
        </table>
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
      :busy="archiving"
      title="归档知识条目"
      :object-name="archiveTarget?.title ?? ''"
      confirm-label="确认归档"
      danger
      @cancel="archiveTarget = null"
      @confirm="archiveItem"
    />
  </main>
</template>

<style scoped>
.knowledge-page {
  width: min(100%, 1180px);
  margin: 0 auto;
  padding: var(--space-6);
}

.knowledge-page__toolbar {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: var(--space-4);
  margin-bottom: var(--space-5);
}

.knowledge-filters {
  display: flex;
  flex: 1;
  flex-wrap: wrap;
  align-items: flex-end;
  gap: var(--space-3);
}

.knowledge-filters label {
  display: grid;
  gap: var(--space-1);
  color: var(--color-text-secondary);
  font-size: 12px;
}

.knowledge-filters select,
.knowledge-filters input {
  min-height: 38px;
  padding: 0 var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-neutral-0);
}

.knowledge-filters__category input {
  width: 180px;
}

.knowledge-page__commands,
.primary-command,
.secondary-command {
  display: flex;
  align-items: center;
}

.knowledge-page__commands {
  flex: 0 0 auto;
  gap: var(--space-2);
}

.primary-command,
.secondary-command {
  min-height: 38px;
  justify-content: center;
  gap: var(--space-2);
  padding: 0 var(--space-4);
  border-radius: var(--radius-md);
  font-weight: 650;
  text-decoration: none;
}

.primary-command {
  border: 1px solid var(--color-action);
  color: var(--color-neutral-0);
  background: var(--color-action);
}

.secondary-command {
  border: 1px solid var(--color-border);
  background: var(--color-neutral-0);
}

.knowledge-table-wrap {
  overflow-x: auto;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: var(--color-neutral-0);
}

.knowledge-table {
  width: 100%;
  min-width: 820px;
  border-collapse: collapse;
  font-size: 13px;
}

.knowledge-table th,
.knowledge-table td {
  padding: var(--space-3) var(--space-4);
  border-bottom: 1px solid var(--color-border);
  text-align: left;
}

.knowledge-table th {
  color: var(--color-text-secondary);
  background: var(--color-neutral-50);
  font-size: 12px;
}

.knowledge-table td:first-child {
  min-width: 240px;
  font-weight: 650;
}

.knowledge-table__actions {
  display: flex;
  justify-content: flex-end;
  gap: var(--space-1);
}

.knowledge-table-wrap :deep(.pagination) {
  padding: var(--space-2) var(--space-3);
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
}

@media (max-width: 760px) {
  .knowledge-page {
    padding: var(--space-4);
  }

  .knowledge-page__toolbar {
    align-items: stretch;
    flex-direction: column-reverse;
  }

  .knowledge-page__commands {
    justify-content: flex-end;
  }

  .knowledge-filters label,
  .knowledge-filters__category input {
    min-width: 0;
    flex: 1 1 130px;
    width: auto;
  }
}
</style>
