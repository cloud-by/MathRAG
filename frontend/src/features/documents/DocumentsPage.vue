<script setup lang="ts">
import { RefreshCw } from '@lucide/vue'
import { computed, onBeforeUnmount, shallowRef, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { ApiError } from '../../api/errors'
import EmptyState from '../../components/EmptyState.vue'
import IconButton from '../../components/IconButton.vue'
import InlineAlert from '../../components/InlineAlert.vue'
import LoadingState from '../../components/LoadingState.vue'
import PaginationControls from '../../components/PaginationControls.vue'
import { documentsApi } from './api'
import DocumentUpload from './DocumentUpload.vue'
import type { DocumentPage, DocumentStatus, KnowledgeDocument } from './types'

const PAGE_SIZE = 20
const STATUSES = new Set<DocumentStatus>([
  'pending',
  'processing',
  'ready',
  'failed',
  'archived',
])
type QueryState =
  | { status: 'idle' | 'loading'; data: DocumentPage | null }
  | { status: 'success'; data: DocumentPage }
  | { status: 'error'; data: DocumentPage | null; error: ApiError }

const route = useRoute()
const router = useRouter()
const state = shallowRef<QueryState>({ status: 'idle', data: null })
let controller: AbortController | null = null
let sequence = 0

const status = computed<DocumentStatus | undefined>(() => {
  const value = route.query.status
  return typeof value === 'string' && STATUSES.has(value as DocumentStatus)
    ? (value as DocumentStatus)
    : undefined
})
const page = computed(() => {
  const value = Number(route.query.page)
  return Number.isInteger(value) && value >= 1 ? value : 1
})

function asApiError(error: unknown): ApiError {
  if (error instanceof ApiError) return error
  return new ApiError({
    status: 0,
    code: 'NETWORK_ERROR',
    message: '无法加载文档，请稍后重试。',
    requestId: 'unavailable',
    details: null,
  })
}

async function load(): Promise<void> {
  controller?.abort()
  controller = new AbortController()
  const request = ++sequence
  const previous = state.value.data
  state.value = { status: 'loading', data: previous }
  try {
    const data = await documentsApi.list(
      { status: status.value, page: page.value, pageSize: PAGE_SIZE },
      controller.signal,
    )
    if (request === sequence) state.value = { status: 'success', data }
  } catch (error) {
    if (
      request !== sequence ||
      (error instanceof DOMException && error.name === 'AbortError')
    ) {
      return
    }
    state.value = { status: 'error', data: previous, error: asApiError(error) }
  }
}

function setStatus(event: Event): void {
  const value = (event.target as HTMLSelectElement).value
  void router.push({
    name: 'documents',
    query: { ...(value ? { status: value } : {}), page: '1' },
  })
}

function setOffset(offset: number): void {
  void router.push({
    name: 'documents',
    query: {
      ...(status.value ? { status: status.value } : {}),
      page: String(Math.floor(offset / PAGE_SIZE) + 1),
    },
  })
}

function statusLabel(value: DocumentStatus): string {
  return {
    pending: '等待处理',
    processing: '处理中',
    ready: '可用',
    failed: '失败',
    archived: '已归档',
  }[value]
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat('zh-CN', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

function documentJobLink(item: KnowledgeDocument) {
  return { name: 'jobs', query: { document_id: item.id } }
}

watch([status, page], () => void load(), { immediate: true })
onBeforeUnmount(() => controller?.abort())
</script>

<template>
  <main class="documents-page">
    <DocumentUpload @uploaded="load" />

    <section class="documents-collection" aria-labelledby="documents-title">
      <header class="documents-toolbar">
        <div>
          <h2 id="documents-title">已提交文档</h2>
          <p>列表来自服务端当前可查询范围。</p>
        </div>
        <div class="documents-toolbar__actions">
          <label>
            <span class="sr-only">文档状态</span>
            <select :value="status ?? ''" @change="setStatus">
              <option value="">所有状态</option>
              <option value="pending">等待处理</option>
              <option value="processing">处理中</option>
              <option value="ready">可用</option>
              <option value="failed">失败</option>
              <option value="archived">已归档</option>
            </select>
          </label>
          <IconButton label="刷新文档列表" @click="load">
            <RefreshCw :size="18" aria-hidden="true" />
          </IconButton>
        </div>
      </header>

      <InlineAlert
        v-if="state.status === 'error'"
        tone="error"
        title="无法加载文档"
      >
        <p>{{ state.error.message }}</p>
        <button class="inline-command" type="button" @click="load">重试</button>
      </InlineAlert>
      <LoadingState
        v-if="state.status === 'loading' && !state.data"
        label="正在加载文档"
      />
      <template v-else-if="state.data">
        <EmptyState v-if="!state.data.items.length" title="暂无文档" />
        <div v-else class="documents-table-wrap">
          <table class="documents-table">
            <thead>
              <tr>
                <th>文件名</th>
                <th>大小</th>
                <th>状态</th>
                <th>提交时间</th>
                <th aria-label="操作"></th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="item in state.data.items" :key="item.id">
                <td>{{ item.original_name }}</td>
                <td>{{ formatSize(item.size_bytes) }}</td>
                <td>{{ statusLabel(item.status) }}</td>
                <td>{{ formatDate(item.created_at) }}</td>
                <td>
                  <RouterLink :to="documentJobLink(item)">查看任务</RouterLink>
                </td>
              </tr>
            </tbody>
          </table>
          <PaginationControls
            :limit="PAGE_SIZE"
            :offset="(page - 1) * PAGE_SIZE"
            :total="state.data.total"
            @update:offset="setOffset"
          />
        </div>
      </template>
    </section>
  </main>
</template>

<style scoped>
.documents-page {
  display: grid;
  width: min(100%, 1180px);
  margin: 0 auto;
  padding: var(--space-6);
  gap: var(--space-6);
}

.documents-collection {
  min-width: 0;
}

.documents-toolbar {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: var(--space-4);
  margin-bottom: var(--space-4);
}

.documents-toolbar h2 {
  margin: 0;
  font-size: 18px;
  letter-spacing: 0;
}

.documents-toolbar p {
  margin: var(--space-1) 0 0;
  color: var(--color-text-secondary);
  font-size: 13px;
}

.documents-toolbar__actions {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

select {
  min-height: 38px;
  padding: 0 var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-neutral-0);
}

.documents-table-wrap {
  overflow-x: auto;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: var(--color-neutral-0);
}

.documents-table {
  width: 100%;
  min-width: 720px;
  border-collapse: collapse;
  font-size: 13px;
}

.documents-table th,
.documents-table td {
  padding: var(--space-3) var(--space-4);
  border-bottom: 1px solid var(--color-border);
  text-align: left;
}

.documents-table th {
  color: var(--color-text-secondary);
  background: var(--color-neutral-50);
  font-size: 12px;
}

.documents-table td:first-child {
  max-width: 360px;
  overflow: hidden;
  font-weight: 650;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.documents-table td:last-child {
  text-align: right;
}

.documents-table a,
.inline-command {
  color: var(--color-action);
  font-weight: 650;
}

.inline-command {
  margin-top: var(--space-2);
  padding: 0;
  border: 0;
  background: transparent;
}

.documents-table-wrap :deep(.pagination) {
  padding: var(--space-2) var(--space-3);
}

@media (max-width: 760px) {
  .documents-page {
    padding: var(--space-4);
  }

  .documents-toolbar {
    align-items: flex-start;
  }
}
</style>
