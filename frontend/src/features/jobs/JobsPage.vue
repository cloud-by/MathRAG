<script setup lang="ts">
import { RefreshCw, RotateCcw, Square } from '@lucide/vue'
import { computed, onBeforeUnmount, ref, shallowRef, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { ApiError } from '../../api/errors'
import EmptyState from '../../components/EmptyState.vue'
import IconButton from '../../components/IconButton.vue'
import InlineAlert from '../../components/InlineAlert.vue'
import LoadingState from '../../components/LoadingState.vue'
import PaginationControls from '../../components/PaginationControls.vue'
import { jobsApi } from './api'
import JobStatusBadge from './JobStatusBadge.vue'
import type {
  IngestionJob,
  IngestionJobPage,
  IngestionJobStatus,
  IngestionJobType,
} from './types'
import { useJobPolling } from './useJobPolling'

const LIMIT = 25
const STATUSES = new Set<IngestionJobStatus>([
  'pending',
  'running',
  'completed',
  'failed',
  'cancelled',
])
const JOB_TYPES = new Set<IngestionJobType>(['text', 'pdf', 'web', 'reindex'])
type QueryState =
  | { status: 'idle' | 'loading'; data: IngestionJobPage | null }
  | { status: 'success'; data: IngestionJobPage }
  | { status: 'error'; data: IngestionJobPage | null; error: ApiError }

const route = useRoute()
const router = useRouter()
const state = shallowRef<QueryState>({ status: 'idle', data: null })
const documentInput = ref(
  typeof route.query.document_id === 'string' ? route.query.document_id : '',
)
const mutating = ref(new Set<string>())
const mutationError = ref<ApiError | null>(null)
let controller: AbortController | null = null
let sequence = 0

const status = computed<IngestionJobStatus | undefined>(() => {
  const value = route.query.status
  return typeof value === 'string' && STATUSES.has(value as IngestionJobStatus)
    ? (value as IngestionJobStatus)
    : undefined
})
const jobType = computed<IngestionJobType | undefined>(() => {
  const value = route.query.job_type
  return typeof value === 'string' && JOB_TYPES.has(value as IngestionJobType)
    ? (value as IngestionJobType)
    : undefined
})
const documentId = computed(() =>
  typeof route.query.document_id === 'string' && route.query.document_id.trim()
    ? route.query.document_id.trim()
    : undefined,
)
const offset = computed(() => {
  const value = Number(route.query.offset)
  return Number.isInteger(value) && value >= 0 ? value : 0
})

function asApiError(error: unknown): ApiError {
  if (error instanceof ApiError) return error
  return new ApiError({
    status: 0,
    code: 'NETWORK_ERROR',
    message: '请求失败，请稍后重试。',
    requestId: 'unavailable',
    details: null,
  })
}

function replaceJob(updated: IngestionJob): void {
  const data = state.value.data
  if (!data) return
  const items = data.items.map((item) =>
    item.id === updated.id ? updated : item,
  )
  state.value = { status: 'success', data: { ...data, items } }
  polling.sync(items)
}

const polling = useJobPolling({
  fetchJob: jobsApi.get,
  onUpdate: replaceJob,
})

async function load(): Promise<void> {
  controller?.abort()
  controller = new AbortController()
  const request = ++sequence
  const previous = state.value.data
  state.value = { status: 'loading', data: previous }
  try {
    const data = await jobsApi.list(
      {
        status: status.value,
        jobType: jobType.value,
        documentId: documentId.value,
        offset: offset.value,
        limit: LIMIT,
      },
      controller.signal,
    )
    if (request === sequence) {
      state.value = { status: 'success', data }
      polling.sync(data.items)
    }
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

function updateFilters(values: {
  status?: IngestionJobStatus
  jobType?: IngestionJobType
  documentId?: string
  offset?: number
}): void {
  void router.push({
    name: 'jobs',
    query: {
      ...(values.status ? { status: values.status } : {}),
      ...(values.jobType ? { job_type: values.jobType } : {}),
      ...(values.documentId ? { document_id: values.documentId } : {}),
      offset: String(values.offset ?? 0),
    },
  })
}

function setStatus(event: Event): void {
  const value = (event.target as HTMLSelectElement).value
  updateFilters({
    status: value ? (value as IngestionJobStatus) : undefined,
    jobType: jobType.value,
    documentId: documentId.value,
  })
}

function setJobType(event: Event): void {
  const value = (event.target as HTMLSelectElement).value
  updateFilters({
    status: status.value,
    jobType: value ? (value as IngestionJobType) : undefined,
    documentId: documentId.value,
  })
}

function applyDocumentFilter(): void {
  updateFilters({
    status: status.value,
    jobType: jobType.value,
    documentId: documentInput.value.trim() || undefined,
  })
}

function setOffset(value: number): void {
  updateFilters({
    status: status.value,
    jobType: jobType.value,
    documentId: documentId.value,
    offset: value,
  })
}

async function mutate(
  item: IngestionJob,
  action: 'cancel' | 'retry',
): Promise<void> {
  if (mutating.value.has(item.id)) return
  mutating.value = new Set(mutating.value).add(item.id)
  mutationError.value = null
  try {
    await jobsApi[action](item.id)
    const [updated] = await Promise.all([jobsApi.get(item.id), load()])
    replaceJob(updated)
  } catch (error) {
    mutationError.value = asApiError(error)
    await load()
  } finally {
    const next = new Set(mutating.value)
    next.delete(item.id)
    mutating.value = next
  }
}

function formatDate(value: string | null): string {
  if (!value) return '—'
  return new Intl.DateTimeFormat('zh-CN', {
    dateStyle: 'short',
    timeStyle: 'short',
  }).format(new Date(value))
}

watch(
  [status, jobType, documentId, offset],
  () => {
    documentInput.value = documentId.value ?? ''
    void load()
  },
  { immediate: true },
)
onBeforeUnmount(() => controller?.abort())
</script>

<template>
  <main class="jobs-page">
    <header class="jobs-toolbar">
      <form class="jobs-filters" @submit.prevent="applyDocumentFilter">
        <label>
          <span>状态</span>
          <select :value="status ?? ''" @change="setStatus">
            <option value="">所有状态</option>
            <option value="pending">等待中</option>
            <option value="running">处理中</option>
            <option value="completed">已完成</option>
            <option value="failed">失败</option>
            <option value="cancelled">已取消</option>
          </select>
        </label>
        <label>
          <span>任务类型</span>
          <select :value="jobType ?? ''" @change="setJobType">
            <option value="">全部类型</option>
            <option value="pdf">PDF</option>
            <option value="text">文本</option>
            <option value="web">网页</option>
            <option value="reindex">重建索引</option>
          </select>
        </label>
        <label class="jobs-filters__document">
          <span>文档 ID</span>
          <input
            v-model="documentInput"
            type="search"
            placeholder="精确文档 ID"
          />
        </label>
        <button class="secondary-command" type="submit">应用筛选</button>
      </form>
      <IconButton label="刷新摄取任务" @click="load">
        <RefreshCw :size="18" aria-hidden="true" />
      </IconButton>
    </header>

    <InlineAlert v-if="mutationError" tone="error" title="操作未完成">
      <p>{{ mutationError.message }}</p>
      <small>请求编号：{{ mutationError.requestId }}</small>
    </InlineAlert>
    <InlineAlert
      v-if="state.status === 'error'"
      tone="error"
      title="无法加载摄取任务"
    >
      <p>{{ state.error.message }}</p>
      <button class="inline-command" type="button" @click="load">重试</button>
    </InlineAlert>
    <LoadingState
      v-if="state.status === 'loading' && !state.data"
      label="正在加载摄取任务"
    />
    <template v-else-if="state.data">
      <EmptyState v-if="!state.data.items.length" title="没有符合条件的任务" />
      <section v-else class="jobs-table-wrap" aria-label="摄取任务列表">
        <table class="jobs-table">
          <thead>
            <tr>
              <th>任务</th>
              <th>类型</th>
              <th>状态</th>
              <th>进度</th>
              <th>尝试</th>
              <th>错误</th>
              <th>更新时间</th>
              <th aria-label="操作"></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="item in state.data.items" :key="item.id">
              <td class="job-id" :title="item.id">{{ item.id }}</td>
              <td>{{ item.job_type }}</td>
              <td><JobStatusBadge :status="item.status" /></td>
              <td>
                <div class="job-progress">
                  <progress :value="item.progress" max="100" />
                  <span>{{ item.progress }}%</span>
                </div>
              </td>
              <td>{{ item.attempt_count }}</td>
              <td class="job-error">
                <strong v-if="item.error_code">{{ item.error_code }}</strong>
                <span v-if="item.error_message">{{ item.error_message }}</span>
                <span v-if="!item.error_code && !item.error_message">—</span>
              </td>
              <td>{{ formatDate(item.updated_at) }}</td>
              <td class="job-actions">
                <IconButton
                  v-if="item.status === 'pending'"
                  :label="`取消任务 ${item.id}`"
                  :disabled="mutating.has(item.id)"
                  @click="mutate(item, 'cancel')"
                >
                  <Square :size="16" aria-hidden="true" />
                </IconButton>
                <IconButton
                  v-if="item.status === 'failed'"
                  :label="`重试任务 ${item.id}`"
                  :disabled="mutating.has(item.id)"
                  @click="mutate(item, 'retry')"
                >
                  <RotateCcw :size="16" aria-hidden="true" />
                </IconButton>
              </td>
            </tr>
          </tbody>
        </table>
        <PaginationControls
          :limit="LIMIT"
          :offset="offset"
          :total="state.data.total"
          @update:offset="setOffset"
        />
      </section>
    </template>
  </main>
</template>

<style scoped>
.jobs-page {
  width: 100%;
  min-width: 0;
  max-width: 1240px;
  margin: 0 auto;
  padding: var(--space-6);
}

.jobs-toolbar,
.jobs-filters,
.job-progress,
.job-actions {
  display: flex;
  align-items: center;
}

.jobs-toolbar {
  align-items: flex-end;
  justify-content: space-between;
  gap: var(--space-4);
  margin-bottom: var(--space-5);
}

.jobs-filters {
  flex: 1;
  flex-wrap: wrap;
  align-items: flex-end;
  gap: var(--space-3);
}

.jobs-filters label {
  display: grid;
  gap: var(--space-1);
  color: var(--color-text-secondary);
  font-size: 12px;
}

.jobs-filters select,
.jobs-filters input,
.secondary-command {
  min-height: 38px;
  padding: 0 var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-neutral-0);
}

.jobs-filters__document input {
  width: 250px;
}

.secondary-command {
  font-weight: 650;
}

.jobs-table-wrap {
  width: 100%;
  min-width: 0;
  max-width: 100%;
  overflow-x: auto;
  contain: inline-size;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: var(--color-neutral-0);
}

.jobs-table {
  width: 100%;
  min-width: 1060px;
  border-collapse: collapse;
  font-size: 13px;
}

.jobs-table th,
.jobs-table td {
  padding: var(--space-3);
  border-bottom: 1px solid var(--color-border);
  text-align: left;
  vertical-align: middle;
}

.jobs-table th {
  color: var(--color-text-secondary);
  background: var(--color-neutral-50);
  font-size: 12px;
}

.job-id {
  width: 150px;
  max-width: 150px;
  overflow: hidden;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.job-progress {
  gap: var(--space-2);
}

.job-progress progress {
  width: 80px;
  height: 6px;
  accent-color: var(--color-action);
}

.job-error {
  display: grid;
  width: 180px;
  max-width: 180px;
  gap: var(--space-1);
}

.job-error span {
  overflow: hidden;
  color: var(--color-text-secondary);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.job-actions {
  justify-content: flex-end;
  gap: var(--space-1);
}

.jobs-table-wrap :deep(.pagination) {
  padding: var(--space-2) var(--space-3);
}

.inline-command {
  margin-top: var(--space-2);
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
  .jobs-page {
    padding: var(--space-4);
  }

  .jobs-toolbar {
    align-items: flex-end;
  }

  .jobs-filters label,
  .jobs-filters__document input {
    min-width: 0;
    flex: 1 1 140px;
    width: auto;
  }
}
</style>
