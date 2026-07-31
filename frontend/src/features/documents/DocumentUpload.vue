<script setup lang="ts">
import { FileUp, Upload } from '@lucide/vue'
import { computed, ref } from 'vue'
import { RouterLink } from 'vue-router'

import { ApiError } from '../../api/errors'
import { documentsApi } from './api'
import type { DocumentAccepted } from './types'

type QueueStatus = 'queued' | 'uploading' | 'success' | 'error'

interface QueueItem {
  key: string
  file: File
  status: QueueStatus
  result?: DocumentAccepted
  error?: ApiError
}

const emit = defineEmits<{ uploaded: [accepted: DocumentAccepted] }>()
const category = ref('')
const queue = ref<QueueItem[]>([])
const busy = computed(() =>
  queue.value.some((item) => item.status === 'uploading'),
)
const canUpload = computed(
  () => !busy.value && queue.value.some((item) => item.status === 'queued'),
)

function asApiError(error: unknown): ApiError {
  if (error instanceof ApiError) return error
  return new ApiError({
    status: 0,
    code: 'NETWORK_ERROR',
    message: '上传失败，请检查网络后重试。',
    requestId: 'unavailable',
    details: null,
  })
}

function selectFiles(event: Event): void {
  const input = event.target as HTMLInputElement
  const files = Array.from(input.files ?? [])
  queue.value = files.map((file, index) => ({
    key: `${file.name}-${file.size}-${file.lastModified}-${index}`,
    file,
    status: 'queued',
  }))
}

async function uploadAll(): Promise<void> {
  for (const item of queue.value) {
    if (item.status !== 'queued') continue
    item.status = 'uploading'
    item.error = undefined
    try {
      const accepted = await documentsApi.upload(item.file, category.value)
      item.result = accepted
      item.status = 'success'
      emit('uploaded', accepted)
    } catch (error) {
      item.error = asApiError(error)
      item.status = 'error'
    }
  }
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
</script>

<template>
  <section class="document-upload" aria-labelledby="document-upload-title">
    <div class="document-upload__heading">
      <FileUp :size="22" aria-hidden="true" />
      <div>
        <h2 id="document-upload-title">上传 PDF 文档</h2>
        <p>文档提交后会创建摄取任务。</p>
      </div>
    </div>
    <div class="document-upload__controls">
      <label class="file-control">
        <span>选择 PDF 文档</span>
        <input
          type="file"
          accept="application/pdf,.pdf"
          multiple
          :disabled="busy"
          @change="selectFiles"
        />
      </label>
      <label class="category-control">
        <span>知识类别（可选）</span>
        <input v-model="category" type="text" :disabled="busy" />
      </label>
      <button
        class="upload-command"
        type="button"
        :disabled="!canUpload"
        @click="uploadAll"
      >
        <Upload :size="18" aria-hidden="true" />
        开始上传
      </button>
    </div>
    <ul v-if="queue.length" class="upload-queue" aria-label="上传队列">
      <li v-for="item in queue" :key="item.key">
        <div>
          <strong>{{ item.file.name }}</strong>
          <span>{{ formatSize(item.file.size) }}</span>
        </div>
        <span v-if="item.status === 'queued'">等待上传</span>
        <span v-else-if="item.status === 'uploading'" role="status"
          >提交中</span
        >
        <span v-else-if="item.status === 'success'" class="upload-success">
          上传成功
        </span>
        <div v-else class="upload-error" role="alert">
          <span>{{ item.error?.message }}</span>
          <small v-if="item.error">请求编号：{{ item.error.requestId }}</small>
        </div>
        <RouterLink
          v-if="item.result"
          :aria-label="`查看 ${item.file.name} 的摄取任务`"
          :to="{
            name: 'jobs',
            query: { document_id: item.result.document.id },
          }"
        >
          查看任务
        </RouterLink>
      </li>
    </ul>
  </section>
</template>

<style scoped>
.document-upload {
  padding: var(--space-5);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: var(--color-neutral-0);
}

.document-upload__heading,
.document-upload__controls,
.upload-queue li {
  display: flex;
  align-items: center;
}

.document-upload__heading {
  gap: var(--space-3);
}

.document-upload h2 {
  margin: 0;
  font-size: 17px;
  letter-spacing: 0;
}

.document-upload p {
  margin: var(--space-1) 0 0;
  color: var(--color-text-secondary);
  font-size: 13px;
}

.document-upload__controls {
  align-items: flex-end;
  gap: var(--space-3);
  margin-top: var(--space-4);
}

.file-control,
.category-control {
  display: grid;
  gap: var(--space-1);
  color: var(--color-text-secondary);
  font-size: 12px;
}

.file-control {
  flex: 1 1 320px;
}

.category-control {
  flex: 0 1 220px;
}

input {
  min-width: 0;
  min-height: 38px;
  padding: var(--space-2) var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-neutral-0);
}

.upload-command {
  display: inline-flex;
  min-height: 38px;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  padding: 0 var(--space-4);
  border: 1px solid var(--color-action);
  border-radius: var(--radius-md);
  color: var(--color-neutral-0);
  background: var(--color-action);
  font-weight: 650;
}

.upload-command:disabled {
  border-color: var(--color-neutral-200);
  color: var(--color-text-secondary);
  background: var(--color-neutral-100);
}

.upload-queue {
  display: grid;
  gap: 0;
  margin: var(--space-4) 0 0;
  padding: 0;
  border-top: 1px solid var(--color-border);
  list-style: none;
}

.upload-queue li {
  min-height: 54px;
  gap: var(--space-3);
  padding: var(--space-2) 0;
  border-bottom: 1px solid var(--color-border);
  font-size: 13px;
}

.upload-queue li > div:first-child {
  display: grid;
  min-width: 0;
  flex: 1;
}

.upload-queue strong {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.upload-queue span,
.upload-queue small {
  color: var(--color-text-secondary);
}

.upload-queue a,
.upload-success {
  color: var(--color-success);
  font-weight: 650;
}

.upload-error {
  display: grid;
  color: var(--color-error);
}

.upload-error span {
  color: inherit;
}

@media (max-width: 760px) {
  .document-upload {
    padding: var(--space-4);
  }

  .document-upload__controls {
    align-items: stretch;
    flex-direction: column;
  }

  .file-control,
  .category-control {
    width: 100%;
    flex: none;
  }

  .upload-queue li {
    align-items: flex-start;
    flex-wrap: wrap;
  }
}
</style>
