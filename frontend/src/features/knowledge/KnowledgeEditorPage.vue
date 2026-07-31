<script setup lang="ts">
import { ArrowLeft, Save } from '@lucide/vue'
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { onBeforeRouteLeave, RouterLink, useRoute, useRouter } from 'vue-router'

import InlineAlert from '../../components/InlineAlert.vue'
import LoadingState from '../../components/LoadingState.vue'
import MathContent from '../../components/MathContent.vue'
import KeywordInput from './KeywordInput.vue'
import StepEditor from './StepEditor.vue'
import type { KnowledgeDraft, KnowledgeItem, KnowledgeUpdate } from './types'
import { draftFromItem, emptyKnowledgeDraft, normalizedDraft } from './types'
import { fieldErrorsFrom, useKnowledgeEditor } from './useKnowledge'

const route = useRoute()
const router = useRouter()
const editor = useKnowledgeEditor()
const draft = reactive<KnowledgeDraft>(emptyKnowledgeDraft())
const baseline = ref(JSON.stringify(normalizedDraft(emptyKnowledgeDraft())))
const revision = ref<number | null>(null)
const saving = ref(false)
const mutationError = ref<Error | null>(null)
const requestId = ref('')
const fieldErrors = ref<Record<string, string>>({})

const isNew = computed(() => route.name === 'knowledge-new')
const itemId = computed(() =>
  isNew.value ? '' : String(route.params.id ?? ''),
)
const dirty = computed(
  () => JSON.stringify(normalizedDraft(draft)) !== baseline.value,
)

function replaceDraft(values: KnowledgeDraft): void {
  Object.assign(draft, {
    ...values,
    keywords: [...values.keywords],
    steps: [...values.steps],
  })
}

function adopt(item: KnowledgeItem): void {
  const values = draftFromItem(item)
  replaceDraft(values)
  baseline.value = JSON.stringify(normalizedDraft(values))
  revision.value = item.revision
  mutationError.value = null
  requestId.value = ''
  fieldErrors.value = {}
  editor.clearConflict()
}

function resetNew(): void {
  const values = emptyKnowledgeDraft()
  replaceDraft(values)
  baseline.value = JSON.stringify(normalizedDraft(values))
  revision.value = null
  mutationError.value = null
  requestId.value = ''
  fieldErrors.value = {}
  editor.clearConflict()
}

function validate(values: KnowledgeDraft): boolean {
  const errors: Record<string, string> = {}
  if (!values.title) errors.title = '标题不能为空。'
  if (!values.category) errors.category = '类别不能为空。'
  if (!values.content) errors.content = '知识内容不能为空。'
  if (!values.keywords.length) errors.keywords = '至少添加一个关键词。'
  if (!values.steps.length) errors.steps = '至少添加一个步骤。'
  fieldErrors.value = errors
  return Object.keys(errors).length === 0
}

function updatePayload(values: KnowledgeDraft): KnowledgeUpdate {
  const original = JSON.parse(baseline.value) as KnowledgeDraft
  const payload: KnowledgeUpdate = { revision: revision.value ?? 1 }
  if (values.category !== original.category) payload.category = values.category
  if (values.title !== original.title) payload.title = values.title
  if (JSON.stringify(values.keywords) !== JSON.stringify(original.keywords)) {
    payload.keywords = values.keywords
  }
  if (values.content !== original.content) payload.content = values.content
  if (values.example !== original.example) payload.example = values.example
  if (JSON.stringify(values.steps) !== JSON.stringify(original.steps)) {
    payload.steps = values.steps
  }
  if (values.difficulty !== original.difficulty) {
    payload.difficulty = values.difficulty
  }
  if (values.visibility !== original.visibility) {
    payload.visibility = values.visibility
  }
  return payload
}

async function save(): Promise<void> {
  const values = normalizedDraft(draft)
  if (!validate(values)) return
  saving.value = true
  mutationError.value = null
  requestId.value = ''
  fieldErrors.value = {}
  const result = isNew.value
    ? await editor.create(values)
    : await editor.update(itemId.value, updatePayload(values))
  saving.value = false

  if (result.kind === 'saved') {
    adopt(result.item)
    if (isNew.value) {
      await router.replace({
        name: 'knowledge-detail',
        params: { id: result.item.id },
      })
    }
    return
  }
  mutationError.value = result.error
  requestId.value = result.error.requestId
  fieldErrors.value = fieldErrorsFrom(result.error)
}

function reloadServerVersion(): void {
  if (editor.conflict.value) adopt(editor.conflict.value)
}

async function reapplyDraft(): Promise<void> {
  const server = editor.conflict.value
  if (!server) return
  revision.value = server.revision
  baseline.value = JSON.stringify(normalizedDraft(draftFromItem(server)))
  editor.clearConflict()
  await save()
}

function onBeforeUnload(event: BeforeUnloadEvent): void {
  if (!dirty.value) return
  event.preventDefault()
  event.returnValue = ''
}

watch(
  [isNew, itemId],
  async ([creating, id]) => {
    if (creating) {
      resetNew()
      return
    }
    if (!id) return
    const item = await editor.load(id)
    if (item) adopt(item)
  },
  { immediate: true },
)

onBeforeRouteLeave(() =>
  !dirty.value ? true : window.confirm('当前更改尚未保存，确定离开吗？'),
)
onMounted(() => window.addEventListener('beforeunload', onBeforeUnload))
onBeforeUnmount(() =>
  window.removeEventListener('beforeunload', onBeforeUnload),
)
</script>

<template>
  <main class="knowledge-editor-page">
    <LoadingState
      v-if="editor.state.value.status === 'loading' && !editor.state.value.data"
      label="正在加载知识条目"
    />
    <InlineAlert
      v-else-if="editor.state.value.status === 'error'"
      tone="error"
      title="无法读取知识条目"
    >
      <p>{{ editor.state.value.error.message }}</p>
      <small>请求编号：{{ editor.state.value.error.requestId }}</small>
    </InlineAlert>
    <template v-else>
      <header class="knowledge-editor-page__header">
        <div>
          <RouterLink class="back-link" to="/knowledge">
            <ArrowLeft :size="16" aria-hidden="true" />
            返回知识库
          </RouterLink>
          <h2>{{ isNew ? '新建知识条目' : draft.title || '知识条目' }}</h2>
          <p v-if="revision">修订版本 {{ revision }}</p>
        </div>
        <button
          class="primary-command"
          type="button"
          :disabled="saving || (!isNew && !dirty)"
          @click="save"
        >
          <Save :size="18" aria-hidden="true" />
          {{ saving ? '正在保存' : isNew ? '创建知识条目' : '保存更改' }}
        </button>
      </header>

      <InlineAlert
        v-if="editor.conflict.value"
        tone="warning"
        title="服务器版本已更新"
      >
        <p>服务器版本：修订 {{ editor.conflict.value.revision }}</p>
        <div class="conflict-actions">
          <button type="button" @click="reloadServerVersion">
            重新载入服务器版本
          </button>
          <button type="button" @click="reapplyDraft">
            保留草稿后重新应用
          </button>
        </div>
      </InlineAlert>
      <InlineAlert v-else-if="mutationError" tone="error" title="保存失败">
        <p>{{ mutationError.message }}</p>
        <small v-if="requestId">请求编号：{{ requestId }}</small>
      </InlineAlert>

      <div class="knowledge-editor">
        <form class="knowledge-form" @submit.prevent="save">
          <div class="field-row field-row--split">
            <label>
              <span>标题</span>
              <input
                v-model="draft.title"
                type="text"
                maxlength="255"
                :aria-invalid="Boolean(fieldErrors.title)"
              />
              <small v-if="fieldErrors.title" class="field-error">{{
                fieldErrors.title
              }}</small>
            </label>
            <label>
              <span>类别</span>
              <input
                v-model="draft.category"
                type="text"
                maxlength="128"
                :aria-invalid="Boolean(fieldErrors.category)"
              />
              <small v-if="fieldErrors.category" class="field-error">{{
                fieldErrors.category
              }}</small>
            </label>
          </div>

          <div class="field-row field-row--split">
            <label>
              <span>难度</span>
              <select v-model="draft.difficulty">
                <option value="easy">简单</option>
                <option value="medium">中等</option>
                <option value="hard">困难</option>
              </select>
            </label>
            <label>
              <span>可见性</span>
              <select v-model="draft.visibility">
                <option value="public">公开</option>
                <option value="private">私有</option>
              </select>
            </label>
          </div>

          <div class="field-row">
            <span>关键词</span>
            <KeywordInput v-model="draft.keywords" />
            <small v-if="fieldErrors.keywords" class="field-error">{{
              fieldErrors.keywords
            }}</small>
          </div>

          <label class="field-row">
            <span>知识内容</span>
            <textarea
              v-model="draft.content"
              rows="9"
              :aria-invalid="Boolean(fieldErrors.content)"
            />
            <small v-if="fieldErrors.content" class="field-error">{{
              fieldErrors.content
            }}</small>
          </label>

          <label class="field-row">
            <span>示例</span>
            <textarea v-model="draft.example" rows="4" />
          </label>

          <div class="field-row">
            <span>解题步骤</span>
            <StepEditor v-model="draft.steps" />
            <small v-if="fieldErrors.steps" class="field-error">{{
              fieldErrors.steps
            }}</small>
          </div>
        </form>

        <aside class="knowledge-preview" aria-label="知识内容预览">
          <h3>预览</h3>
          <h4>{{ draft.title || '未命名条目' }}</h4>
          <p class="knowledge-preview__meta">
            {{ draft.category || '未分类' }} ·
            {{
              draft.difficulty === 'easy'
                ? '简单'
                : draft.difficulty === 'hard'
                  ? '困难'
                  : '中等'
            }}
          </p>
          <MathContent v-if="draft.content" :content="draft.content" />
          <template v-if="draft.example">
            <h4>示例</h4>
            <MathContent :content="draft.example" />
          </template>
          <template v-if="draft.steps.some((step) => step.trim())">
            <h4>解题步骤</h4>
            <ol>
              <li
                v-for="(step, index) in draft.steps.filter((value) =>
                  value.trim(),
                )"
                :key="index"
              >
                <MathContent :content="step" />
              </li>
            </ol>
          </template>
        </aside>
      </div>
    </template>
  </main>
</template>

<style scoped>
.knowledge-editor-page {
  width: min(100%, 1240px);
  margin: 0 auto;
  padding: var(--space-6);
}

.knowledge-editor-page__header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: var(--space-5);
  margin-bottom: var(--space-5);
}

.back-link,
.primary-command {
  display: flex;
  align-items: center;
}

.back-link {
  width: fit-content;
  gap: var(--space-1);
  color: var(--color-text-secondary);
  font-size: 13px;
  text-decoration: none;
}

.knowledge-editor-page__header h2 {
  margin: var(--space-2) 0 0;
  font-size: 24px;
  letter-spacing: 0;
}

.knowledge-editor-page__header p {
  margin: var(--space-1) 0 0;
  color: var(--color-text-secondary);
  font-size: 12px;
}

.primary-command {
  min-height: 40px;
  justify-content: center;
  gap: var(--space-2);
  padding: 0 var(--space-4);
  border: 1px solid var(--color-action);
  border-radius: var(--radius-md);
  color: var(--color-neutral-0);
  background: var(--color-action);
  font-weight: 650;
}

.knowledge-editor-page :deep(.inline-alert) {
  margin-bottom: var(--space-5);
}

.conflict-actions {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-3);
  margin-top: var(--space-3);
}

.conflict-actions button {
  padding: 0;
  border: 0;
  color: var(--color-action);
  background: transparent;
  font-weight: 650;
}

.knowledge-editor {
  display: grid;
  grid-template-columns: minmax(0, 3fr) minmax(300px, 2fr);
  gap: var(--space-8);
  align-items: start;
}

.knowledge-form {
  display: grid;
  gap: var(--space-5);
}

.field-row,
.field-row label {
  display: grid;
  gap: var(--space-2);
}

.field-row--split {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--space-4);
}

.field-row > span,
.field-row label > span,
.field-row--split > label > span {
  color: var(--color-text-secondary);
  font-size: 13px;
  font-weight: 650;
}

.knowledge-form input,
.knowledge-form select,
.knowledge-form textarea {
  width: 100%;
  min-height: 40px;
  padding: var(--space-2) var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-neutral-0);
}

.knowledge-form textarea {
  resize: vertical;
  line-height: 1.6;
}

.knowledge-form [aria-invalid='true'] {
  border-color: var(--color-error);
}

.field-error {
  color: var(--color-error);
  font-size: 12px;
}

.knowledge-preview {
  position: sticky;
  top: 82px;
  min-width: 0;
  padding-left: var(--space-6);
  border-left: 1px solid var(--color-border);
}

.knowledge-preview h3 {
  margin: 0 0 var(--space-5);
  color: var(--color-text-secondary);
  font-size: 13px;
  letter-spacing: 0;
}

.knowledge-preview h4 {
  margin: var(--space-5) 0 var(--space-2);
  font-size: 16px;
  letter-spacing: 0;
}

.knowledge-preview__meta {
  margin: 0 0 var(--space-4);
  color: var(--color-text-secondary);
  font-size: 12px;
}

.knowledge-preview ol {
  padding-left: var(--space-5);
}

small {
  display: block;
}

@media (max-width: 900px) {
  .knowledge-editor {
    grid-template-columns: minmax(0, 1fr);
  }

  .knowledge-preview {
    position: static;
    padding: var(--space-5) 0 0;
    border-top: 1px solid var(--color-border);
    border-left: 0;
  }
}

@media (max-width: 600px) {
  .knowledge-editor-page {
    padding: var(--space-4);
  }

  .knowledge-editor-page__header {
    align-items: stretch;
    flex-direction: column;
  }

  .field-row--split {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
