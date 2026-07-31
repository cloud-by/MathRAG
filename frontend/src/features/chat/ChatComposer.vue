<script setup lang="ts">
import { Minus, Plus, Send, Square } from '@lucide/vue'
import { computed, ref } from 'vue'

import IconButton from '../../components/IconButton.vue'
import type { ChatStatus } from './types'

const props = defineProps<{
  modelValue: string
  state: ChatStatus
  topK: number
}>()

const emit = defineEmits<{
  cancel: []
  submit: []
  'update:modelValue': [value: string]
  'update:topK': [value: number]
}>()

const textarea = ref<HTMLTextAreaElement | null>(null)
const composing = ref(false)
const submitting = computed(() => props.state === 'submitting')
const canSubmit = computed(
  () => !submitting.value && props.modelValue.trim().length > 0,
)
const question = computed({
  get: () => props.modelValue,
  set: (value: string) => emit('update:modelValue', value),
})
const retrievalCount = computed({
  get: () => props.topK,
  set: (value: number) => setTopK(value),
})

function setTopK(value: number): void {
  if (!Number.isFinite(value)) return
  emit('update:topK', Math.min(10, Math.max(1, Math.round(value))))
}

function onKeydown(event: KeyboardEvent): void {
  if (
    event.key !== 'Enter' ||
    event.shiftKey ||
    composing.value ||
    event.isComposing
  ) {
    return
  }
  event.preventDefault()
  if (canSubmit.value) emit('submit')
}

defineExpose({ focus: () => textarea.value?.focus() })
</script>

<template>
  <section class="chat-composer" aria-label="提问输入">
    <label class="sr-only" for="chat-question">数学问题</label>
    <textarea
      id="chat-question"
      ref="textarea"
      v-model="question"
      rows="3"
      maxlength="8000"
      :disabled="submitting"
      placeholder="输入数学问题"
      @compositionstart="composing = true"
      @compositionend="composing = false"
      @keydown="onKeydown"
    />
    <footer>
      <div class="top-k-control">
        <span>检索数量</span>
        <IconButton
          label="减少检索数量"
          :disabled="submitting || topK <= 1"
          @click="setTopK(topK - 1)"
        >
          <Minus :size="16" aria-hidden="true" />
        </IconButton>
        <label class="sr-only" for="chat-top-k">检索知识条数</label>
        <input
          id="chat-top-k"
          v-model.number="retrievalCount"
          type="number"
          min="1"
          max="10"
          :disabled="submitting"
        />
        <IconButton
          label="增加检索数量"
          :disabled="submitting || topK >= 10"
          @click="setTopK(topK + 1)"
        >
          <Plus :size="16" aria-hidden="true" />
        </IconButton>
      </div>

      <IconButton
        v-if="submitting"
        class="chat-composer__stop"
        label="停止回答"
        @click="emit('cancel')"
      >
        <Square :size="17" aria-hidden="true" />
      </IconButton>
      <IconButton
        v-else
        class="chat-composer__send"
        label="发送问题"
        :disabled="!canSubmit"
        @click="emit('submit')"
      >
        <Send :size="18" aria-hidden="true" />
      </IconButton>
    </footer>
  </section>
</template>

<style scoped>
.chat-composer {
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: var(--color-neutral-0);
  box-shadow: 0 8px 24px rgb(23 32 51 / 6%);
}

.chat-composer textarea {
  display: block;
  width: 100%;
  min-height: 92px;
  resize: vertical;
  padding: var(--space-4);
  border: 0;
  outline: 0;
  background: transparent;
  line-height: 1.6;
}

.chat-composer textarea::placeholder {
  color: #8a93a3;
}

.chat-composer footer {
  display: flex;
  min-height: 48px;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3);
  padding: var(--space-2);
  border-top: 1px solid var(--color-neutral-100);
}

.top-k-control {
  display: flex;
  align-items: center;
  gap: var(--space-1);
  color: var(--color-text-secondary);
  font-size: 12px;
}

.top-k-control > span {
  margin-right: var(--space-1);
}

.top-k-control input {
  width: 42px;
  height: 32px;
  padding: 0;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-neutral-0);
  font-variant-numeric: tabular-nums;
  text-align: center;
}

.chat-composer__send {
  color: var(--color-neutral-0);
  background: var(--color-action);
}

.chat-composer__send:hover:not(:disabled) {
  color: var(--color-neutral-0);
  background: var(--color-action-hover);
}

.chat-composer__stop {
  color: var(--color-error);
  border-color: #e4b5b8;
  background: #fff4f4;
}

@media (max-width: 460px) {
  .top-k-control > span {
    display: none;
  }
}
</style>
