<script setup lang="ts">
import { X } from '@lucide/vue'
import { ref } from 'vue'

import IconButton from '../../components/IconButton.vue'

const props = defineProps<{ modelValue: string[] }>()
const emit = defineEmits<{ 'update:modelValue': [values: string[]] }>()
const input = ref('')

function add(): void {
  const candidates = input.value
    .split(/[,，]/)
    .map((value) => value.trim())
    .filter(Boolean)
  if (candidates.length) {
    emit('update:modelValue', [
      ...new Set([...props.modelValue, ...candidates]),
    ])
  }
  input.value = ''
}

function remove(value: string): void {
  emit(
    'update:modelValue',
    props.modelValue.filter((item) => item !== value),
  )
}
</script>

<template>
  <div class="keyword-input">
    <div v-if="modelValue.length" class="keyword-input__tokens">
      <span v-for="keyword in modelValue" :key="keyword">
        <span data-testid="keyword-token">{{ keyword }}</span>
        <IconButton :label="`删除关键词“${keyword}”`" @click="remove(keyword)">
          <X :size="13" aria-hidden="true" />
        </IconButton>
      </span>
    </div>
    <label class="sr-only" for="knowledge-keyword">添加关键词</label>
    <input
      id="knowledge-keyword"
      v-model="input"
      type="text"
      placeholder="输入后按回车"
      @blur="add"
      @keydown.enter.prevent="add"
    />
  </div>
</template>

<style scoped>
.keyword-input {
  display: grid;
  gap: var(--space-2);
}

.keyword-input__tokens {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
}

.keyword-input__tokens > span {
  display: inline-flex;
  min-height: 30px;
  align-items: center;
  gap: var(--space-1);
  padding-left: var(--space-2);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-neutral-50);
  font-size: 13px;
}

.keyword-input__tokens :deep(.icon-button) {
  width: 28px;
  height: 28px;
}

.keyword-input > input {
  min-height: 40px;
  padding: 0 var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-neutral-0);
}
</style>
