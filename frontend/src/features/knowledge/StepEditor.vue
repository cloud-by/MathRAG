<script setup lang="ts">
import { ArrowDown, ArrowUp, Plus, Trash2 } from '@lucide/vue'

import IconButton from '../../components/IconButton.vue'

const props = defineProps<{ modelValue: string[] }>()
const emit = defineEmits<{ 'update:modelValue': [values: string[]] }>()

function update(index: number, value: string): void {
  const values = [...props.modelValue]
  values[index] = value
  emit('update:modelValue', values)
}

function add(): void {
  emit('update:modelValue', [...props.modelValue, ''])
}

function remove(index: number): void {
  const values = props.modelValue.filter((_, itemIndex) => itemIndex !== index)
  emit('update:modelValue', values.length ? values : [''])
}

function move(index: number, direction: -1 | 1): void {
  const target = index + direction
  if (target < 0 || target >= props.modelValue.length) return
  const values = [...props.modelValue]
  ;[values[index], values[target]] = [values[target]!, values[index]!]
  emit('update:modelValue', values)
}
</script>

<template>
  <div class="step-editor">
    <div
      v-for="(step, index) in modelValue"
      :key="index"
      class="step-editor__row"
    >
      <span class="step-editor__position">{{ index + 1 }}</span>
      <label class="sr-only" :for="`knowledge-step-${index}`"
        >步骤 {{ index + 1 }}</label
      >
      <input
        :id="`knowledge-step-${index}`"
        type="text"
        :value="step"
        @input="update(index, ($event.target as HTMLInputElement).value)"
      />
      <IconButton
        :label="`上移步骤 ${index + 1}`"
        :disabled="index === 0"
        @click="move(index, -1)"
      >
        <ArrowUp :size="16" aria-hidden="true" />
      </IconButton>
      <IconButton
        :label="`下移步骤 ${index + 1}`"
        :disabled="index === modelValue.length - 1"
        @click="move(index, 1)"
      >
        <ArrowDown :size="16" aria-hidden="true" />
      </IconButton>
      <IconButton :label="`删除步骤 ${index + 1}`" @click="remove(index)">
        <Trash2 :size="16" aria-hidden="true" />
      </IconButton>
    </div>
    <button class="secondary-command" type="button" @click="add">
      <Plus :size="17" aria-hidden="true" />
      添加步骤
    </button>
  </div>
</template>

<style scoped>
.step-editor {
  display: grid;
  gap: var(--space-2);
}

.step-editor__row {
  display: grid;
  grid-template-columns: 24px minmax(0, 1fr) 36px 36px 36px;
  align-items: center;
  gap: var(--space-1);
}

.step-editor__position {
  color: var(--color-text-secondary);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  text-align: center;
}

.step-editor__row input {
  min-width: 0;
  min-height: 40px;
  padding: 0 var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-neutral-0);
}

.secondary-command {
  display: inline-flex;
  width: fit-content;
  min-height: 36px;
  align-items: center;
  gap: var(--space-2);
  padding: 0 var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-neutral-0);
  font-weight: 600;
}

@media (max-width: 540px) {
  .step-editor__row {
    grid-template-columns: 20px minmax(0, 1fr) repeat(3, 32px);
  }
}
</style>
