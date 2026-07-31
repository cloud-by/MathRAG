<script setup lang="ts">
import { ref } from 'vue'

withDefaults(
  defineProps<{
    disabled?: boolean
    label: string
    pressed?: boolean
    type?: 'button' | 'submit' | 'reset'
  }>(),
  {
    disabled: false,
    pressed: undefined,
    type: 'button',
  },
)

const button = ref<HTMLButtonElement | null>(null)

defineExpose({
  focus: () => button.value?.focus(),
})
</script>

<template>
  <button
    ref="button"
    class="icon-button"
    :type="type"
    :aria-label="label"
    :aria-pressed="pressed"
    :title="label"
    :disabled="disabled"
  >
    <slot />
  </button>
</template>

<style scoped>
.icon-button {
  display: inline-grid;
  flex: 0 0 36px;
  width: 36px;
  height: 36px;
  padding: 0;
  place-items: center;
  border: 1px solid transparent;
  border-radius: var(--radius-md);
  color: var(--color-text-secondary);
  background: transparent;
}

.icon-button:hover:not(:disabled) {
  border-color: var(--color-border);
  color: var(--color-text-primary);
  background: var(--color-neutral-100);
}

.icon-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
