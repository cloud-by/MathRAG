<script setup lang="ts">
import { AlertCircle, CircleCheck, Info, TriangleAlert } from '@lucide/vue'
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{
    title?: string
    tone?: 'error' | 'info' | 'success' | 'warning'
  }>(),
  {
    title: '',
    tone: 'info',
  },
)

const icon = computed(() => {
  if (props.tone === 'error') return AlertCircle
  if (props.tone === 'warning') return TriangleAlert
  if (props.tone === 'success') return CircleCheck
  return Info
})
</script>

<template>
  <section
    class="inline-alert"
    :class="`inline-alert--${tone}`"
    :role="tone === 'error' || tone === 'warning' ? 'alert' : 'status'"
  >
    <component :is="icon" :size="18" aria-hidden="true" />
    <div>
      <strong v-if="title">{{ title }}</strong>
      <div class="inline-alert__content"><slot /></div>
    </div>
  </section>
</template>

<style scoped>
.inline-alert {
  display: grid;
  grid-template-columns: 20px minmax(0, 1fr);
  gap: var(--space-3);
  padding: var(--space-3) var(--space-4);
  border-left: 3px solid var(--color-action);
  color: var(--color-text-primary);
  background: #f3f6ff;
  font-size: 14px;
}

.inline-alert--success {
  border-color: var(--color-success);
  background: #f1faf5;
}

.inline-alert--warning {
  border-color: var(--color-warning);
  background: #fff8eb;
}

.inline-alert--error {
  border-color: var(--color-error);
  background: #fff4f4;
}

.inline-alert strong {
  display: block;
  margin-bottom: var(--space-1);
}

.inline-alert__content :deep(p) {
  margin: 0;
}
</style>
