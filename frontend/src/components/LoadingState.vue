<script setup lang="ts">
import { LoaderCircle } from '@lucide/vue'

withDefaults(defineProps<{ label?: string }>(), {
  label: '正在加载',
})
</script>

<template>
  <div class="loading-state" role="status" aria-live="polite">
    <LoaderCircle class="loading-state__icon" :size="22" aria-hidden="true" />
    <span>{{ label }}</span>
  </div>
</template>

<style scoped>
.loading-state {
  display: flex;
  min-height: 160px;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  color: var(--color-text-secondary);
  font-size: 14px;
}

.loading-state__icon {
  animation: loading-state-spin 0.9s linear infinite;
}

@keyframes loading-state-spin {
  to {
    transform: rotate(360deg);
  }
}

@media (prefers-reduced-motion: reduce) {
  .loading-state__icon {
    animation: none;
  }
}
</style>
