<script setup lang="ts">
import { Inbox } from '@lucide/vue'

withDefaults(
  defineProps<{
    description?: string
    title: string
  }>(),
  { description: '' },
)
</script>

<template>
  <section class="empty-state">
    <Inbox :size="28" aria-hidden="true" />
    <h2>{{ title }}</h2>
    <p v-if="description">{{ description }}</p>
    <div v-if="$slots.action" class="empty-state__action">
      <slot name="action" />
    </div>
  </section>
</template>

<style scoped>
.empty-state {
  display: grid;
  min-height: 220px;
  place-items: center;
  align-content: center;
  padding: var(--space-8) var(--space-4);
  color: var(--color-text-secondary);
  text-align: center;
}

.empty-state h2 {
  margin: var(--space-3) 0 var(--space-1);
  color: var(--color-text-primary);
  font-size: 17px;
  letter-spacing: 0;
}

.empty-state p {
  max-width: 460px;
  margin: 0;
  font-size: 14px;
}

.empty-state__action {
  margin-top: var(--space-4);
}
</style>
