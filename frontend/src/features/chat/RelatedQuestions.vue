<script setup lang="ts">
import { ArrowUpRight } from '@lucide/vue'
import { useId } from 'vue'

defineProps<{ questions: string[] }>()

const titleId = useId()

const emit = defineEmits<{
  select: [question: string]
}>()
</script>

<template>
  <section
    v-if="questions.length"
    class="related-questions"
    :aria-labelledby="titleId"
  >
    <h3 :id="titleId">继续探索</h3>
    <div class="related-questions__list">
      <button
        v-for="question in questions"
        :key="question"
        type="button"
        data-testid="related-question"
        @click="emit('select', question)"
      >
        <span>{{ question }}</span>
        <ArrowUpRight :size="16" aria-hidden="true" />
      </button>
    </div>
  </section>
</template>

<style scoped>
.related-questions {
  padding-top: var(--space-5);
  border-top: 1px solid var(--color-border);
}

.related-questions h3 {
  margin: 0 0 var(--space-3);
  font-size: 15px;
  letter-spacing: 0;
}

.related-questions__list {
  display: grid;
  gap: var(--space-2);
}

.related-questions button {
  display: flex;
  width: 100%;
  min-height: 40px;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3);
  padding: var(--space-2) var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  color: var(--color-text-primary);
  background: var(--color-neutral-0);
  text-align: left;
}

.related-questions button:hover {
  border-color: #aeb5c2;
  background: var(--color-neutral-50);
}

.related-questions button span {
  min-width: 0;
  overflow-wrap: anywhere;
}
</style>
