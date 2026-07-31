<script setup lang="ts">
import { ChevronRight } from '@lucide/vue'

import MathContent from '../../components/MathContent.vue'

withDefaults(
  defineProps<{
    reasoning?: string | null
    steps?: string[]
  }>(),
  {
    reasoning: null,
    steps: () => [],
  },
)
</script>

<template>
  <details
    v-if="steps.length || reasoning"
    class="reasoning-steps"
    data-testid="reasoning-details"
  >
    <summary>
      <ChevronRight
        class="reasoning-steps__chevron"
        :size="17"
        aria-hidden="true"
      />
      <span>解题过程</span>
    </summary>
    <ol v-if="steps.length">
      <li v-for="(step, index) in steps" :key="`${index}-${step}`">
        <MathContent :content="step" />
      </li>
    </ol>
    <div v-if="reasoning" class="reasoning-steps__additional">
      <h4>补充说明</h4>
      <MathContent :content="reasoning" />
    </div>
  </details>
</template>

<style scoped>
.reasoning-steps {
  border-top: 1px solid var(--color-border);
}

.reasoning-steps summary {
  display: flex;
  min-height: 46px;
  align-items: center;
  gap: var(--space-2);
  color: var(--color-text-secondary);
  font-size: 14px;
  font-weight: 600;
  list-style: none;
  cursor: pointer;
}

.reasoning-steps summary::-webkit-details-marker {
  display: none;
}

.reasoning-steps__chevron {
  transition: transform 160ms ease;
}

.reasoning-steps[open] .reasoning-steps__chevron {
  transform: rotate(90deg);
}

.reasoning-steps ol {
  margin: 0 0 var(--space-5) 28px;
  padding-left: var(--space-4);
}

.reasoning-steps li {
  margin-bottom: var(--space-3);
  padding-left: var(--space-1);
}

.reasoning-steps__additional {
  margin: 0 0 var(--space-5) 28px;
  padding-left: var(--space-4);
  border-left: 2px solid var(--color-neutral-200);
}

.reasoning-steps__additional h4 {
  margin: 0 0 var(--space-2);
  font-size: 13px;
}
</style>
