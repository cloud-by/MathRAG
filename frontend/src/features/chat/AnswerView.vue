<script setup lang="ts">
import { ChevronRight } from '@lucide/vue'

import MathContent from '../../components/MathContent.vue'
import ReasoningSteps from './ReasoningSteps.vue'
import ReferenceList from './ReferenceList.vue'
import RelatedQuestions from './RelatedQuestions.vue'
import type { AnswerContent } from './types'

defineProps<{ answer: AnswerContent }>()

const emit = defineEmits<{
  selectRelated: [question: string]
}>()
</script>

<template>
  <article class="answer-view">
    <section class="answer-view__main" data-testid="answer-main">
      <h2>回答</h2>
      <MathContent :content="answer.answer" />
    </section>

    <section
      v-if="answer.used_knowledge?.length"
      class="answer-view__knowledge"
    >
      <h3>本次使用的知识</h3>
      <ul>
        <li v-for="item in answer.used_knowledge" :key="item">{{ item }}</li>
      </ul>
    </section>

    <ReasoningSteps
      :steps="answer.steps"
      :reasoning="answer.reasoning_content"
    />

    <details
      v-if="answer.agentic_plan"
      class="answer-view__agentic"
      data-testid="agentic-details"
    >
      <summary>
        <ChevronRight :size="17" aria-hidden="true" />
        <span>检索规划</span>
      </summary>
      <div class="answer-view__agentic-content">
        <MathContent :content="answer.agentic_plan.strategy" />
        <ul v-if="answer.agentic_plan.retrieval_queries?.length">
          <li
            v-for="query in answer.agentic_plan.retrieval_queries"
            :key="query"
          >
            <MathContent :content="query" />
          </li>
        </ul>
      </div>
    </details>

    <ReferenceList
      v-if="answer.references?.length"
      :references="answer.references"
    />
    <RelatedQuestions
      v-if="answer.related_questions?.length"
      :questions="answer.related_questions"
      @select="emit('selectRelated', $event)"
    />
  </article>
</template>

<style scoped>
.answer-view {
  min-width: 0;
  max-width: 860px;
  color: var(--color-text-primary);
}

.answer-view__main h2 {
  margin: 0 0 var(--space-3);
  font-size: 16px;
  letter-spacing: 0;
}

.answer-view__knowledge {
  margin: var(--space-5) 0;
}

.answer-view__knowledge h3 {
  margin: 0 0 var(--space-2);
  color: var(--color-text-secondary);
  font-size: 13px;
  letter-spacing: 0;
}

.answer-view__knowledge ul {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2) var(--space-4);
  margin: 0;
  padding: 0;
  list-style: none;
}

.answer-view__knowledge li {
  position: relative;
  padding-left: var(--space-3);
  font-size: 13px;
}

.answer-view__knowledge li::before {
  position: absolute;
  top: 0.65em;
  left: 0;
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: var(--color-success);
  content: '';
}

.answer-view__agentic {
  border-top: 1px solid var(--color-border);
}

.answer-view__agentic summary {
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

.answer-view__agentic summary::-webkit-details-marker {
  display: none;
}

.answer-view__agentic summary svg {
  transition: transform 160ms ease;
}

.answer-view__agentic[open] summary svg {
  transform: rotate(90deg);
}

.answer-view__agentic-content {
  margin: 0 0 var(--space-5) 28px;
  padding-left: var(--space-4);
  border-left: 2px solid var(--color-neutral-200);
  color: var(--color-text-secondary);
  font-size: 13px;
}

.answer-view__agentic-content ul {
  margin: var(--space-2) 0 0;
  padding-left: var(--space-5);
}
</style>
