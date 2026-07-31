<script setup lang="ts">
import { useId } from 'vue'

import MathContent from '../../components/MathContent.vue'
import type { ReferenceItem } from './types'

defineProps<{ references: ReferenceItem[] }>()

const titleId = useId()

function sourceLabel(reference: ReferenceItem): string {
  const metadataSource = reference.metadata?.source
  if (typeof metadataSource === 'string' && metadataSource.trim()) {
    return metadataSource
  }
  return reference.source_id || reference.category
}

function snippet(reference: ReferenceItem): string {
  return (
    reference.answer_context || reference.content || reference.retrieval_text
  )
}
</script>

<template>
  <section
    v-if="references.length"
    class="reference-list"
    :aria-labelledby="titleId"
  >
    <h3 :id="titleId">参考知识</h3>
    <ol>
      <li v-for="(reference, index) in references" :key="reference.chunk_id">
        <div class="reference-list__heading">
          <span class="reference-list__rank">{{
            reference.rank || index + 1
          }}</span>
          <div>
            <h4>{{ reference.title }}</h4>
            <p>{{ sourceLabel(reference) }} · {{ reference.category }}</p>
          </div>
          <span class="reference-list__score"
            >相关度 {{ reference.score.toFixed(3) }}</span
          >
        </div>
        <MathContent
          class="reference-list__snippet"
          :content="snippet(reference)"
        />
      </li>
    </ol>
  </section>
</template>

<style scoped>
.reference-list {
  padding-top: var(--space-5);
  border-top: 1px solid var(--color-border);
}

.reference-list > h3 {
  margin: 0 0 var(--space-3);
  font-size: 15px;
  letter-spacing: 0;
}

.reference-list ol {
  margin: 0;
  padding: 0;
  list-style: none;
}

.reference-list li {
  padding: var(--space-4) 0;
  border-top: 1px solid var(--color-neutral-100);
}

.reference-list li:first-child {
  border-top: 0;
}

.reference-list__heading {
  display: grid;
  grid-template-columns: 28px minmax(0, 1fr) auto;
  align-items: start;
  gap: var(--space-3);
}

.reference-list__rank {
  display: grid;
  width: 26px;
  height: 26px;
  place-items: center;
  border: 1px solid var(--color-border);
  border-radius: 50%;
  color: var(--color-text-secondary);
  font-size: 12px;
  font-weight: 700;
}

.reference-list h4 {
  margin: 1px 0 var(--space-1);
  font-size: 14px;
  letter-spacing: 0;
}

.reference-list__heading p,
.reference-list__score {
  margin: 0;
  color: var(--color-text-secondary);
  font-size: 12px;
}

.reference-list__score {
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.reference-list__snippet {
  margin: var(--space-3) 0 0 40px;
  color: var(--color-text-secondary);
  font-size: 13px;
}

@media (max-width: 560px) {
  .reference-list__heading {
    grid-template-columns: 28px minmax(0, 1fr);
  }

  .reference-list__score {
    grid-column: 2;
    margin-top: var(--space-1);
  }
}
</style>
