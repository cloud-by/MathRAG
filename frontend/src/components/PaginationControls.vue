<script setup lang="ts">
import { ChevronLeft, ChevronRight } from '@lucide/vue'
import { computed } from 'vue'

import IconButton from './IconButton.vue'

const props = defineProps<{
  limit: number
  offset: number
  total: number
}>()

const emit = defineEmits<{
  'update:offset': [offset: number]
}>()

const start = computed(() => (props.total ? props.offset + 1 : 0))
const end = computed(() => Math.min(props.offset + props.limit, props.total))
const hasPrevious = computed(() => props.offset > 0)
const hasNext = computed(() => props.offset + props.limit < props.total)

function previous(): void {
  emit('update:offset', Math.max(0, props.offset - props.limit))
}

function next(): void {
  emit('update:offset', props.offset + props.limit)
}
</script>

<template>
  <nav class="pagination" aria-label="分页">
    <span class="pagination__summary">{{ start }}-{{ end }} / {{ total }}</span>
    <IconButton label="上一页" :disabled="!hasPrevious" @click="previous">
      <ChevronLeft :size="18" aria-hidden="true" />
    </IconButton>
    <IconButton label="下一页" :disabled="!hasNext" @click="next">
      <ChevronRight :size="18" aria-hidden="true" />
    </IconButton>
  </nav>
</template>

<style scoped>
.pagination {
  display: flex;
  min-height: 44px;
  align-items: center;
  justify-content: flex-end;
  gap: var(--space-1);
}

.pagination__summary {
  min-width: 112px;
  margin-right: var(--space-2);
  color: var(--color-text-secondary);
  font-variant-numeric: tabular-nums;
  text-align: right;
}
</style>
