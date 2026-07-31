<script setup lang="ts">
import { computed } from 'vue'

import type { IngestionJobStatus } from './types'

const props = defineProps<{ status: IngestionJobStatus }>()

const label = computed(
  () =>
    ({
      pending: '等待中',
      running: '处理中',
      completed: '已完成',
      failed: '失败',
      cancelled: '已取消',
    })[props.status],
)
</script>

<template>
  <span class="job-status" :class="`job-status--${status}`">{{ label }}</span>
</template>

<style scoped>
.job-status {
  display: inline-flex;
  min-height: 24px;
  align-items: center;
  padding: 0 var(--space-2);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  color: var(--color-text-secondary);
  background: var(--color-neutral-50);
  font-size: 12px;
  white-space: nowrap;
}

.job-status--running,
.job-status--pending {
  border-color: #b9c3e6;
  color: #344b9c;
  background: #f2f5ff;
}

.job-status--completed {
  border-color: #b9d8c8;
  color: #27714e;
  background: #f1faf5;
}

.job-status--failed {
  border-color: #e4babe;
  color: #9e343d;
  background: #fff4f4;
}
</style>
