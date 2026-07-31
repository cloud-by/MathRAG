<script setup lang="ts">
import { Archive, Pencil, X } from '@lucide/vue'
import { nextTick, ref } from 'vue'
import { RouterLink } from 'vue-router'

import IconButton from '../../components/IconButton.vue'
import type { Conversation } from './types'

const props = defineProps<{
  busy?: boolean
  conversation: Conversation
}>()

const emit = defineEmits<{
  archive: [conversation: Conversation]
  rename: [conversation: Conversation, title: string]
}>()

const editing = ref(false)
const input = ref<HTMLInputElement | null>(null)
const title = ref(props.conversation.title)
const dateFormatter = new Intl.DateTimeFormat('zh-CN', { dateStyle: 'short' })

function formatDate(value: string): string {
  return dateFormatter.format(new Date(value))
}

function fullDate(value: string): string {
  return new Date(value).toISOString()
}

async function beginRename(): Promise<void> {
  title.value = props.conversation.title
  editing.value = true
  await nextTick()
  input.value?.select()
}

function cancelRename(): void {
  editing.value = false
  title.value = props.conversation.title
}

function save(): void {
  const normalized = title.value.trim().replace(/\s+/g, ' ')
  if (!normalized) return
  emit('rename', props.conversation, normalized)
  editing.value = false
}
</script>

<template>
  <li class="conversation-row">
    <div class="conversation-row__main">
      <form
        v-if="editing"
        class="conversation-row__rename"
        @submit.prevent="save"
      >
        <label class="sr-only" :for="`conversation-title-${conversation.id}`">
          会话标题
        </label>
        <input
          :id="`conversation-title-${conversation.id}`"
          ref="input"
          v-model="title"
          maxlength="255"
          :disabled="busy"
        />
        <button
          class="primary-button"
          type="submit"
          :disabled="busy || !title.trim()"
        >
          保存
        </button>
        <IconButton label="取消重命名" :disabled="busy" @click="cancelRename">
          <X :size="17" aria-hidden="true" />
        </IconButton>
      </form>
      <RouterLink
        v-else
        class="conversation-row__title"
        :to="`/conversations/${conversation.id}`"
      >
        {{ conversation.title }}
      </RouterLink>
      <p>
        最后活动
        <time
          :datetime="conversation.updated_at"
          :title="fullDate(conversation.updated_at)"
        >
          {{ formatDate(conversation.updated_at) }}
        </time>
      </p>
    </div>

    <span
      v-if="conversation.status === 'archived'"
      class="conversation-row__status"
    >
      已归档
    </span>
    <div v-else class="conversation-row__actions">
      <IconButton
        :label="`重命名“${conversation.title}”`"
        :disabled="busy || editing"
        @click="beginRename"
      >
        <Pencil :size="17" aria-hidden="true" />
      </IconButton>
      <IconButton
        :label="`归档“${conversation.title}”`"
        :disabled="busy || editing"
        @click="emit('archive', conversation)"
      >
        <Archive :size="17" aria-hidden="true" />
      </IconButton>
    </div>
  </li>
</template>

<style scoped>
.conversation-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: var(--space-4);
  min-height: 76px;
  padding: var(--space-3) var(--space-4);
  border-top: 1px solid var(--color-border);
}

.conversation-row:first-child {
  border-top: 0;
}

.conversation-row__main {
  min-width: 0;
}

.conversation-row__title {
  display: inline-block;
  max-width: 100%;
  overflow: hidden;
  font-size: 15px;
  font-weight: 650;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.conversation-row__title:hover {
  color: var(--color-action);
}

.conversation-row__main p {
  margin: var(--space-1) 0 0;
  color: var(--color-text-secondary);
  font-size: 12px;
}

.conversation-row__actions,
.conversation-row__rename {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.conversation-row__rename input {
  width: min(100%, 360px);
  min-height: 38px;
  padding: 0 var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-neutral-0);
}

.conversation-row__status {
  color: var(--color-text-secondary);
  font-size: 12px;
}

.primary-button {
  min-height: 38px;
  padding: 0 var(--space-4);
  border: 1px solid var(--color-action);
  border-radius: var(--radius-md);
  color: var(--color-neutral-0);
  background: var(--color-action);
  font-weight: 650;
}

@media (max-width: 560px) {
  .conversation-row {
    align-items: start;
    padding-inline: var(--space-3);
  }

  .conversation-row__rename {
    flex-wrap: wrap;
  }
}
</style>
