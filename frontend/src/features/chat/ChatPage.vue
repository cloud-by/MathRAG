<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import InlineAlert from '../../components/InlineAlert.vue'
import ConversationHistoryPage from '../conversations/ConversationHistoryPage.vue'
import ChatComposer from './ChatComposer.vue'
import type { ChatTurn } from './types'
import { useChat } from './useChat'

const route = useRoute()
const router = useRouter()
const chat = useChat()
const draft = ref('')
const topK = ref(3)
const composer = ref<InstanceType<typeof ChatComposer> | null>(null)
const history = ref<InstanceType<typeof ConversationHistoryPage> | null>(null)

const routeConversationId = computed(() =>
  route.name === 'conversation' ? String(route.params.id ?? '') : '',
)
const currentTurns = computed<ChatTurn[]>(() =>
  chat.turns.value.filter(
    (turn) => turn.response.conversation_id === routeConversationId.value,
  ),
)

async function submit(): Promise<void> {
  const response = await chat.submit({
    conversationId: routeConversationId.value || null,
    question: draft.value,
    topK: topK.value,
    onConversationCreated: async (id) => {
      await router.replace({ name: 'conversation', params: { id } })
    },
  })
  if (!response) return
  draft.value = ''
  await nextTick()
  await history.value?.refresh()
}

async function retry(): Promise<void> {
  const response = await chat.retry()
  if (!response) return
  draft.value = ''
  await history.value?.refresh()
}

async function fillQuestion(question: string): Promise<void> {
  draft.value = question
  await nextTick()
  composer.value?.focus()
}

onMounted(() => {
  const conversationId =
    typeof route.query.conversation_id === 'string'
      ? route.query.conversation_id
      : ''
  const question =
    typeof route.query.question === 'string' ? route.query.question : ''
  if (question) draft.value = question
  if (route.name === 'chat' && conversationId) {
    void router.replace({
      name: 'conversation',
      params: { id: conversationId },
      query: question ? { question } : {},
    })
  }
})

watch(routeConversationId, (nextId, previousId) => {
  if (previousId && nextId !== previousId) {
    chat.cancel()
    chat.clearTurns()
    draft.value = ''
  }
})
</script>

<template>
  <ConversationHistoryPage
    v-if="route.name === 'conversation'"
    ref="history"
    :pending="chat.pending.value"
    :turns="currentTurns"
    @select-related="fillQuestion"
  >
    <template #composer>
      <div class="chat-page__controls">
        <ChatComposer
          ref="composer"
          v-model="draft"
          v-model:top-k="topK"
          :state="chat.status.value"
          @cancel="chat.cancel"
          @submit="submit"
        />
        <InlineAlert
          v-if="chat.state.value.status === 'error'"
          tone="error"
          title="回答未完成"
        >
          <p>{{ chat.state.value.error.message }}</p>
          <small>请求编号：{{ chat.state.value.error.requestId }}</small>
          <button
            v-if="chat.state.value.retryable"
            class="inline-command"
            type="button"
            @click="retry"
          >
            使用原请求重试
          </button>
        </InlineAlert>
        <InlineAlert
          v-else-if="chat.state.value.status === 'cancelled'"
          title="已停止回答"
        >
          <p>可以修改问题后重新发送。</p>
        </InlineAlert>
      </div>
    </template>
  </ConversationHistoryPage>

  <main v-else class="new-chat-page">
    <section class="new-chat-page__workspace">
      <h2>今天想解决什么问题？</h2>
      <ChatComposer
        ref="composer"
        v-model="draft"
        v-model:top-k="topK"
        :state="chat.status.value"
        @cancel="chat.cancel"
        @submit="submit"
      />
      <InlineAlert
        v-if="chat.state.value.status === 'error'"
        tone="error"
        title="回答未完成"
      >
        <p>{{ chat.state.value.error.message }}</p>
        <small>请求编号：{{ chat.state.value.error.requestId }}</small>
        <button
          v-if="chat.state.value.retryable"
          class="inline-command"
          type="button"
          @click="retry"
        >
          使用原请求重试
        </button>
      </InlineAlert>
      <InlineAlert
        v-else-if="chat.state.value.status === 'cancelled'"
        title="已停止回答"
      >
        <p>可以修改问题后重新发送。</p>
      </InlineAlert>
    </section>
  </main>
</template>

<style scoped>
.new-chat-page {
  display: grid;
  min-height: calc(100vh - 58px);
  place-items: center;
  padding: var(--space-6);
}

.new-chat-page__workspace {
  width: min(100%, 760px);
  transform: translateY(-7vh);
}

.new-chat-page h2 {
  margin: 0 0 var(--space-5);
  font-size: 24px;
  letter-spacing: 0;
  text-align: center;
}

.chat-page__controls {
  position: sticky;
  bottom: 0;
  padding: var(--space-4) 0 var(--space-2);
  background: linear-gradient(
    to bottom,
    rgb(246 247 249 / 0%),
    var(--color-neutral-50) var(--space-4)
  );
}

.chat-page__controls :deep(.inline-alert),
.new-chat-page__workspace :deep(.inline-alert) {
  margin-top: var(--space-3);
}

.inline-command {
  margin-top: var(--space-3);
  padding: 0;
  border: 0;
  color: var(--color-action);
  background: transparent;
  font-weight: 650;
}

small {
  display: block;
  margin-top: var(--space-1);
  color: var(--color-text-secondary);
}

@media (max-width: 640px) {
  .new-chat-page {
    align-items: start;
    padding: var(--space-6) var(--space-4);
  }

  .new-chat-page__workspace {
    transform: none;
  }

  .new-chat-page h2 {
    margin-top: var(--space-8);
    font-size: 21px;
  }
}
</style>
