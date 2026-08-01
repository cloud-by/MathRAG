<script setup lang="ts">
import { TriangleAlert } from '@lucide/vue'
import { nextTick, onBeforeUnmount, ref, watch } from 'vue'

const props = withDefaults(
  defineProps<{
    busy?: boolean
    cancelLabel?: string
    confirmLabel?: string
    danger?: boolean
    objectName: string
    open: boolean
    title: string
  }>(),
  {
    busy: false,
    cancelLabel: '取消',
    confirmLabel: '确认',
    danger: false,
  },
)

const emit = defineEmits<{
  cancel: []
  confirm: []
}>()

const panel = ref<HTMLElement | null>(null)
let returnFocus: HTMLElement | null = null

function focusableElements(): HTMLElement[] {
  if (!panel.value) return []
  return Array.from(
    panel.value.querySelectorAll<HTMLElement>(
      'button:not(:disabled), [href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex="-1"])',
    ),
  )
}

function cancel(): void {
  if (!props.busy) emit('cancel')
}

function onKeydown(event: KeyboardEvent): void {
  if (event.key === 'Escape') {
    event.preventDefault()
    cancel()
    return
  }
  if (event.key !== 'Tab') return

  const focusable = focusableElements()
  const first = focusable[0]
  const last = focusable.at(-1)
  if (!first || !last) {
    event.preventDefault()
    panel.value?.focus()
    return
  }
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault()
    last.focus()
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault()
    first.focus()
  }
}

watch(
  () => props.open,
  async (open) => {
    if (open) {
      returnFocus =
        document.activeElement instanceof HTMLElement
          ? document.activeElement
          : null
      await nextTick()
      focusableElements()[0]?.focus()
      return
    }
    await nextTick()
    returnFocus?.focus()
    returnFocus = null
  },
  { immediate: true },
)

onBeforeUnmount(() => returnFocus?.focus())
</script>

<template>
  <Teleport to="body">
    <div v-if="open" class="dialog-backdrop" @mousedown.self="cancel">
      <section
        ref="panel"
        class="confirm-dialog"
        :role="danger ? 'alertdialog' : 'dialog'"
        aria-modal="true"
        :aria-labelledby="'confirm-dialog-title'"
        :aria-describedby="'confirm-dialog-description'"
        tabindex="-1"
        @keydown="onKeydown"
      >
        <header>
          <TriangleAlert v-if="danger" :size="20" aria-hidden="true" />
          <h2 id="confirm-dialog-title">{{ title }}</h2>
        </header>
        <p id="confirm-dialog-description">
          此操作将影响“<strong>{{ objectName }}</strong
          >”。
        </p>
        <div v-if="$slots.default" class="confirm-dialog__content">
          <slot />
        </div>
        <div class="confirm-dialog__actions">
          <button
            class="secondary-button"
            type="button"
            :disabled="busy"
            @click="cancel"
          >
            {{ cancelLabel }}
          </button>
          <button
            class="primary-button"
            :class="{ 'primary-button--danger': danger }"
            type="button"
            :disabled="busy"
            @click="emit('confirm')"
          >
            {{ busy ? '正在处理' : confirmLabel }}
          </button>
        </div>
      </section>
    </div>
  </Teleport>
</template>

<style scoped>
.dialog-backdrop {
  position: fixed;
  z-index: 100;
  inset: 0;
  display: grid;
  place-items: center;
  padding: var(--space-4);
  background: rgb(23 32 51 / 48%);
}

.confirm-dialog {
  width: min(100%, 440px);
  padding: var(--space-6);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: var(--color-neutral-0);
  box-shadow: 0 18px 52px rgb(23 32 51 / 18%);
}

.confirm-dialog header {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  color: var(--color-error);
}

.confirm-dialog h2 {
  margin: 0;
  color: var(--color-text-primary);
  font-size: 18px;
  letter-spacing: 0;
}

.confirm-dialog p {
  margin: var(--space-4) 0 var(--space-6);
  color: var(--color-text-secondary);
  line-height: 1.6;
}

.confirm-dialog__actions {
  display: flex;
  justify-content: flex-end;
  gap: var(--space-2);
}

.confirm-dialog__content {
  display: grid;
  gap: var(--space-3);
  margin: calc(var(--space-2) * -1) 0 var(--space-6);
}

.secondary-button,
.primary-button {
  min-height: 38px;
  padding: 0 var(--space-4);
  border-radius: var(--radius-md);
  font-weight: 600;
}

.secondary-button {
  border: 1px solid var(--color-border);
  background: var(--color-neutral-0);
}

.primary-button {
  border: 1px solid var(--color-action);
  color: var(--color-neutral-0);
  background: var(--color-action);
}

.primary-button--danger {
  border-color: var(--color-error);
  background: var(--color-error);
}

.secondary-button:disabled,
.primary-button:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}
</style>
