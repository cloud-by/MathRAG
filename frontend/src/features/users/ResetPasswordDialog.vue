<script setup lang="ts">
import { ref, watch } from 'vue'

import ConfirmDialog from '../../components/ConfirmDialog.vue'
import InlineAlert from '../../components/InlineAlert.vue'
import type { ApiError } from '../../api/errors'

const props = withDefaults(
  defineProps<{
    busy?: boolean
    error?: ApiError | null
    open: boolean
    username: string
  }>(),
  { busy: false, error: null },
)

const emit = defineEmits<{
  cancel: []
  confirm: [password: string]
}>()

const password = ref('')
const confirmation = ref('')
const passwordError = ref('')
const confirmationError = ref('')

function clear(): void {
  password.value = ''
  confirmation.value = ''
  passwordError.value = ''
  confirmationError.value = ''
}

function cancel(): void {
  if (!props.busy) emit('cancel')
}

function submit(): void {
  passwordError.value =
    password.value.length >= 12 && password.value.length <= 128
      ? ''
      : '临时密码长度必须为 12 至 128 个字符。'
  confirmationError.value =
    confirmation.value === password.value ? '' : '两次输入的临时密码不一致。'
  if (passwordError.value || confirmationError.value) return
  emit('confirm', password.value)
}

watch(
  () => props.open,
  (open) => {
    if (open) clear()
  },
)
</script>

<template>
  <ConfirmDialog
    :open="open"
    :busy="busy"
    title="重置临时密码"
    :object-name="username"
    confirm-label="确认重置"
    @cancel="cancel"
    @confirm="submit"
  >
    <InlineAlert v-if="error" tone="error" title="重置失败">
      <p>{{ error.message }}</p>
      <small>请求编号：{{ error.requestId }}</small>
    </InlineAlert>
    <label class="reset-field" for="reset-password">
      <span>重置临时密码</span>
      <input
        id="reset-password"
        v-model="password"
        type="password"
        autocomplete="new-password"
        :disabled="busy"
        :aria-invalid="Boolean(passwordError)"
        aria-describedby="reset-password-error"
      />
      <small v-if="passwordError" id="reset-password-error">{{
        passwordError
      }}</small>
    </label>
    <label class="reset-field" for="reset-confirmation">
      <span>确认重置临时密码</span>
      <input
        id="reset-confirmation"
        v-model="confirmation"
        type="password"
        autocomplete="new-password"
        :disabled="busy"
        :aria-invalid="Boolean(confirmationError)"
        aria-describedby="reset-confirmation-error"
      />
      <small v-if="confirmationError" id="reset-confirmation-error">{{
        confirmationError
      }}</small>
    </label>
  </ConfirmDialog>
</template>

<style scoped>
.reset-field {
  display: grid;
  gap: var(--space-2);
  color: var(--color-text-secondary);
  font-size: 13px;
  font-weight: 650;
}

.reset-field input {
  width: 100%;
  min-height: 40px;
  padding: 0 var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-neutral-0);
}

.reset-field input[aria-invalid='true'] {
  border-color: var(--color-error);
}

.reset-field small {
  color: var(--color-error);
  font-size: 12px;
  font-weight: 400;
}
</style>
