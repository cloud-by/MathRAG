<script setup lang="ts">
import { KeyRound } from '@lucide/vue'
import { ref } from 'vue'
import { useRouter } from 'vue-router'

import InlineAlert from '../../components/InlineAlert.vue'
import { useAuth } from './useAuth'

const auth = useAuth()
const router = useRouter()

const currentPassword = ref('')
const newPassword = ref('')
const confirmation = ref('')
const currentPasswordError = ref('')
const newPasswordError = ref('')
const confirmationError = ref('')
const error = ref<Error | null>(null)
const submitting = ref(false)

function validate(): boolean {
  currentPasswordError.value = currentPassword.value ? '' : '请输入当前密码。'
  newPasswordError.value =
    newPassword.value.length >= 12 && newPassword.value.length <= 128
      ? ''
      : '新密码长度必须为 12 至 128 个字符。'
  confirmationError.value =
    confirmation.value === newPassword.value ? '' : '两次输入的新密码不一致。'
  return !(
    currentPasswordError.value ||
    newPasswordError.value ||
    confirmationError.value
  )
}

async function submit(): Promise<void> {
  error.value = null
  if (!validate()) return

  submitting.value = true
  try {
    await auth.changePassword({
      current_password: currentPassword.value,
      new_password: newPassword.value,
    })
    await router.replace({ name: 'login', query: { password_changed: '1' } })
  } catch (caught) {
    error.value =
      caught instanceof Error ? caught : new Error('密码修改失败，请稍后重试。')
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <main class="password-page" aria-labelledby="password-page-title">
    <div class="password-page__inner">
      <header class="password-page__header">
        <div class="password-page__icon" aria-hidden="true">
          <KeyRound :size="22" />
        </div>
        <div>
          <h1 id="password-page-title">修改临时密码</h1>
          <p>设置新密码后需要重新登录。</p>
        </div>
      </header>

      <form class="password-form" novalidate @submit.prevent="submit">
        <div class="form-field">
          <label for="current-password">当前密码</label>
          <input
            id="current-password"
            v-model="currentPassword"
            type="password"
            autocomplete="current-password"
            :disabled="submitting"
            :aria-invalid="Boolean(currentPasswordError)"
            :aria-describedby="
              currentPasswordError ? 'current-password-error' : undefined
            "
            @input="currentPasswordError = ''"
          />
          <p
            v-if="currentPasswordError"
            id="current-password-error"
            class="field-error"
          >
            {{ currentPasswordError }}
          </p>
        </div>

        <div class="form-field">
          <label for="new-password">新密码</label>
          <input
            id="new-password"
            v-model="newPassword"
            type="password"
            autocomplete="new-password"
            :disabled="submitting"
            :aria-invalid="Boolean(newPasswordError)"
            :aria-describedby="
              newPasswordError ? 'new-password-error' : undefined
            "
            @input="newPasswordError = ''"
          />
          <p
            v-if="newPasswordError"
            id="new-password-error"
            class="field-error"
          >
            {{ newPasswordError }}
          </p>
        </div>

        <div class="form-field">
          <label for="confirm-password">确认新密码</label>
          <input
            id="confirm-password"
            v-model="confirmation"
            type="password"
            autocomplete="new-password"
            :disabled="submitting"
            :aria-invalid="Boolean(confirmationError)"
            :aria-describedby="
              confirmationError ? 'confirmation-error' : undefined
            "
            @input="confirmationError = ''"
          />
          <p
            v-if="confirmationError"
            id="confirmation-error"
            class="field-error"
          >
            {{ confirmationError }}
          </p>
        </div>

        <InlineAlert v-if="error" tone="error" title="密码修改失败">
          <p>{{ error.message }}</p>
        </InlineAlert>

        <button class="submit-button" type="submit" :disabled="submitting">
          <KeyRound :size="18" aria-hidden="true" />
          {{ submitting ? '正在修改' : '修改密码' }}
        </button>
      </form>
    </div>
  </main>
</template>

<style scoped>
.password-page {
  min-height: calc(100vh - 58px);
  padding: clamp(32px, 6vw, 72px) clamp(20px, 5vw, 64px);
  background: var(--color-neutral-0);
}

.password-page__inner {
  width: min(100%, 560px);
  margin: 0 auto;
}

.password-page__header {
  display: flex;
  align-items: flex-start;
  gap: var(--space-4);
  padding-bottom: var(--space-6);
  border-bottom: 1px solid var(--color-border);
}

.password-page__icon {
  display: grid;
  flex: 0 0 42px;
  width: 42px;
  height: 42px;
  place-items: center;
  border-radius: var(--radius-md);
  color: var(--color-neutral-0);
  background: #253252;
}

.password-page h1 {
  margin: 0;
  font-size: 24px;
  letter-spacing: 0;
}

.password-page__header p {
  margin: var(--space-2) 0 0;
  color: var(--color-text-secondary);
  font-size: 14px;
}

.password-form {
  display: grid;
  gap: var(--space-5);
  padding-top: var(--space-6);
}

.form-field {
  min-height: 84px;
}

.form-field label {
  display: block;
  margin-bottom: var(--space-2);
  color: var(--color-text-secondary);
  font-size: 14px;
  font-weight: 600;
}

.form-field input {
  width: 100%;
  height: 44px;
  padding: 0 var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-neutral-0);
}

.form-field input[aria-invalid='true'] {
  border-color: var(--color-error);
}

.form-field input:disabled {
  background: var(--color-neutral-100);
}

.field-error {
  margin: var(--space-1) 0 0;
  color: var(--color-error);
  font-size: 13px;
}

.submit-button {
  display: inline-flex;
  min-height: 44px;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  border: 1px solid var(--color-action);
  border-radius: var(--radius-md);
  color: var(--color-neutral-0);
  background: var(--color-action);
  font-weight: 700;
}

.submit-button:hover:not(:disabled) {
  border-color: var(--color-action-hover);
  background: var(--color-action-hover);
}

.submit-button:disabled {
  opacity: 0.65;
  cursor: wait;
}

@media (max-width: 560px) {
  .password-page {
    padding: var(--space-6) var(--space-5);
  }
}
</style>
