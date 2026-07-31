<script setup lang="ts">
import { LogIn } from '@lucide/vue'
import { computed, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { safeNextPath } from '../../router'
import { useAuth } from './useAuth'

const auth = useAuth()
const route = useRoute()
const router = useRouter()

const username = ref('')
const password = ref('')
const usernameError = ref('')
const passwordError = ref('')
const serverError = ref('')
const submitting = ref(false)
const visibleError = computed(
  () =>
    serverError.value ||
    (route.query.auth_error === '1'
      ? '暂时无法确认登录状态，请重新登录。'
      : ''),
)

function validate(): boolean {
  usernameError.value = username.value.trim() ? '' : '请输入邮箱或用户名。'
  passwordError.value = password.value ? '' : '请输入密码。'
  return !usernameError.value && !passwordError.value
}

function destinationAfterLogin(): string {
  const destination = safeNextPath(route.query.next)
  return destination && !destination.startsWith('/login')
    ? destination
    : '/chat'
}

async function submit(): Promise<void> {
  serverError.value = ''
  if (!validate()) {
    return
  }

  submitting.value = true
  try {
    await auth.login({
      username: username.value.trim(),
      password: password.value,
    })
    await router.replace(destinationAfterLogin())
  } catch (error) {
    serverError.value =
      error instanceof Error ? error.message : '登录失败，请稍后重试。'
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <main class="login-page">
    <section class="login-brand" aria-labelledby="product-name">
      <div class="brand-mark" aria-hidden="true">M</div>
      <div>
        <h1 id="product-name">MathRAG</h1>
        <p>数学知识工作台</p>
      </div>
    </section>

    <section class="login-form-section" aria-labelledby="login-title">
      <form class="login-form" novalidate @submit.prevent="submit">
        <header>
          <p class="form-eyebrow">欢迎回来</p>
          <h2 id="login-title">登录账户</h2>
        </header>

        <div class="form-field">
          <label for="username">邮箱或用户名</label>
          <input
            id="username"
            v-model="username"
            name="username"
            type="text"
            autocomplete="username"
            inputmode="email"
            :aria-describedby="usernameError ? 'username-error' : undefined"
            :aria-invalid="Boolean(usernameError)"
            :disabled="submitting"
            @input="usernameError = ''"
          />
          <p v-if="usernameError" id="username-error" class="field-error">
            {{ usernameError }}
          </p>
        </div>

        <div class="form-field">
          <label for="password">密码</label>
          <input
            id="password"
            v-model="password"
            name="password"
            type="password"
            autocomplete="current-password"
            :aria-describedby="passwordError ? 'password-error' : undefined"
            :aria-invalid="Boolean(passwordError)"
            :disabled="submitting"
            @input="passwordError = ''"
          />
          <p v-if="passwordError" id="password-error" class="field-error">
            {{ passwordError }}
          </p>
        </div>

        <p v-if="visibleError" class="server-error" role="alert">
          {{ visibleError }}
        </p>

        <button class="submit-button" type="submit" :disabled="submitting">
          <LogIn :size="18" aria-hidden="true" />
          {{ submitting ? '正在登录' : '登录' }}
        </button>
      </form>
    </section>
  </main>
</template>

<style scoped>
.login-page {
  display: grid;
  grid-template-columns: minmax(280px, 0.85fr) minmax(420px, 1.15fr);
  min-height: 100vh;
  background: var(--color-neutral-0);
}

.login-brand {
  display: flex;
  align-items: flex-start;
  gap: var(--space-4);
  padding: 64px clamp(32px, 6vw, 88px);
  color: var(--color-neutral-0);
  background: #253252;
}

.brand-mark {
  display: grid;
  flex: 0 0 42px;
  width: 42px;
  height: 42px;
  place-items: center;
  border: 1px solid rgb(255 255 255 / 42%);
  border-radius: var(--radius-md);
  font-size: 22px;
  font-weight: 700;
}

.login-brand h1 {
  margin: 0;
  font-size: 28px;
  letter-spacing: 0;
}

.login-brand p {
  margin: var(--space-2) 0 0;
  color: rgb(255 255 255 / 72%);
  font-size: 14px;
}

.login-form-section {
  display: grid;
  align-items: center;
  padding: 48px clamp(32px, 9vw, 128px);
}

.login-form {
  width: min(100%, 420px);
}

.login-form header {
  margin-bottom: var(--space-8);
}

.form-eyebrow {
  margin: 0 0 var(--space-2);
  color: var(--color-action);
  font-size: 13px;
  font-weight: 700;
}

.login-form h2 {
  margin: 0;
  font-size: 26px;
  letter-spacing: 0;
}

.form-field {
  min-height: 100px;
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

.form-field input:hover:not(:disabled) {
  border-color: #aeb5c2;
}

.form-field input[aria-invalid='true'] {
  border-color: var(--color-error);
}

.form-field input:disabled {
  color: var(--color-text-secondary);
  background: var(--color-neutral-100);
  cursor: not-allowed;
}

.field-error {
  margin: var(--space-1) 0 0;
  color: var(--color-error);
  font-size: 13px;
}

.server-error {
  margin: 0 0 var(--space-4);
  padding: var(--space-3);
  border-left: 3px solid var(--color-error);
  color: #8e343c;
  background: #fff4f4;
  font-size: 14px;
}

.submit-button {
  display: inline-flex;
  width: 100%;
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

@media (max-width: 760px) {
  .login-page {
    grid-template-columns: 1fr;
    grid-template-rows: auto 1fr;
  }

  .login-brand {
    padding: var(--space-6) var(--space-5);
  }

  .login-form-section {
    align-items: start;
    padding: 48px var(--space-5);
  }

  .login-form {
    margin: 0 auto;
  }
}
</style>
