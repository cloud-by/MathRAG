<script setup lang="ts">
import { ArrowLeft, KeyRound, Save } from '@lucide/vue'
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'

import ConfirmDialog from '../../components/ConfirmDialog.vue'
import InlineAlert from '../../components/InlineAlert.vue'
import LoadingState from '../../components/LoadingState.vue'
import { ApiError } from '../../api/errors'
import { useAuth } from '../auth/useAuth'
import { usersApi } from './api'
import ResetPasswordDialog from './ResetPasswordDialog.vue'
import type {
  ManagedUser,
  UserCreate,
  UserRole,
  UserStatus,
  UserUpdate,
} from './types'

interface UserDraft {
  username: string
  email: string
  role: UserRole
  status: UserStatus
  password: string
  confirmation: string
}

const route = useRoute()
const router = useRouter()
const auth = useAuth()
const draft = reactive<UserDraft>({
  username: '',
  email: '',
  role: 'student',
  status: 'active',
  password: '',
  confirmation: '',
})
const baseline = ref<ManagedUser | null>(null)
const loading = ref(false)
const loadError = ref<ApiError | null>(null)
const saving = ref(false)
const mutationError = ref<ApiError | null>(null)
const successMessage = ref('')
const fieldErrors = ref<Record<string, string>>({})
const statusTarget = ref<UserStatus | null>(null)
const resetOpen = ref(false)
const resetSubmitting = ref(false)
const resetError = ref<ApiError | null>(null)
let loadController: AbortController | null = null
let loadSequence = 0
let routeGeneration = 0

const isNew = computed(() => route.name === 'user-new')
const userId = computed(() =>
  isNew.value ? '' : String(route.params.id ?? ''),
)
const currentUser = computed(() =>
  auth.state.value.status === 'authenticated' ? auth.state.value.user : null,
)
const isAdmin = computed(() => currentUser.value?.role === 'admin')
const isSelf = computed(
  () => !isNew.value && currentUser.value?.id === userId.value,
)
const canEditRole = computed(() => isAdmin.value && !isSelf.value)
const statusDialogTitle = computed(() =>
  statusTarget.value === 'disabled' ? '禁用账号' : '启用账号',
)

function asApiError(error: unknown): ApiError {
  if (error instanceof ApiError) return error
  return new ApiError({
    code: 'NETWORK_ERROR',
    message: '请求失败，请稍后重试。',
    requestId: 'unavailable',
    status: 0,
    details: null,
  })
}

function isAbort(error: unknown): boolean {
  return (
    typeof error === 'object' &&
    error !== null &&
    'name' in error &&
    error.name === 'AbortError'
  )
}

function resetDraft(): void {
  Object.assign(draft, {
    username: '',
    email: '',
    role: 'student',
    status: 'active',
    password: '',
    confirmation: '',
  })
  baseline.value = null
  fieldErrors.value = {}
  mutationError.value = null
  loadError.value = null
  successMessage.value = ''
  statusTarget.value = null
  resetOpen.value = false
  resetError.value = null
}

function adopt(user: ManagedUser): void {
  Object.assign(draft, {
    username: user.username,
    email: user.email ?? '',
    role: user.role,
    status: user.status,
    password: '',
    confirmation: '',
  })
  baseline.value = user
  fieldErrors.value = {}
  loadError.value = null
  mutationError.value = null
}

async function load(id: string): Promise<void> {
  loadController?.abort()
  loadController = new AbortController()
  const request = ++loadSequence
  loading.value = true
  loadError.value = null
  try {
    const user = await usersApi.get(id, loadController.signal)
    if (request === loadSequence) adopt(user)
  } catch (error) {
    if (request === loadSequence && !isAbort(error)) {
      loadError.value = asApiError(error)
    }
  } finally {
    if (request === loadSequence) loading.value = false
  }
}

function validateCreation(): boolean {
  const errors: Record<string, string> = {}
  if (!draft.username.trim()) errors.username = '请输入用户名。'
  if (draft.password.length < 12 || draft.password.length > 128) {
    errors.password = '临时密码长度必须为 12 至 128 个字符。'
  }
  if (draft.confirmation !== draft.password) {
    errors.confirmation = '两次输入的临时密码不一致。'
  }
  fieldErrors.value = errors
  return !Object.keys(errors).length
}

function validateUpdate(): boolean {
  const errors: Record<string, string> = {}
  if (!draft.username.trim()) errors.username = '请输入用户名。'
  fieldErrors.value = errors
  return !Object.keys(errors).length
}

function updateValues(): UserUpdate {
  const original = baseline.value
  if (!original) return {}
  const values: UserUpdate = {}
  const username = draft.username.trim()
  const email = draft.email.trim() || null
  if (username !== original.username) values.username = username
  if (email !== original.email) values.email = email
  if (canEditRole.value && draft.role !== original.role) {
    values.role = draft.role
  }
  if (!isSelf.value && draft.status !== original.status) {
    values.status = draft.status
  }
  return values
}

async function save(): Promise<void> {
  if (!isNew.value && !baseline.value) return
  mutationError.value = null
  successMessage.value = ''
  if (isNew.value ? !validateCreation() : !validateUpdate()) return

  saving.value = true
  const operation = routeGeneration
  try {
    if (isNew.value) {
      const values: UserCreate = {
        username: draft.username.trim(),
        email: draft.email.trim() || null,
        password: draft.password,
        role: isAdmin.value ? draft.role : 'student',
      }
      const created = await usersApi.create(values)
      if (operation !== routeGeneration) return
      await router.push(`/users/${created.id}`)
      return
    }

    const values = updateValues()
    if (!Object.keys(values).length) {
      successMessage.value = '账号信息没有变化。'
      return
    }
    const updated = await usersApi.update(userId.value, values)
    if (operation !== routeGeneration) return
    adopt(updated)
    successMessage.value = '账号信息已保存。'
  } catch (error) {
    mutationError.value = asApiError(error)
  } finally {
    saving.value = false
  }
}

function requestStatusToggle(): void {
  if (isSelf.value || !baseline.value) return
  statusTarget.value = draft.status === 'active' ? 'disabled' : 'active'
}

function confirmStatusToggle(): void {
  if (statusTarget.value) draft.status = statusTarget.value
  statusTarget.value = null
}

function openReset(): void {
  if (!baseline.value) return
  resetError.value = null
  resetOpen.value = true
}

function closeReset(): void {
  if (resetSubmitting.value) return
  resetOpen.value = false
  resetError.value = null
}

async function resetPassword(password: string): Promise<void> {
  if (!baseline.value) return
  resetSubmitting.value = true
  resetError.value = null
  successMessage.value = ''
  const operation = routeGeneration
  try {
    await usersApi.resetPassword(userId.value, { password })
    if (operation !== routeGeneration) return
    resetOpen.value = false
    successMessage.value = '临时密码已重置。'
  } catch (error) {
    if (operation === routeGeneration) resetError.value = asApiError(error)
  } finally {
    resetSubmitting.value = false
  }
}

watch(
  [isNew, userId],
  ([newMode, id]) => {
    loadController?.abort()
    loadSequence += 1
    routeGeneration += 1
    resetDraft()
    if (!newMode && id) void load(id)
    else loading.value = false
  },
  { immediate: true },
)

onBeforeUnmount(() => loadController?.abort())
</script>

<template>
  <main class="user-editor-page">
    <LoadingState v-if="loading" label="正在加载账号信息" />
    <section v-else-if="!isNew && loadError && !baseline" class="detail-error">
      <RouterLink class="back-link" to="/users">
        <ArrowLeft :size="16" aria-hidden="true" />
        返回用户管理
      </RouterLink>
      <InlineAlert tone="error" title="无法读取账号信息">
        <p>{{ loadError.message }}</p>
        <small>请求编号：{{ loadError.requestId }}</small>
        <button class="inline-command" type="button" @click="load(userId)">
          重试
        </button>
      </InlineAlert>
    </section>
    <template v-else>
      <header class="user-editor-page__header">
        <div>
          <RouterLink class="back-link" to="/users">
            <ArrowLeft :size="16" aria-hidden="true" />
            返回用户管理
          </RouterLink>
          <h2>{{ isNew ? '创建账号' : draft.username || '账号详情' }}</h2>
          <p v-if="!isNew && baseline">
            由 {{ baseline.created_by_username || '系统' }} 创建
          </p>
        </div>
        <div class="header-actions">
          <button
            v-if="!isNew"
            class="secondary-command"
            type="button"
            @click="openReset"
          >
            <KeyRound :size="18" aria-hidden="true" />
            重置密码
          </button>
          <button
            class="primary-command"
            type="button"
            :disabled="saving"
            @click="save"
          >
            <Save :size="18" aria-hidden="true" />
            {{ saving ? '正在保存' : isNew ? '创建账号' : '保存更改' }}
          </button>
        </div>
      </header>

      <InlineAlert v-if="mutationError" tone="error" title="操作未完成">
        <p>{{ mutationError.message }}</p>
        <small>请求编号：{{ mutationError.requestId }}</small>
      </InlineAlert>
      <InlineAlert v-if="successMessage" tone="success" title="操作成功">
        <p>{{ successMessage }}</p>
      </InlineAlert>
      <InlineAlert v-if="isSelf" tone="info" title="当前管理员账号">
        <p>不能修改自己的角色或账号状态。</p>
      </InlineAlert>

      <form class="user-form" @submit.prevent="save">
        <div class="field-grid">
          <label class="form-field">
            <span>用户名</span>
            <input
              v-model="draft.username"
              type="text"
              maxlength="255"
              autocomplete="off"
              :aria-invalid="Boolean(fieldErrors.username)"
            />
            <small v-if="fieldErrors.username" class="field-error">{{
              fieldErrors.username
            }}</small>
          </label>
          <label class="form-field">
            <span>邮箱</span>
            <input
              v-model="draft.email"
              type="email"
              maxlength="320"
              autocomplete="off"
            />
          </label>
        </div>

        <div class="field-grid">
          <label v-if="isAdmin" class="form-field">
            <span>角色</span>
            <select v-model="draft.role" :disabled="isSelf">
              <option value="student">学生</option>
              <option value="teacher">教师</option>
              <option value="admin">管理员</option>
            </select>
          </label>
          <div v-else class="form-field form-field--static">
            <span>账号角色</span>
            <strong>学生</strong>
          </div>

          <div v-if="!isNew" class="form-field form-field--status">
            <span>账号状态</span>
            <button
              class="status-switch"
              type="button"
              role="switch"
              :aria-checked="draft.status === 'active'"
              :aria-label="
                draft.status === 'active' ? '账号已启用' : '账号已禁用'
              "
              :disabled="isSelf"
              @click="requestStatusToggle"
            >
              <span aria-hidden="true"></span>
              {{ draft.status === 'active' ? '启用' : '禁用' }}
            </button>
          </div>
        </div>

        <template v-if="isNew">
          <div class="field-grid">
            <label class="form-field">
              <span>临时密码</span>
              <input
                v-model="draft.password"
                type="password"
                autocomplete="new-password"
                :aria-invalid="Boolean(fieldErrors.password)"
              />
              <small v-if="fieldErrors.password" class="field-error">{{
                fieldErrors.password
              }}</small>
            </label>
            <label class="form-field">
              <span>确认临时密码</span>
              <input
                v-model="draft.confirmation"
                type="password"
                autocomplete="new-password"
                :aria-invalid="Boolean(fieldErrors.confirmation)"
              />
              <small v-if="fieldErrors.confirmation" class="field-error">{{
                fieldErrors.confirmation
              }}</small>
            </label>
          </div>
          <p class="temporary-note">新账号首次登录后必须修改临时密码。</p>
        </template>
      </form>
    </template>

    <ConfirmDialog
      :open="statusTarget !== null"
      :title="statusDialogTitle"
      :object-name="draft.username"
      :confirm-label="statusTarget === 'disabled' ? '确认禁用' : '确认启用'"
      :danger="statusTarget === 'disabled'"
      @cancel="statusTarget = null"
      @confirm="confirmStatusToggle"
    />
    <ResetPasswordDialog
      :open="resetOpen"
      :busy="resetSubmitting"
      :error="resetError"
      :username="draft.username"
      @cancel="closeReset"
      @confirm="resetPassword"
    />
  </main>
</template>

<style scoped>
.user-editor-page {
  width: min(100%, 920px);
  margin: 0 auto;
  padding: var(--space-6);
}

.user-editor-page__header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: var(--space-5);
  margin-bottom: var(--space-6);
}

.back-link,
.header-actions,
.primary-command,
.secondary-command {
  display: flex;
  align-items: center;
}

.back-link {
  width: fit-content;
  gap: var(--space-1);
  color: var(--color-text-secondary);
  font-size: 13px;
  text-decoration: none;
}

.user-editor-page__header h2 {
  margin: var(--space-2) 0 0;
  font-size: 24px;
  letter-spacing: 0;
}

.user-editor-page__header p {
  margin: var(--space-1) 0 0;
  color: var(--color-text-secondary);
  font-size: 12px;
}

.header-actions {
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: var(--space-2);
}

.primary-command,
.secondary-command {
  min-height: 40px;
  justify-content: center;
  gap: var(--space-2);
  padding: 0 var(--space-4);
  border-radius: var(--radius-md);
  font-weight: 650;
}

.primary-command {
  border: 1px solid var(--color-action);
  color: var(--color-neutral-0);
  background: var(--color-action);
}

.secondary-command {
  border: 1px solid var(--color-border);
  background: var(--color-neutral-0);
}

.user-editor-page :deep(.inline-alert) {
  margin-bottom: var(--space-4);
}

.detail-error {
  display: grid;
  gap: var(--space-5);
  padding-top: var(--space-2);
}

.inline-command {
  margin-top: var(--space-3);
  padding: 0;
  border: 0;
  color: var(--color-action);
  background: transparent;
  font-weight: 650;
}

.user-form {
  display: grid;
  gap: var(--space-5);
  padding-top: var(--space-2);
}

.field-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--space-5);
}

.form-field {
  display: grid;
  gap: var(--space-2);
  min-width: 0;
}

.form-field > span {
  color: var(--color-text-secondary);
  font-size: 13px;
  font-weight: 650;
}

.form-field input,
.form-field select {
  width: 100%;
  min-height: 40px;
  padding: 0 var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-neutral-0);
}

.form-field input[aria-invalid='true'] {
  border-color: var(--color-error);
}

.form-field--static strong {
  min-height: 40px;
  padding: var(--space-2) var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-neutral-50);
  font-size: 14px;
}

.field-error {
  color: var(--color-error);
  font-size: 12px;
}

.status-switch {
  display: flex;
  width: fit-content;
  min-height: 40px;
  align-items: center;
  gap: var(--space-2);
  padding: 0 var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-neutral-0);
  font-weight: 650;
}

.status-switch > span {
  position: relative;
  width: 32px;
  height: 18px;
  border-radius: 9px;
  background: var(--color-neutral-200);
}

.status-switch > span::after {
  position: absolute;
  top: 3px;
  left: 3px;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: white;
  content: '';
  transition: transform 120ms ease;
}

.status-switch[aria-checked='true'] > span {
  background: var(--color-success);
}

.status-switch[aria-checked='true'] > span::after {
  transform: translateX(14px);
}

.status-switch:disabled,
.primary-command:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.temporary-note {
  margin: calc(var(--space-2) * -1) 0 0;
  color: var(--color-text-secondary);
  font-size: 13px;
}

small {
  display: block;
}

@media (max-width: 640px) {
  .user-editor-page {
    padding: var(--space-4);
  }

  .user-editor-page__header {
    align-items: stretch;
    flex-direction: column;
  }

  .header-actions > button {
    flex: 1;
  }

  .field-grid {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
