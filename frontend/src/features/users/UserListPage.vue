<script setup lang="ts">
import { Pencil, Plus, RefreshCw } from '@lucide/vue'
import { computed, ref, watch } from 'vue'
import { RouterLink, useRoute, useRouter } from 'vue-router'

import EmptyState from '../../components/EmptyState.vue'
import IconButton from '../../components/IconButton.vue'
import InlineAlert from '../../components/InlineAlert.vue'
import LoadingState from '../../components/LoadingState.vue'
import PaginationControls from '../../components/PaginationControls.vue'
import { useAuth } from '../auth/useAuth'
import type { ManagedUser, UserRole, UserStatus } from './types'
import { useUserList } from './useUsers'

const PAGE_SIZE = 20
const ROLES = new Set<UserRole>(['student', 'teacher', 'admin'])
const STATUSES = new Set<UserStatus>(['active', 'disabled'])

const route = useRoute()
const router = useRouter()
const auth = useAuth()
const users = useUserList()
const queryInput = ref(typeof route.query.q === 'string' ? route.query.q : '')

const isAdmin = computed(
  () =>
    auth.state.value.status === 'authenticated' &&
    auth.state.value.user.role === 'admin',
)
const query = computed(() => {
  const value = route.query.q
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
})
const role = computed<UserRole | undefined>(() => {
  if (!isAdmin.value) return undefined
  const value = route.query.role
  return typeof value === 'string' && ROLES.has(value as UserRole)
    ? (value as UserRole)
    : undefined
})
const status = computed<UserStatus | undefined>(() => {
  const value = route.query.status
  return typeof value === 'string' && STATUSES.has(value as UserStatus)
    ? (value as UserStatus)
    : undefined
})
const page = computed(() => {
  const value = Number(route.query.page)
  return Number.isInteger(value) && value >= 1 ? value : 1
})

function updateFilters(values: {
  query?: string
  role?: UserRole
  status?: UserStatus
  page?: number
}): void {
  void router.push({
    name: 'users',
    query: {
      ...(values.query ? { q: values.query } : {}),
      ...(isAdmin.value && values.role ? { role: values.role } : {}),
      ...(values.status ? { status: values.status } : {}),
      page: String(values.page ?? 1),
    },
  })
}

function applyFilters(): void {
  updateFilters({
    query: queryInput.value.trim() || undefined,
    role: role.value,
    status: status.value,
  })
}

function setRole(event: Event): void {
  const value = (event.target as HTMLSelectElement).value
  updateFilters({
    query: query.value,
    role: value ? (value as UserRole) : undefined,
    status: status.value,
  })
}

function setStatus(event: Event): void {
  const value = (event.target as HTMLSelectElement).value
  updateFilters({
    query: query.value,
    role: role.value,
    status: value ? (value as UserStatus) : undefined,
  })
}

function setOffset(offset: number): void {
  updateFilters({
    query: query.value,
    role: role.value,
    status: status.value,
    page: Math.floor(offset / PAGE_SIZE) + 1,
  })
}

function roleLabel(value: UserRole): string {
  return { student: '学生', teacher: '教师', admin: '管理员' }[value]
}

function statusLabel(value: UserStatus): string {
  return value === 'active' ? '启用' : '禁用'
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat('zh-CN', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value))
}

function creatorLabel(user: ManagedUser): string {
  return user.created_by_username ?? '系统'
}

watch(
  [query, role, status, page, isAdmin],
  ([nextQuery, nextRole, nextStatus, nextPage]) => {
    queryInput.value = nextQuery ?? ''
    void users.load({
      query: nextQuery,
      role: nextRole,
      status: nextStatus,
      page: nextPage,
      pageSize: PAGE_SIZE,
    })
  },
  { immediate: true },
)
</script>

<template>
  <main class="users-page">
    <header class="users-page__toolbar">
      <form class="user-filters" @submit.prevent="applyFilters">
        <label>
          <span>搜索</span>
          <input
            v-model="queryInput"
            type="search"
            placeholder="用户名或邮箱"
          />
        </label>
        <label v-if="isAdmin">
          <span>角色</span>
          <select :value="role ?? ''" @change="setRole">
            <option value="">全部角色</option>
            <option value="student">学生</option>
            <option value="teacher">教师</option>
            <option value="admin">管理员</option>
          </select>
        </label>
        <label>
          <span>状态</span>
          <select :value="status ?? ''" @change="setStatus">
            <option value="">全部状态</option>
            <option value="active">启用</option>
            <option value="disabled">禁用</option>
          </select>
        </label>
        <button class="secondary-command" type="submit">应用筛选</button>
      </form>
      <div class="users-page__commands">
        <IconButton label="刷新用户列表" @click="users.refresh">
          <RefreshCw :size="18" aria-hidden="true" />
        </IconButton>
        <RouterLink class="primary-command" to="/users/new">
          <Plus :size="18" aria-hidden="true" />
          创建账号
        </RouterLink>
      </div>
    </header>

    <InlineAlert
      v-if="users.state.value.status === 'error'"
      tone="error"
      title="无法加载用户列表"
    >
      <p>{{ users.state.value.error.message }}</p>
      <small>请求编号：{{ users.state.value.error.requestId }}</small>
      <button class="inline-command" type="button" @click="users.refresh">
        重试
      </button>
    </InlineAlert>
    <LoadingState
      v-if="users.state.value.status === 'loading' && !users.state.value.data"
      label="正在加载用户列表"
    />
    <template v-else-if="users.state.value.data">
      <EmptyState
        v-if="!users.state.value.data.items.length"
        title="没有符合条件的账号"
      />
      <section v-else class="user-table-wrap" aria-label="用户列表">
        <table class="user-table">
          <thead>
            <tr>
              <th>用户名</th>
              <th>邮箱</th>
              <th>角色</th>
              <th>状态</th>
              <th>创建者</th>
              <th>创建时间</th>
              <th aria-label="操作"></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="item in users.state.value.data.items" :key="item.id">
              <td data-label="用户名">
                <RouterLink :to="`/users/${item.id}`">{{
                  item.username
                }}</RouterLink>
              </td>
              <td data-label="邮箱">{{ item.email || '未设置' }}</td>
              <td data-label="角色">{{ roleLabel(item.role) }}</td>
              <td data-label="状态">
                <span
                  class="status-badge"
                  :class="`status-badge--${item.status}`"
                >
                  {{ statusLabel(item.status) }}
                </span>
              </td>
              <td data-label="创建者">{{ creatorLabel(item) }}</td>
              <td data-label="创建时间">{{ formatDate(item.created_at) }}</td>
              <td class="user-table__actions" data-label="操作">
                <IconButton
                  :label="`编辑“${item.username}”`"
                  @click="router.push(`/users/${item.id}`)"
                >
                  <Pencil :size="17" aria-hidden="true" />
                </IconButton>
              </td>
            </tr>
          </tbody>
        </table>
        <PaginationControls
          :limit="PAGE_SIZE"
          :offset="(page - 1) * PAGE_SIZE"
          :total="users.state.value.data.total"
          @update:offset="setOffset"
        />
      </section>
    </template>
  </main>
</template>

<style scoped>
.users-page {
  width: min(100%, 1180px);
  margin: 0 auto;
  padding: var(--space-6);
}

.users-page__toolbar {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: var(--space-4);
  margin-bottom: var(--space-5);
}

.user-filters {
  display: flex;
  min-width: 0;
  flex: 1;
  flex-wrap: wrap;
  align-items: flex-end;
  gap: var(--space-3);
}

.user-filters label {
  display: grid;
  gap: var(--space-1);
  color: var(--color-text-secondary);
  font-size: 12px;
}

.user-filters input,
.user-filters select {
  min-height: 38px;
  padding: 0 var(--space-3);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-neutral-0);
}

.user-filters input {
  width: 210px;
}

.users-page__commands,
.primary-command,
.secondary-command {
  display: flex;
  align-items: center;
}

.users-page__commands {
  flex: 0 0 auto;
  gap: var(--space-2);
}

.primary-command,
.secondary-command {
  min-height: 38px;
  justify-content: center;
  gap: var(--space-2);
  padding: 0 var(--space-4);
  border-radius: var(--radius-md);
  font-weight: 650;
  text-decoration: none;
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

.user-table-wrap {
  overflow: hidden;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  background: var(--color-neutral-0);
}

.user-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  table-layout: fixed;
}

.user-table th,
.user-table td {
  padding: var(--space-3) var(--space-4);
  border-bottom: 1px solid var(--color-border);
  text-align: left;
  overflow-wrap: anywhere;
}

.user-table th {
  color: var(--color-text-secondary);
  background: var(--color-neutral-50);
  font-size: 12px;
}

.user-table th:nth-child(1) {
  width: 16%;
}

.user-table th:nth-child(2) {
  width: 21%;
}

.user-table th:nth-child(3),
.user-table th:nth-child(4) {
  width: 9%;
}

.user-table th:nth-child(5) {
  width: 14%;
}

.user-table th:nth-child(6) {
  width: 23%;
}

.user-table th:nth-child(7) {
  width: 8%;
}

.user-table td:first-child {
  font-weight: 650;
}

.user-table__actions {
  text-align: right;
}

.status-badge {
  display: inline-flex;
  min-height: 24px;
  align-items: center;
  padding: 0 var(--space-2);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  font-size: 12px;
  font-weight: 650;
}

.status-badge--active {
  color: var(--color-success);
  border-color: color-mix(in srgb, var(--color-success) 35%, white);
  background: color-mix(in srgb, var(--color-success) 8%, white);
}

.status-badge--disabled {
  color: var(--color-text-secondary);
  background: var(--color-neutral-50);
}

.user-table-wrap :deep(.pagination) {
  padding: var(--space-2) var(--space-3);
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
}

@media (max-width: 780px) {
  .users-page {
    padding: var(--space-4);
  }

  .users-page__toolbar {
    align-items: stretch;
    flex-direction: column-reverse;
  }

  .users-page__commands {
    justify-content: flex-end;
  }

  .user-filters label,
  .user-filters input {
    min-width: 0;
    flex: 1 1 140px;
    width: auto;
  }

  .user-table-wrap {
    border: 0;
    background: transparent;
  }

  .user-table,
  .user-table tbody,
  .user-table tr,
  .user-table td {
    display: block;
    width: 100%;
  }

  .user-table thead {
    display: none;
  }

  .user-table tr {
    margin-bottom: var(--space-3);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    background: var(--color-neutral-0);
  }

  .user-table td {
    display: grid;
    grid-template-columns: 88px minmax(0, 1fr);
    gap: var(--space-3);
    padding: var(--space-2) var(--space-3);
  }

  .user-table td::before {
    color: var(--color-text-secondary);
    content: attr(data-label);
    font-size: 12px;
    font-weight: 600;
  }

  .user-table__actions {
    align-items: center;
    text-align: left;
  }
}
</style>
