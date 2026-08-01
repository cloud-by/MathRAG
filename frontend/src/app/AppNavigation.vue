<script setup lang="ts">
import {
  Activity,
  Files,
  Library,
  MessageSquarePlus,
  MessagesSquare,
  Users,
} from '@lucide/vue'
import { computed } from 'vue'
import { RouterLink, useRoute } from 'vue-router'

import { useAuth } from '../features/auth/useAuth'

defineProps<{ compact?: boolean }>()

const emit = defineEmits<{
  navigate: []
}>()

const auth = useAuth()
const route = useRoute()

const primaryItems = [
  { href: '/chat', icon: MessageSquarePlus, label: '新建问答' },
  { href: '/conversations', icon: MessagesSquare, label: '会话记录' },
]
const adminItems = [
  { href: '/knowledge', icon: Library, label: '知识库' },
  { href: '/documents', icon: Files, label: '文档管理' },
  { href: '/jobs', icon: Activity, label: '摄取任务' },
]
const managerItems = [{ href: '/users', icon: Users, label: '用户管理' }]
const isUserManager = computed(
  () =>
    auth.state.value.status === 'authenticated' &&
    ['teacher', 'admin'].includes(auth.state.value.user.role),
)
const isAdmin = computed(
  () =>
    auth.state.value.status === 'authenticated' &&
    auth.state.value.user.role === 'admin',
)

function isCurrent(href: string): boolean {
  return route.path === href || route.path.startsWith(`${href}/`)
}
</script>

<template>
  <div class="app-navigation" :class="{ 'app-navigation--compact': compact }">
    <RouterLink
      class="app-navigation__brand"
      to="/chat"
      @click="emit('navigate')"
    >
      <span class="app-navigation__mark" aria-hidden="true">M</span>
      <span>MathRAG</span>
    </RouterLink>

    <nav aria-label="主导航">
      <p class="app-navigation__section-label">学习</p>
      <RouterLink
        v-for="item in primaryItems"
        :key="item.href"
        class="app-navigation__link"
        :class="{ 'app-navigation__link--current': isCurrent(item.href) }"
        :to="item.href"
        :aria-current="isCurrent(item.href) ? 'page' : undefined"
        @click="emit('navigate')"
      >
        <component :is="item.icon" :size="19" aria-hidden="true" />
        <span>{{ item.label }}</span>
      </RouterLink>

      <template v-if="isUserManager">
        <p
          class="app-navigation__section-label app-navigation__section-label--admin"
        >
          管理
        </p>
        <RouterLink
          v-for="item in managerItems"
          :key="item.href"
          class="app-navigation__link"
          :class="{ 'app-navigation__link--current': isCurrent(item.href) }"
          :to="item.href"
          :aria-current="isCurrent(item.href) ? 'page' : undefined"
          @click="emit('navigate')"
        >
          <component :is="item.icon" :size="19" aria-hidden="true" />
          <span>{{ item.label }}</span>
        </RouterLink>
        <RouterLink
          v-for="item in isAdmin ? adminItems : []"
          :key="item.href"
          class="app-navigation__link"
          :class="{ 'app-navigation__link--current': isCurrent(item.href) }"
          :to="item.href"
          :aria-current="isCurrent(item.href) ? 'page' : undefined"
          @click="emit('navigate')"
        >
          <component :is="item.icon" :size="19" aria-hidden="true" />
          <span>{{ item.label }}</span>
        </RouterLink>
      </template>
    </nav>
  </div>
</template>

<style scoped>
.app-navigation {
  display: flex;
  height: 100%;
  flex-direction: column;
  padding: var(--space-5) var(--space-3);
  color: #edf1f8;
  background: #253252;
}

.app-navigation__brand {
  display: flex;
  min-height: 42px;
  align-items: center;
  gap: var(--space-3);
  margin: 0 var(--space-2) var(--space-8);
  color: #ffffff;
  font-size: 18px;
  font-weight: 700;
  text-decoration: none;
}

.app-navigation__mark {
  display: grid;
  width: 34px;
  height: 34px;
  place-items: center;
  border: 1px solid rgb(255 255 255 / 40%);
  border-radius: var(--radius-md);
}

.app-navigation__section-label {
  margin: 0 var(--space-3) var(--space-2);
  color: rgb(255 255 255 / 48%);
  font-size: 11px;
  font-weight: 700;
}

.app-navigation__section-label--admin {
  margin-top: var(--space-6);
}

.app-navigation__link {
  display: flex;
  min-height: 42px;
  align-items: center;
  gap: var(--space-3);
  padding: 0 var(--space-3);
  border-radius: var(--radius-md);
  color: rgb(255 255 255 / 72%);
  font-size: 14px;
  font-weight: 600;
  text-decoration: none;
}

.app-navigation__link:hover {
  color: #ffffff;
  background: rgb(255 255 255 / 8%);
}

.app-navigation__link--current {
  color: #ffffff;
  background: rgb(255 255 255 / 13%);
}

.app-navigation--compact {
  min-height: 100%;
}
</style>
