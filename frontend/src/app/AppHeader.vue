<script setup lang="ts">
import { ChevronDown, LogOut, Menu, UserRound } from '@lucide/vue'
import { nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'

import IconButton from '../components/IconButton.vue'
import { useAuth } from '../features/auth/useAuth'

defineProps<{ title: string }>()

const emit = defineEmits<{
  openNavigation: []
}>()

const auth = useAuth()
const router = useRouter()
const navigationButton = ref<InstanceType<typeof IconButton> | null>(null)
const userMenuRoot = ref<HTMLElement | null>(null)
const userMenuButton = ref<HTMLButtonElement | null>(null)
const userMenuOpen = ref(false)
const loggingOut = ref(false)

const user =
  auth.state.value.status === 'authenticated' ? auth.state.value.user : null

function closeUserMenu(restoreFocus = false): void {
  userMenuOpen.value = false
  if (restoreFocus) void nextTick(() => userMenuButton.value?.focus())
}

function onDocumentPointerDown(event: PointerEvent): void {
  if (
    userMenuOpen.value &&
    event.target instanceof Node &&
    !userMenuRoot.value?.contains(event.target)
  ) {
    closeUserMenu()
  }
}

async function logout(): Promise<void> {
  loggingOut.value = true
  try {
    await auth.logout()
    await router.replace('/login')
  } finally {
    loggingOut.value = false
    closeUserMenu()
  }
}

onMounted(() => document.addEventListener('pointerdown', onDocumentPointerDown))
onBeforeUnmount(() =>
  document.removeEventListener('pointerdown', onDocumentPointerDown),
)

defineExpose({
  focusNavigationToggle: () => navigationButton.value?.focus(),
})
</script>

<template>
  <header class="app-header">
    <div class="app-header__title-group">
      <IconButton
        ref="navigationButton"
        class="app-header__menu-button"
        label="打开导航"
        @click="emit('openNavigation')"
      >
        <Menu :size="20" aria-hidden="true" />
      </IconButton>
      <h1>{{ title }}</h1>
    </div>

    <div v-if="user" ref="userMenuRoot" class="user-menu">
      <button
        ref="userMenuButton"
        class="user-menu__trigger"
        type="button"
        aria-haspopup="menu"
        :aria-expanded="userMenuOpen"
        :aria-label="`${user.username}，打开用户菜单`"
        @click="userMenuOpen = !userMenuOpen"
        @keydown.esc="closeUserMenu(true)"
      >
        <span class="user-menu__avatar" aria-hidden="true">
          <UserRound :size="16" />
        </span>
        <span class="user-menu__name">{{ user.username }}</span>
        <ChevronDown :size="15" aria-hidden="true" />
      </button>

      <div v-if="userMenuOpen" class="user-menu__panel" role="menu">
        <button
          type="button"
          role="menuitem"
          :disabled="loggingOut"
          @click="logout"
          @keydown.esc="closeUserMenu(true)"
        >
          <LogOut :size="16" aria-hidden="true" />
          <span>{{ loggingOut ? '正在退出' : '退出登录' }}</span>
        </button>
      </div>
    </div>
  </header>
</template>

<style scoped>
.app-header {
  position: sticky;
  z-index: 20;
  top: 0;
  display: flex;
  min-height: 58px;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-4);
  padding: 0 var(--space-6);
  border-bottom: 1px solid var(--color-border);
  background: rgb(255 255 255 / 96%);
}

.app-header__title-group {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: var(--space-2);
}

.app-header h1 {
  overflow: hidden;
  margin: 0;
  font-size: 18px;
  letter-spacing: 0;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.app-header__menu-button {
  display: none;
}

.user-menu {
  position: relative;
}

.user-menu__trigger {
  display: flex;
  min-height: 38px;
  align-items: center;
  gap: var(--space-2);
  padding: 0 var(--space-2);
  border: 1px solid transparent;
  border-radius: var(--radius-md);
  background: transparent;
}

.user-menu__trigger:hover,
.user-menu__trigger[aria-expanded='true'] {
  border-color: var(--color-border);
  background: var(--color-neutral-50);
}

.user-menu__avatar {
  display: grid;
  width: 28px;
  height: 28px;
  place-items: center;
  border-radius: 50%;
  color: #ffffff;
  background: var(--color-action);
}

.user-menu__name {
  max-width: 160px;
  overflow: hidden;
  font-size: 13px;
  font-weight: 600;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.user-menu__panel {
  position: absolute;
  top: calc(100% + var(--space-1));
  right: 0;
  width: 168px;
  padding: var(--space-1);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-neutral-0);
  box-shadow: 0 12px 30px rgb(23 32 51 / 14%);
}

.user-menu__panel button {
  display: flex;
  width: 100%;
  min-height: 38px;
  align-items: center;
  gap: var(--space-2);
  padding: 0 var(--space-3);
  border: 0;
  border-radius: var(--radius-sm);
  color: var(--color-error);
  background: transparent;
  font-size: 13px;
  font-weight: 600;
}

.user-menu__panel button:hover:not(:disabled) {
  background: #fff4f4;
}

@media (max-width: 900px) {
  .app-header {
    padding: 0 var(--space-4);
  }

  .app-header__menu-button {
    display: inline-grid;
  }

  .user-menu__name {
    display: none;
  }
}
</style>
