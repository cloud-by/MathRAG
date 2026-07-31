<script setup lang="ts">
import { X } from '@lucide/vue'
import { nextTick, ref } from 'vue'
import { RouterView, useRoute } from 'vue-router'

import IconButton from '../components/IconButton.vue'
import AppHeader from './AppHeader.vue'
import AppNavigation from './AppNavigation.vue'

const route = useRoute()
const header = ref<InstanceType<typeof AppHeader> | null>(null)
const drawer = ref<HTMLElement | null>(null)
const navigationOpen = ref(false)

function pageTitle(): string {
  return typeof route.meta.title === 'string' ? route.meta.title : 'MathRAG'
}

function drawerFocusableElements(): HTMLElement[] {
  if (!drawer.value) return []
  return Array.from(
    drawer.value.querySelectorAll<HTMLElement>(
      'button:not(:disabled), a[href], [tabindex]:not([tabindex="-1"])',
    ),
  )
}

async function openNavigation(): Promise<void> {
  navigationOpen.value = true
  await nextTick()
  drawerFocusableElements()[0]?.focus()
}

async function closeNavigation(): Promise<void> {
  navigationOpen.value = false
  await nextTick()
  header.value?.focusNavigationToggle()
}

function onDrawerKeydown(event: KeyboardEvent): void {
  if (event.key === 'Escape') {
    event.preventDefault()
    void closeNavigation()
    return
  }
  if (event.key !== 'Tab') return

  const focusable = drawerFocusableElements()
  const first = focusable[0]
  const last = focusable.at(-1)
  if (!first || !last) return
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault()
    last.focus()
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault()
    first.focus()
  }
}
</script>

<template>
  <div class="app-shell">
    <aside class="app-shell__desktop-navigation">
      <AppNavigation />
    </aside>

    <div class="app-shell__workspace">
      <AppHeader
        ref="header"
        :title="pageTitle()"
        @open-navigation="openNavigation"
      />
      <div class="app-shell__content">
        <RouterView />
      </div>
    </div>
  </div>

  <Teleport to="body">
    <div
      v-if="navigationOpen"
      class="navigation-backdrop"
      @mousedown.self="closeNavigation"
    >
      <aside
        ref="drawer"
        class="navigation-drawer"
        role="dialog"
        aria-modal="true"
        aria-label="主导航"
        tabindex="-1"
        @keydown="onDrawerKeydown"
      >
        <IconButton
          class="navigation-drawer__close"
          label="关闭导航"
          @click="closeNavigation"
        >
          <X :size="20" aria-hidden="true" />
        </IconButton>
        <AppNavigation compact @navigate="closeNavigation" />
      </aside>
    </div>
  </Teleport>
</template>

<style scoped>
.app-shell {
  display: grid;
  grid-template-columns: 216px minmax(0, 1fr);
  min-height: 100vh;
  background: var(--color-neutral-50);
}

.app-shell__desktop-navigation {
  position: sticky;
  top: 0;
  height: 100vh;
}

.app-shell__workspace {
  min-width: 0;
}

.app-shell__content {
  min-width: 0;
  min-height: calc(100vh - 58px);
}

.navigation-backdrop {
  position: fixed;
  z-index: 80;
  inset: 0;
  display: none;
  background: rgb(23 32 51 / 42%);
}

.navigation-drawer {
  position: relative;
  width: min(84vw, 280px);
  height: 100%;
  outline: 0;
  box-shadow: 14px 0 38px rgb(23 32 51 / 20%);
}

.navigation-drawer__close {
  position: absolute;
  z-index: 1;
  top: var(--space-4);
  right: var(--space-3);
  color: #ffffff;
}

@media (max-width: 900px) {
  .app-shell {
    grid-template-columns: minmax(0, 1fr);
  }

  .app-shell__desktop-navigation {
    display: none;
  }

  .navigation-backdrop {
    display: block;
  }
}
</style>
