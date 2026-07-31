import { defineComponent, h } from 'vue'
import {
  createRouter,
  createWebHistory,
  type Router,
  type RouterHistory,
  type RouteRecordRaw,
} from 'vue-router'

import { subscribeUnauthorized, type UnauthorizedListener } from '../api/client'
import { authController, type AuthController } from '../features/auth/useAuth'

const EMPTY_ROUTE = defineComponent({
  name: 'PendingFeaturePage',
  setup() {
    return () => h('main', { class: 'pending-feature-page' })
  },
})

const routes: RouteRecordRaw[] = [
  { path: '/', name: 'home', component: EMPTY_ROUTE },
  {
    path: '/login',
    name: 'login',
    component: () => import('../features/auth/LoginPage.vue'),
  },
  {
    path: '/chat',
    name: 'chat',
    component: EMPTY_ROUTE,
    meta: { requiresAuth: true },
  },
  {
    path: '/conversations',
    name: 'conversations',
    component: EMPTY_ROUTE,
    meta: { requiresAuth: true },
  },
  {
    path: '/conversations/:id',
    name: 'conversation',
    component: EMPTY_ROUTE,
    meta: { requiresAuth: true },
  },
  {
    path: '/knowledge',
    name: 'knowledge',
    component: EMPTY_ROUTE,
    meta: { requiresAdmin: true, requiresAuth: true },
  },
  {
    path: '/knowledge/new',
    name: 'knowledge-new',
    component: EMPTY_ROUTE,
    meta: { requiresAdmin: true, requiresAuth: true },
  },
  {
    path: '/knowledge/:id',
    name: 'knowledge-detail',
    component: EMPTY_ROUTE,
    meta: { requiresAdmin: true, requiresAuth: true },
  },
  {
    path: '/documents',
    name: 'documents',
    component: EMPTY_ROUTE,
    meta: { requiresAdmin: true, requiresAuth: true },
  },
  {
    path: '/jobs',
    name: 'jobs',
    component: EMPTY_ROUTE,
    meta: { requiresAdmin: true, requiresAuth: true },
  },
]

export function safeNextPath(value: unknown): string | null {
  if (typeof value !== 'string' || value.trim() !== value) {
    return null
  }
  if (
    !value.startsWith('/') ||
    value.startsWith('//') ||
    value.includes('\\')
  ) {
    return null
  }

  let decoded = value
  for (let pass = 0; pass < 3; pass += 1) {
    try {
      const nextDecoded = decodeURIComponent(decoded)
      if (nextDecoded === decoded) {
        break
      }
      decoded = nextDecoded
    } catch {
      return null
    }
  }
  if (
    !decoded.startsWith('/') ||
    decoded.startsWith('//') ||
    decoded.includes('\\') ||
    Array.from(decoded).some((character) => {
      const code = character.charCodeAt(0)
      return code <= 31 || code === 127
    })
  ) {
    return null
  }

  const localOrigin = 'http://mathrag.local'
  return new URL(value, localOrigin).origin === localOrigin ? value : null
}

function postLoginDestination(value: unknown): string {
  const destination = safeNextPath(value)
  return destination && !destination.startsWith('/login')
    ? destination
    : '/chat'
}

export interface CreateAppRouterOptions {
  auth?: AuthController
  history?: RouterHistory
  subscribeUnauthorized?: UnauthorizedSubscriber
}

export type UnauthorizedSubscriber = (
  listener: UnauthorizedListener,
) => () => void

export function createAppRouter(options: CreateAppRouterOptions = {}): Router {
  const auth = options.auth ?? authController
  const router = createRouter({
    history: options.history ?? createWebHistory(),
    routes,
  })
  const subscribe = options.subscribeUnauthorized ?? subscribeUnauthorized

  subscribe(() => {
    const currentRoute = router.currentRoute.value
    auth.invalidate()
    if (currentRoute.meta.requiresAuth && currentRoute.name !== 'login') {
      void router.replace({
        name: 'login',
        query: { next: currentRoute.fullPath },
      })
    }
  })

  router.beforeEach(async (to) => {
    if (to.name === 'login' && to.query.auth_error === '1') {
      return auth.state.value.status === 'authenticated'
        ? postLoginDestination(to.query.next)
        : true
    }
    try {
      await auth.bootstrap()
    } catch {
      const next =
        to.name === 'login' ? safeNextPath(to.query.next) : to.fullPath
      return {
        name: 'login',
        query: {
          auth_error: '1',
          ...(next ? { next } : {}),
        },
      }
    }
    const state = auth.state.value

    if (to.name === 'home') {
      return state.status === 'authenticated' ? '/chat' : '/login'
    }
    if (to.name === 'login') {
      return state.status === 'authenticated'
        ? postLoginDestination(to.query.next)
        : true
    }
    if (to.meta.requiresAuth && state.status !== 'authenticated') {
      return { name: 'login', query: { next: to.fullPath } }
    }
    if (
      to.meta.requiresAdmin &&
      (state.status !== 'authenticated' || state.user.role !== 'admin')
    ) {
      return '/chat'
    }
    return true
  })

  return router
}

const router = createAppRouter()

export default router
