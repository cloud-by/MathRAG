import 'vue-router'

export {}

declare module 'vue-router' {
  interface RouteMeta {
    requiresAdmin?: boolean
    requiresAuth?: boolean
    requiresUserManager?: boolean
    title?: string
  }
}
