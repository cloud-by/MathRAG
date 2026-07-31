import 'vue-router'

export {}

declare module 'vue-router' {
  interface RouteMeta {
    requiresAdmin?: boolean
    requiresAuth?: boolean
    title?: string
  }
}
