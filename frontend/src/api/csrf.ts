const CSRF_COOKIE_NAMES = ['__Host-mathrag_csrf', 'mathrag_csrf'] as const
const CSRF_COOKIE_NAME_SET = new Set<string>(CSRF_COOKIE_NAMES)

export function readCsrfToken(cookieHeader = document.cookie): string | null {
  const cookies = new Map<string, string>()
  for (const part of cookieHeader.split(';')) {
    const separator = part.indexOf('=')
    if (separator < 0) {
      continue
    }
    const name = part.slice(0, separator).trim()
    const value = part.slice(separator + 1).trim()
    if (CSRF_COOKIE_NAME_SET.has(name)) {
      cookies.set(name, value)
    }
  }

  for (const name of CSRF_COOKIE_NAMES) {
    const token = cookies.get(name)
    if (token) {
      try {
        return decodeURIComponent(token)
      } catch {
        continue
      }
    }
  }
  return null
}

export function requestNeedsCsrf(method: string, path: string): boolean {
  const normalizedMethod = method.toUpperCase()
  if (!['POST', 'PATCH', 'DELETE'].includes(normalizedMethod)) {
    return false
  }
  return path !== '/api/v1/auth/login'
}
