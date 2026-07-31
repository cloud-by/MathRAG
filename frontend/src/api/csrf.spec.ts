import { describe, expect, it } from 'vitest'

import { readCsrfToken, requestNeedsCsrf } from './csrf'

describe('CSRF helpers', () => {
  it('ignores unrelated malformed cookies and decodes the CSRF token', () => {
    expect(
      readCsrfToken('unrelated=%E0%A4%A; mathrag_csrf=valid%20token'),
    ).toBe('valid token')
  })

  it('requires CSRF only for unsafe non-login requests', () => {
    expect(requestNeedsCsrf('GET', '/api/v1/example')).toBe(false)
    expect(requestNeedsCsrf('POST', '/api/v1/auth/login')).toBe(false)
    expect(requestNeedsCsrf('PATCH', '/api/v1/example')).toBe(true)
    expect(requestNeedsCsrf('DELETE', '/api/v1/example')).toBe(true)
  })
})
