import { http, HttpResponse, delay } from 'msw'
import { setupServer } from 'msw/node'
import {
  afterAll,
  afterEach,
  beforeAll,
  describe,
  expect,
  it,
  vi,
} from 'vitest'

import { apiRequest } from './client'
import { ApiError } from './errors'

const server = setupServer()

function clearCookies(): void {
  for (const cookie of document.cookie.split(';')) {
    const name = cookie.split('=', 1)[0]?.trim()
    if (name) {
      document.cookie = `${name}=; Max-Age=0; path=/`
    }
  }
}

beforeAll(() => {
  server.listen({ onUnhandledRequest: 'error' })
  const interceptedFetch = globalThis.fetch
  vi.stubGlobal(
    'fetch',
    vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const resolved =
        typeof input === 'string' && input.startsWith('/')
          ? new URL(input, window.location.origin)
          : input
      return interceptedFetch(resolved, init)
    }),
  )
})

afterEach(() => {
  server.resetHandlers()
  clearCookies()
  vi.mocked(globalThis.fetch).mockClear()
})

afterAll(() => {
  vi.unstubAllGlobals()
  server.close()
})

describe('apiRequest', () => {
  it('uses a relative URL, same-origin credentials, and a request id', async () => {
    server.use(
      http.get('http://localhost:3000/api/v1/example', ({ request }) => {
        expect(request.headers.get('X-Request-ID')).toMatch(
          /^[A-Za-z0-9._-]{1,128}$/,
        )
        expect(request.headers.has('Origin')).toBe(false)
        return HttpResponse.json({ value: 'ok' })
      }),
    )

    await expect(
      apiRequest<{ value: string }>('/api/v1/example'),
    ).resolves.toEqual({ value: 'ok' })

    expect(globalThis.fetch).toHaveBeenCalledWith(
      '/api/v1/example',
      expect.objectContaining({ credentials: 'same-origin' }),
    )
  })

  it.each([
    ['mathrag_csrf', 'development-token'],
    ['__Host-mathrag_csrf', 'production-token'],
  ])('sends the %s cookie on unsafe requests', async (name, token) => {
    const attributes = name.startsWith('__Host-')
      ? '; Secure; Path=/'
      : '; Path=/'
    document.cookie = `${name}=${token}${attributes}`
    server.use(
      http.post('http://localhost:3000/api/v1/example', ({ request }) => {
        expect(request.headers.get('X-CSRF-Token')).toBe(token)
        return HttpResponse.json({ saved: true })
      }),
    )

    await apiRequest('/api/v1/example', {
      method: 'POST',
      body: { value: 1 },
    })
  })

  it('does not send a CSRF header for login', async () => {
    document.cookie = 'mathrag_csrf=unused-token; path=/'
    server.use(
      http.post('http://localhost:3000/api/v1/auth/login', ({ request }) => {
        expect(request.headers.has('X-CSRF-Token')).toBe(false)
        return HttpResponse.json({ authenticated: true })
      }),
    )

    await apiRequest('/api/v1/auth/login', {
      method: 'POST',
      body: { username: 'student', password: 'secret' },
    })
  })

  it('lets the browser set the FormData content type', async () => {
    const body = new FormData()
    body.set('file', new Blob(['content']), 'lesson.pdf')
    server.use(
      http.post('http://localhost:3000/api/v1/documents', ({ request }) => {
        expect(request.headers.get('Content-Type')).toMatch(
          /^multipart\/form-data; boundary=/,
        )
        return HttpResponse.json({ accepted: true }, { status: 202 })
      }),
    )

    await apiRequest('/api/v1/documents', { method: 'POST', body })

    const init = vi.mocked(globalThis.fetch).mock.calls.at(-1)?.[1]
    expect(new Headers(init?.headers).has('Content-Type')).toBe(false)
  })

  it('returns undefined for a 204 response', async () => {
    server.use(
      http.delete('http://localhost:3000/api/v1/example', () => {
        return new HttpResponse(null, { status: 204 })
      }),
    )

    await expect(
      apiRequest('/api/v1/example', { method: 'DELETE' }),
    ).resolves.toBeUndefined()
  })

  it('converts the v1 error envelope into ApiError', async () => {
    server.use(
      http.get('http://localhost:3000/api/v1/example', () => {
        return HttpResponse.json(
          {
            error: {
              code: 'EXAMPLE_ERROR',
              message: '示例失败。',
              request_id: 'request-123',
              details: { field: 'question' },
            },
          },
          { status: 409 },
        )
      }),
    )

    await expect(apiRequest('/api/v1/example')).rejects.toMatchObject({
      name: 'ApiError',
      code: 'EXAMPLE_ERROR',
      message: '示例失败。',
      requestId: 'request-123',
      status: 409,
      details: { field: 'question' },
    } satisfies Partial<ApiError>)
  })

  it('preserves AbortError for caller state machines', async () => {
    server.use(
      http.get('http://localhost:3000/api/v1/slow', async () => {
        await delay('infinite')
        return HttpResponse.json({ value: 'late' })
      }),
    )
    const controller = new AbortController()
    const request = apiRequest('/api/v1/slow', { signal: controller.signal })

    controller.abort()

    await expect(request).rejects.toMatchObject({ name: 'AbortError' })
  })
})
