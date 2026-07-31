import { ApiError, isApiErrorEnvelope, type ApiErrorEnvelope } from './errors'
import { readCsrfToken, requestNeedsCsrf } from './csrf'

export interface ApiRequestOptions<TBody = unknown> {
  method?: 'GET' | 'POST' | 'PATCH' | 'DELETE'
  body?: TBody | FormData
  signal?: AbortSignal
  requestId?: string
}

export interface UnauthorizedEvent {
  path: `/api/${string}`
  error: ApiError
}

export type UnauthorizedListener = (event: UnauthorizedEvent) => void

const unauthorizedListeners = new Set<UnauthorizedListener>()

export function subscribeUnauthorized(
  listener: UnauthorizedListener,
): () => void {
  unauthorizedListeners.add(listener)
  return () => unauthorizedListeners.delete(listener)
}

function notifyUnauthorized(event: UnauthorizedEvent): void {
  for (const listener of unauthorizedListeners) {
    try {
      listener(event)
    } catch {
      continue
    }
  }
}

function createRequestId(requestId?: string): string {
  const supplied = requestId?.trim()
  return supplied || crypto.randomUUID()
}

function createBodyAndHeaders<TBody>(body: TBody | FormData | undefined): {
  body?: BodyInit
  headers: Headers
} {
  const headers = new Headers()
  if (body === undefined) {
    return { headers }
  }
  if (body instanceof FormData) {
    return { body, headers }
  }
  headers.set('Content-Type', 'application/json')
  return { body: JSON.stringify(body), headers }
}

async function readResponseBody(response: Response): Promise<unknown> {
  const text = await response.text()
  if (!text) {
    return undefined
  }
  if (response.headers.get('Content-Type')?.includes('application/json')) {
    try {
      return JSON.parse(text) as unknown
    } catch {
      return text
    }
  }
  return text
}

function fallbackMessage(payload: unknown, status: number): string {
  if (
    typeof payload === 'object' &&
    payload !== null &&
    'detail' in payload &&
    typeof payload.detail === 'string'
  ) {
    return payload.detail
  }
  return `请求失败（HTTP ${status}）。`
}

function toApiError(
  response: Response,
  payload: unknown,
  requestId: string,
): ApiError {
  if (isApiErrorEnvelope(payload)) {
    const envelope: ApiErrorEnvelope = payload
    return new ApiError({
      code: envelope.error.code,
      message: envelope.error.message,
      requestId: envelope.error.request_id,
      status: response.status,
      details: envelope.error.details,
    })
  }
  return new ApiError({
    code: 'HTTP_ERROR',
    message: fallbackMessage(payload, response.status),
    requestId: response.headers.get('X-Request-ID') || requestId,
    status: response.status,
    details: payload,
  })
}

export async function apiRequest<TResponse, TBody = unknown>(
  path: `/api/${string}`,
  options: ApiRequestOptions<TBody> = {},
): Promise<TResponse> {
  const method = options.method ?? 'GET'
  const requestId = createRequestId(options.requestId)
  const request = createBodyAndHeaders(options.body)
  request.headers.set('X-Request-ID', requestId)

  if (requestNeedsCsrf(method, path)) {
    const csrfToken = readCsrfToken()
    if (csrfToken) {
      request.headers.set('X-CSRF-Token', csrfToken)
    }
  }

  const response = await fetch(path, {
    method,
    headers: request.headers,
    body: request.body,
    signal: options.signal,
    credentials: 'same-origin',
  })

  if (response.status === 204) {
    return undefined as TResponse
  }

  const payload = await readResponseBody(response)
  if (!response.ok) {
    const error = toApiError(response, payload, requestId)
    if (response.status === 401) {
      notifyUnauthorized({ path, error })
    }
    throw error
  }
  return payload as TResponse
}
