export interface ApiErrorBody {
  code: string
  message: string
  request_id: string
  details: unknown
}

export interface ApiErrorEnvelope {
  error: ApiErrorBody
}

export class ApiError extends Error {
  readonly code: string
  readonly requestId: string
  readonly status: number
  readonly details: unknown

  constructor(options: {
    code: string
    message: string
    requestId: string
    status: number
    details: unknown
  }) {
    super(options.message)
    this.name = 'ApiError'
    this.code = options.code
    this.requestId = options.requestId
    this.status = options.status
    this.details = options.details
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

export function isApiErrorEnvelope(value: unknown): value is ApiErrorEnvelope {
  if (!isRecord(value) || !isRecord(value.error)) {
    return false
  }
  return (
    typeof value.error.code === 'string' &&
    typeof value.error.message === 'string' &&
    typeof value.error.request_id === 'string' &&
    'details' in value.error
  )
}
