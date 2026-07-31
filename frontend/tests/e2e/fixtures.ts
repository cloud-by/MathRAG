import type { Page, Route } from '@playwright/test'

export const IDS = {
  conversation: '11111111-1111-4111-8111-111111111111',
  question: '22222222-2222-4222-8222-222222222222',
  answer: '33333333-3333-4333-8333-333333333333',
  run: '44444444-4444-4444-8444-444444444444',
  knowledge: '55555555-5555-4555-8555-555555555555',
  document: '66666666-6666-4666-8666-666666666666',
  pendingJob: '77777777-7777-4777-8777-777777777777',
  failedJob: '88888888-8888-4888-8888-888888888888',
  user: '99999999-9999-4999-8999-999999999999',
}

const now = '2026-07-31T12:30:00Z'

function errorBody(code: string, message: string) {
  return {
    error: { code, message, request_id: `e2e-${code}`, details: {} },
  }
}

function json(route: Route, body: unknown, status = 200) {
  return route.fulfill({ status, contentType: 'application/json', json: body })
}

function reference() {
  return {
    rank: 1,
    score: 0.96,
    index: null,
    chunk_id: 'k0001_chunk_0',
    source_id: 'k0001',
    category: 'algebra',
    title: '一元二次方程求根公式',
    keywords: ['方程', '求根公式'],
    content: '一元二次方程可以使用求根公式求解。',
    example: '$x^2-5x+6=0$',
    steps: ['计算判别式', '代入公式'],
    difficulty: 'medium',
    answer_context: '求根公式上下文',
    retrieval_text: '一元二次方程 求根公式',
    source_line: 1,
    metadata: {},
  }
}

function answerResponse(clientRequestId: string) {
  return {
    conversation_id: IDS.conversation,
    question_message_id: IDS.question,
    answer_message_id: IDS.answer,
    rag_run_id: IDS.run,
    client_request_id: clientRequestId,
    question: '如何解一元二次方程？',
    answer: '使用求根公式 $x=\\frac{-b\\pm\\sqrt{b^2-4ac}}{2a}$。',
    steps: ['整理为标准形式', '计算判别式', '代入求根公式'],
    used_knowledge: ['一元二次方程求根公式'],
    related_questions: ['判别式如何判断根的个数？'],
    references: [reference()],
    agentic_plan: {
      strategy: '检索定义与例题',
      retrieval_queries: ['一元二次方程 求根公式'],
    },
    reasoning_content: '先识别系数，再计算判别式。',
  }
}

function conversation(title = '二次方程复习') {
  return {
    id: IDS.conversation,
    title,
    status: 'active',
    created_at: now,
    updated_at: now,
  }
}

function knowledgeItem(overrides: Record<string, unknown> = {}) {
  return {
    id: IDS.knowledge,
    legacy_id: 'k0001',
    owner_id: IDS.user,
    category: 'algebra',
    title: '一元二次方程求根公式',
    keywords: ['方程', '判别式'],
    content: '当 $a\\ne0$ 时可以使用求根公式。',
    example: '$x^2-5x+6=0$',
    steps: ['计算判别式', '代入求根公式'],
    difficulty: 'medium',
    visibility: 'public',
    status: 'ready',
    revision: 1,
    created_at: now,
    updated_at: now,
    ...overrides,
  }
}

function documentItem() {
  return {
    id: IDS.document,
    owner_id: IDS.user,
    original_name: 'lesson.pdf',
    mime_type: 'application/pdf',
    size_bytes: 2048,
    sha256: 'a'.repeat(64),
    status: 'processing',
    created_at: now,
    updated_at: now,
  }
}

function job(id: string, status: string) {
  return {
    id,
    requested_by: IDS.user,
    document_id: IDS.document,
    job_type: 'pdf',
    status,
    progress: status === 'failed' ? 18 : status === 'completed' ? 100 : 35,
    attempt_count: 1,
    error_code: status === 'failed' ? 'PDF_PARSE_FAILED' : null,
    error_message: status === 'failed' ? '无法解析 PDF。' : null,
    started_at: status === 'pending' ? null : now,
    finished_at: ['failed', 'completed', 'cancelled'].includes(status)
      ? now
      : null,
    created_at: now,
    updated_at: now,
  }
}

export interface MockApiState {
  loggedIn: boolean
  role: 'admin' | 'student'
  conversationTitle: string
  conversationArchived: boolean
  knowledge: ReturnType<typeof knowledgeItem>
  conflictNextKnowledgeUpdate: boolean
  failNextChat: boolean
  holdNextChat: boolean
  releaseChat: (() => void) | null
  chatRequestIds: string[]
  pendingJobStatus: string
  failedJobStatus: string
}

export async function installMockApi(
  page: Page,
  options: Partial<Pick<MockApiState, 'loggedIn' | 'role'>> = {},
): Promise<MockApiState> {
  const state: MockApiState = {
    loggedIn: options.loggedIn ?? true,
    role: options.role ?? 'admin',
    conversationTitle: '二次方程复习',
    conversationArchived: false,
    knowledge: knowledgeItem(),
    conflictNextKnowledgeUpdate: false,
    failNextChat: false,
    holdNextChat: false,
    releaseChat: null,
    chatRequestIds: [],
    pendingJobStatus: 'pending',
    failedJobStatus: 'failed',
  }

  await page.route('**/api/v1/**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname
    const method = request.method()
    const user = {
      id: IDS.user,
      username: state.role === 'admin' ? 'admin' : 'student',
      email: `${state.role}@example.com`,
      role: state.role,
      status: 'active',
    }

    if (path === '/api/v1/auth/me') {
      return state.loggedIn
        ? json(route, user)
        : json(route, errorBody('AUTH_REQUIRED', '请先登录。'), 401)
    }
    if (path === '/api/v1/auth/login' && method === 'POST') {
      const body = request.postDataJSON() as { username: string }
      if (body.username === 'invalid') {
        return json(route, errorBody('AUTH_INVALID', '用户名或密码错误。'), 401)
      }
      state.loggedIn = true
      return json(route, user)
    }
    if (path === '/api/v1/auth/logout' && method === 'POST') {
      state.loggedIn = false
      return route.fulfill({ status: 204 })
    }
    if (!state.loggedIn) {
      return json(route, errorBody('AUTH_REQUIRED', '请先登录。'), 401)
    }

    if (path === '/api/v1/conversations' && method === 'GET') {
      const requestedStatus = url.searchParams.get('status') ?? 'active'
      const visible = state.conversationArchived
        ? requestedStatus === 'archived'
        : requestedStatus === 'active'
      return json(route, {
        items: visible ? [conversation(state.conversationTitle)] : [],
        page: 1,
        page_size: 20,
        total: visible ? 1 : 0,
      })
    }
    if (path === '/api/v1/conversations' && method === 'POST') {
      const body = request.postDataJSON() as { title: string }
      state.conversationTitle = body.title
      state.conversationArchived = false
      return json(route, conversation(body.title), 201)
    }
    if (path.endsWith('/messages') && method === 'GET') {
      return json(route, {
        items: [
          {
            id: IDS.question,
            conversation_id: IDS.conversation,
            role: 'user',
            status: 'completed',
            content: '如何解一元二次方程？',
            model_metadata: {},
            created_at: now,
          },
          {
            id: IDS.answer,
            conversation_id: IDS.conversation,
            role: 'assistant',
            status: 'completed',
            content: '使用求根公式。',
            model_metadata: { response: answerResponse('history-request') },
            created_at: now,
          },
        ],
        page: 1,
        page_size: 50,
        total: 2,
      })
    }
    if (path === `/api/v1/conversations/${IDS.conversation}`) {
      if (method === 'PATCH') {
        const body = request.postDataJSON() as { title?: string }
        state.conversationTitle = body.title ?? state.conversationTitle
        return json(route, conversation(state.conversationTitle))
      }
      if (method === 'DELETE') {
        state.conversationArchived = true
        return route.fulfill({ status: 204 })
      }
      return json(route, conversation(state.conversationTitle))
    }
    if (path === '/api/v1/chat' && method === 'POST') {
      const body = request.postDataJSON() as { client_request_id: string }
      state.chatRequestIds.push(body.client_request_id)
      if (state.failNextChat) {
        state.failNextChat = false
        return json(
          route,
          errorBody('RAG_UNAVAILABLE', '问答服务暂不可用。'),
          503,
        )
      }
      if (state.holdNextChat) {
        state.holdNextChat = false
        await new Promise<void>((resolve) => {
          state.releaseChat = resolve
        })
      }
      try {
        return await json(route, answerResponse(body.client_request_id))
      } catch {
        return undefined
      }
    }

    if (path === '/api/v1/knowledge-items' && method === 'GET') {
      return json(route, {
        items: [state.knowledge],
        page: 1,
        page_size: 20,
        total: 1,
      })
    }
    if (path === '/api/v1/knowledge-items' && method === 'POST') {
      state.knowledge = knowledgeItem({
        ...(request.postDataJSON() as Record<string, unknown>),
        revision: 1,
      })
      return json(route, state.knowledge, 201)
    }
    if (path === `/api/v1/knowledge-items/${IDS.knowledge}`) {
      if (method === 'GET') return json(route, state.knowledge)
      if (method === 'DELETE') {
        state.knowledge = { ...state.knowledge, status: 'archived' }
        return route.fulfill({ status: 204 })
      }
      if (method === 'PATCH') {
        if (state.conflictNextKnowledgeUpdate) {
          state.conflictNextKnowledgeUpdate = false
          state.knowledge = knowledgeItem({
            ...state.knowledge,
            title: '服务器标题',
            revision: Number(state.knowledge.revision) + 1,
          })
          return json(
            route,
            errorBody(
              'KNOWLEDGE_REVISION_CONFLICT',
              '知识条目已被其他操作更新。',
            ),
            409,
          )
        }
        const body = request.postDataJSON() as Record<string, unknown>
        state.knowledge = knowledgeItem({
          ...state.knowledge,
          ...body,
          revision: Number(state.knowledge.revision) + 1,
        })
        return json(route, state.knowledge)
      }
    }

    if (path === '/api/v1/documents' && method === 'GET') {
      return json(route, {
        items: [documentItem()],
        page: 1,
        page_size: 20,
        total: 1,
      })
    }
    if (path === '/api/v1/documents' && method === 'POST') {
      return json(
        route,
        {
          document: documentItem(),
          job: job(IDS.pendingJob, state.pendingJobStatus),
        },
        202,
      )
    }
    if (path === '/api/v1/ingestion-jobs' && method === 'GET') {
      return json(route, {
        items: [
          job(IDS.pendingJob, state.pendingJobStatus),
          job(IDS.failedJob, state.failedJobStatus),
        ],
        total: 2,
        offset: 0,
        limit: 25,
      })
    }
    if (path.endsWith('/cancel') && method === 'POST') {
      state.pendingJobStatus = 'cancelled'
      return json(route, job(IDS.pendingJob, state.pendingJobStatus))
    }
    if (path.endsWith('/retry') && method === 'POST') {
      state.failedJobStatus = 'running'
      return json(route, job(IDS.failedJob, state.failedJobStatus), 202)
    }
    if (path === `/api/v1/ingestion-jobs/${IDS.pendingJob}`) {
      return json(route, job(IDS.pendingJob, state.pendingJobStatus))
    }
    if (path === `/api/v1/ingestion-jobs/${IDS.failedJob}`) {
      return json(route, job(IDS.failedJob, state.failedJobStatus))
    }

    return json(route, errorBody('HTTP_ERROR', 'Not found'), 404)
  })
  return state
}
