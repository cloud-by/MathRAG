import { fireEvent, render, screen, waitFor } from '@testing-library/vue'
import { defineComponent } from 'vue'
import { createMemoryHistory, createRouter, RouterView } from 'vue-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ApiError } from '../../api/errors'
import { documentsApi } from './api'
import DocumentUpload from './DocumentUpload.vue'
import DocumentsPage from './DocumentsPage.vue'
import type { DocumentAccepted, DocumentPage } from './types'

const DOCUMENT_ID = '11111111-1111-4111-8111-111111111111'
const JOB_ID = '22222222-2222-4222-8222-222222222222'
const ACCEPTED: DocumentAccepted = {
  document: {
    id: DOCUMENT_ID,
    owner_id: '33333333-3333-4333-8333-333333333333',
    original_name: 'lesson.pdf',
    mime_type: 'application/pdf',
    size_bytes: 2048,
    sha256: 'a'.repeat(64),
    status: 'pending',
    created_at: '2026-07-31T08:00:00Z',
    updated_at: '2026-07-31T08:00:00Z',
  },
  job: {
    id: JOB_ID,
    requested_by: '33333333-3333-4333-8333-333333333333',
    document_id: DOCUMENT_ID,
    job_type: 'pdf',
    status: 'pending',
    progress: 0,
    attempt_count: 0,
    error_code: null,
    error_message: null,
    started_at: null,
    finished_at: null,
    created_at: '2026-07-31T08:00:00Z',
    updated_at: '2026-07-31T08:00:00Z',
  },
}

function documentPage(overrides: Partial<DocumentPage> = {}): DocumentPage {
  return {
    items: [ACCEPTED.document],
    page: 1,
    page_size: 20,
    total: 1,
    ...overrides,
  }
}

async function renderDocuments(path = '/documents') {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/documents', name: 'documents', component: DocumentsPage },
      {
        path: '/jobs',
        name: 'jobs',
        component: defineComponent({ template: '<main>任务页</main>' }),
      },
    ],
  })
  await router.push(path)
  await router.isReady()
  const Host = defineComponent({
    components: { RouterView },
    template: '<RouterView />',
  })
  render(Host, { global: { plugins: [router] } })
  return router
}

beforeEach(() => {
  vi.spyOn(documentsApi, 'list').mockResolvedValue(documentPage())
  vi.spyOn(documentsApi, 'upload').mockResolvedValue(ACCEPTED)
})

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('documents api', () => {
  it('uploads a real FormData body without forcing Content-Type', async () => {
    vi.mocked(documentsApi.upload).mockRestore()
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(ACCEPTED), {
        status: 202,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)
    const file = new File(['%PDF-1.4'], 'lesson.pdf', {
      type: 'application/pdf',
    })

    await documentsApi.upload(file, 'algebra')

    const [, options] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(options.body).toBeInstanceOf(FormData)
    expect((options.body as FormData).get('file')).toBe(file)
    expect((options.body as FormData).get('category')).toBe('algebra')
    expect(new Headers(options.headers).has('Content-Type')).toBe(false)
  })
})

describe('document upload queue', () => {
  it('shows per-file success and structured failure without losing either row', async () => {
    vi.mocked(documentsApi.upload)
      .mockResolvedValueOnce(ACCEPTED)
      .mockRejectedValueOnce(
        new ApiError({
          status: 413,
          code: 'UPLOAD_TOO_LARGE',
          message: '文件超过允许大小。',
          requestId: 'request-upload',
          details: null,
        }),
      )
    const router = createRouter({
      history: createMemoryHistory(),
      routes: [
        {
          path: '/jobs',
          name: 'jobs',
          component: defineComponent({ template: '<main>任务页</main>' }),
        },
      ],
    })
    await router.push('/jobs')
    await router.isReady()
    render(DocumentUpload, { global: { plugins: [router] } })
    const files = [
      new File(['%PDF-a'], 'lesson.pdf', { type: 'application/pdf' }),
      new File(['%PDF-b'], 'large.pdf', { type: 'application/pdf' }),
    ]

    await fireEvent.change(screen.getByLabelText('选择 PDF 文档'), {
      target: { files },
    })
    await fireEvent.click(screen.getByRole('button', { name: '开始上传' }))

    expect(await screen.findByText('上传成功')).toBeTruthy()
    expect(await screen.findByText('文件超过允许大小。')).toBeTruthy()
    expect(screen.getByText('lesson.pdf')).toBeTruthy()
    expect(screen.getByText('large.pdf')).toBeTruthy()
    expect(
      screen
        .getByRole('link', { name: '查看 lesson.pdf 的摄取任务' })
        .getAttribute('href'),
    ).toBe(`/jobs?document_id=${DOCUMENT_ID}`)
  })
})

describe('documents page', () => {
  it('uses only server-side status and pagination query parameters', async () => {
    vi.mocked(documentsApi.list).mockResolvedValue(
      documentPage({ page: 2, total: 22 }),
    )
    await renderDocuments('/documents?status=ready&page=2')

    expect(await screen.findByText('lesson.pdf')).toBeTruthy()
    expect(documentsApi.list).toHaveBeenCalledWith(
      { status: 'ready', page: 2, pageSize: 20 },
      expect.any(AbortSignal),
    )
  })

  it('refreshes the server collection after an upload succeeds', async () => {
    await renderDocuments()
    await screen.findByText('lesson.pdf')
    const file = new File(['%PDF-new'], 'new.pdf', {
      type: 'application/pdf',
    })
    await fireEvent.change(screen.getByLabelText('选择 PDF 文档'), {
      target: { files: [file] },
    })
    await fireEvent.click(screen.getByRole('button', { name: '开始上传' }))

    await waitFor(() => expect(documentsApi.list).toHaveBeenCalledTimes(2))
  })
})
