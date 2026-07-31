import { fireEvent, render, screen, waitFor } from '@testing-library/vue'
import { defineComponent, onMounted } from 'vue'
import { createMemoryHistory, createRouter, RouterView } from 'vue-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { jobsApi } from './api'
import JobsPage from './JobsPage.vue'
import type { IngestionJob, IngestionJobPage } from './types'
import { useJobPolling } from './useJobPolling'

const PENDING_ID = '11111111-1111-4111-8111-111111111111'
const FAILED_ID = '22222222-2222-4222-8222-222222222222'

function job(id: string, status: IngestionJob['status']): IngestionJob {
  return {
    id,
    requested_by: '33333333-3333-4333-8333-333333333333',
    document_id: '44444444-4444-4444-8444-444444444444',
    job_type: 'pdf',
    status,
    progress: status === 'completed' ? 100 : 20,
    attempt_count: 1,
    error_code: status === 'failed' ? 'PDF_PARSE_FAILED' : null,
    error_message: status === 'failed' ? '无法解析 PDF。' : null,
    started_at: status === 'pending' ? null : '2026-07-31T08:01:00Z',
    finished_at:
      status === 'failed' || status === 'completed'
        ? '2026-07-31T08:02:00Z'
        : null,
    created_at: '2026-07-31T08:00:00Z',
    updated_at: '2026-07-31T08:01:00Z',
  }
}

function jobPage(overrides: Partial<IngestionJobPage> = {}): IngestionJobPage {
  return {
    items: [job(PENDING_ID, 'pending'), job(FAILED_ID, 'failed')],
    total: 2,
    offset: 0,
    limit: 25,
    ...overrides,
  }
}

async function renderJobs(path = '/jobs') {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [{ path: '/jobs', name: 'jobs', component: JobsPage }],
  })
  await router.push(path)
  await router.isReady()
  const Host = defineComponent({
    components: { RouterView },
    template: '<RouterView />',
  })
  const result = render(Host, { global: { plugins: [router] } })
  return { router, ...result }
}

beforeEach(() => {
  vi.spyOn(jobsApi, 'list').mockResolvedValue(jobPage())
  vi.spyOn(jobsApi, 'get').mockResolvedValue(job(PENDING_ID, 'running'))
  vi.spyOn(jobsApi, 'cancel').mockResolvedValue(job(PENDING_ID, 'cancelled'))
  vi.spyOn(jobsApi, 'retry').mockResolvedValue(job(FAILED_ID, 'running'))
})

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('jobs page', () => {
  it('loads exact status, type, document, and offset filters', async () => {
    vi.mocked(jobsApi.list).mockResolvedValue(
      jobPage({ offset: 25, total: 51 }),
    )
    await renderJobs(
      '/jobs?status=failed&job_type=pdf&document_id=44444444-4444-4444-8444-444444444444&offset=25',
    )

    expect(await screen.findByText('PDF_PARSE_FAILED')).toBeTruthy()
    expect(jobsApi.list).toHaveBeenCalledWith(
      {
        status: 'failed',
        jobType: 'pdf',
        documentId: '44444444-4444-4444-8444-444444444444',
        offset: 25,
        limit: 25,
      },
      expect.any(AbortSignal),
    )
  })

  it('shows actions only in server-allowed states and refreshes after mutation', async () => {
    await renderJobs()
    await screen.findByText('PDF_PARSE_FAILED')

    expect(
      screen.getByRole('button', { name: `取消任务 ${PENDING_ID}` }),
    ).toBeTruthy()
    expect(
      screen.getByRole('button', { name: `重试任务 ${FAILED_ID}` }),
    ).toBeTruthy()
    expect(
      screen.queryByRole('button', { name: `重试任务 ${PENDING_ID}` }),
    ).toBeNull()
    expect(
      screen.queryByRole('button', { name: `取消任务 ${FAILED_ID}` }),
    ).toBeNull()

    await fireEvent.click(
      screen.getByRole('button', { name: `取消任务 ${PENDING_ID}` }),
    )
    await waitFor(() => expect(jobsApi.cancel).toHaveBeenCalledWith(PENDING_ID))
    await waitFor(() => expect(jobsApi.get).toHaveBeenCalledWith(PENDING_ID))
    await waitFor(() => expect(jobsApi.list).toHaveBeenCalledTimes(2))
  })
})

describe('job polling', () => {
  function setVisibility(value: DocumentVisibilityState): void {
    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      value,
    })
    document.dispatchEvent(new Event('visibilitychange'))
  }

  it('deduplicates timers and stops after a terminal response', async () => {
    vi.useFakeTimers()
    vi.mocked(jobsApi.get).mockResolvedValue(job(PENDING_ID, 'completed'))
    const updates: IngestionJob[] = []
    const Harness = defineComponent({
      setup() {
        const polling = useJobPolling({
          fetchJob: jobsApi.get,
          onUpdate: (value) => updates.push(value),
        })
        onMounted(() => {
          polling.sync([job(PENDING_ID, 'pending')])
          polling.sync([job(PENDING_ID, 'pending')])
        })
        return () => null
      },
    })
    render(Harness)

    await vi.advanceTimersByTimeAsync(2_000)
    expect(jobsApi.get).toHaveBeenCalledTimes(1)
    expect(updates.at(-1)?.status).toBe('completed')
    await vi.advanceTimersByTimeAsync(4_000)
    expect(jobsApi.get).toHaveBeenCalledTimes(1)
  })

  it('pauses while hidden, refreshes immediately when visible, and stops on unmount', async () => {
    vi.useFakeTimers()
    setVisibility('visible')
    vi.mocked(jobsApi.get).mockResolvedValue(job(PENDING_ID, 'running'))
    const Harness = defineComponent({
      setup() {
        const polling = useJobPolling({
          fetchJob: jobsApi.get,
          onUpdate: () => undefined,
        })
        onMounted(() => polling.sync([job(PENDING_ID, 'pending')]))
        return () => null
      },
    })
    const view = render(Harness)

    setVisibility('hidden')
    await vi.advanceTimersByTimeAsync(3_000)
    expect(jobsApi.get).not.toHaveBeenCalled()
    setVisibility('visible')
    await vi.waitFor(() => expect(jobsApi.get).toHaveBeenCalledTimes(1))

    view.unmount()
    await vi.advanceTimersByTimeAsync(4_000)
    expect(jobsApi.get).toHaveBeenCalledTimes(1)
  })

  it('keeps polling after an error without creating duplicate timers', async () => {
    vi.useFakeTimers()
    setVisibility('visible')
    vi.mocked(jobsApi.get)
      .mockRejectedValueOnce(new Error('temporary'))
      .mockResolvedValueOnce(job(PENDING_ID, 'completed'))
    const Harness = defineComponent({
      setup() {
        const polling = useJobPolling({
          fetchJob: jobsApi.get,
          onUpdate: () => undefined,
        })
        onMounted(() => polling.sync([job(PENDING_ID, 'running')]))
        return () => null
      },
    })
    render(Harness)

    await vi.advanceTimersByTimeAsync(2_000)
    expect(jobsApi.get).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(2_000)
    expect(jobsApi.get).toHaveBeenCalledTimes(2)
  })
})
