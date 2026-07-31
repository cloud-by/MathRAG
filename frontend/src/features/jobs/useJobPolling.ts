import { onBeforeUnmount, onMounted } from 'vue'

import type { IngestionJob } from './types'

export const ACTIVE_JOB_STATUSES = new Set<IngestionJob['status']>([
  'pending',
  'running',
])
export const POLL_INTERVAL_MS = 2_000

interface JobPollingOptions {
  fetchJob: (id: string) => Promise<IngestionJob>
  onUpdate: (job: IngestionJob) => void
  intervalMs?: number
}

export function useJobPolling(options: JobPollingOptions) {
  const intervalMs = options.intervalMs ?? POLL_INTERVAL_MS
  const activeIds = new Set<string>()
  const timers = new Map<string, ReturnType<typeof setTimeout>>()
  const inFlight = new Set<string>()
  let stopped = false

  function isVisible(): boolean {
    return document.visibilityState !== 'hidden'
  }

  function clearTimer(id: string): void {
    const timer = timers.get(id)
    if (timer !== undefined) clearTimeout(timer)
    timers.delete(id)
  }

  function schedule(id: string): void {
    if (
      stopped ||
      !isVisible() ||
      !activeIds.has(id) ||
      timers.has(id) ||
      inFlight.has(id)
    ) {
      return
    }
    timers.set(
      id,
      setTimeout(() => {
        timers.delete(id)
        void poll(id)
      }, intervalMs),
    )
  }

  async function poll(id: string): Promise<void> {
    if (stopped || !isVisible() || !activeIds.has(id) || inFlight.has(id)) {
      return
    }
    inFlight.add(id)
    try {
      const updated = await options.fetchJob(id)
      options.onUpdate(updated)
      if (!ACTIVE_JOB_STATUSES.has(updated.status)) activeIds.delete(id)
    } catch {
      // 临时轮询失败时保留最后一次成功数据，并沿用同一调度链。
    } finally {
      inFlight.delete(id)
      schedule(id)
    }
  }

  function sync(jobs: IngestionJob[]): void {
    const next = new Set(
      jobs
        .filter((job) => ACTIVE_JOB_STATUSES.has(job.status))
        .map((job) => job.id),
    )
    for (const id of activeIds) {
      if (!next.has(id)) clearTimer(id)
    }
    activeIds.clear()
    for (const id of next) {
      activeIds.add(id)
      schedule(id)
    }
  }

  function refreshActive(): void {
    if (!isVisible()) return
    for (const id of activeIds) {
      clearTimer(id)
      void poll(id)
    }
  }

  function handleVisibility(): void {
    if (!isVisible()) {
      for (const id of timers.keys()) clearTimer(id)
      return
    }
    refreshActive()
  }

  function stop(): void {
    stopped = true
    for (const id of timers.keys()) clearTimer(id)
    activeIds.clear()
  }

  onMounted(() =>
    document.addEventListener('visibilitychange', handleVisibility),
  )
  onBeforeUnmount(() => {
    document.removeEventListener('visibilitychange', handleVisibility)
    stop()
  })

  return { sync, refreshActive, stop }
}
