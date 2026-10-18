import type { AnalysisJob } from './types'

export function normalizeBaseUrl(url: string): string {
  return url.trim() === '/' ? '' : url.trim().replace(/\/+$/, '')
}

export function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '00:00'
  const s = Math.floor(seconds)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = String(s % 60).padStart(2, '0')
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${sec}` : `${String(m).padStart(2, '0')}:${sec}`
}

export function safeEvidenceSrc(base64: string): string | null {
  const trimmed = base64.replace(/\s/g, '')
  if (!trimmed || !/^[A-Za-z0-9+/]*={0,2}$/.test(trimmed)) return null
  return `data:image/jpeg;base64,${trimmed}`
}

function messageFromStatus(status: number): string {
  if (status === 401 || status === 403) return 'Неверный API-ключ. Проверьте настройки подключения.'
  if (status === 413) return 'Файл слишком большой для сервера.'
  if (status === 415 || status === 422) return 'Сервер не принял формат видео.'
  if (status >= 500) return 'Ошибка на сервере. Проверьте Backend.'
  return `Ошибка запроса (${status}).`
}

async function checkedJson(res: Response): Promise<unknown> {
  if (!res.ok) {
    let detail = ''
    try {
      const json = await res.json() as { detail?: unknown }
      if (typeof json.detail === 'string' && json.detail.length < 300) detail = json.detail
    } catch { /* Server can send non-JSON errors. */ }
    throw new Error(detail || messageFromStatus(res.status))
  }
  try { return await res.json() } catch { throw new Error('Сервер вернул некорректный JSON.') }
}

export async function uploadVideo(baseUrl: string, apiKey: string, file: File, signal?: AbortSignal): Promise<string> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${normalizeBaseUrl(baseUrl)}/api/v1/analyze`, {
    method: 'POST',
    headers: { 'X-API-Key': apiKey },
    body: form,
    signal,
  })
  const data = await checkedJson(res) as { job_id?: unknown }
  if (typeof data.job_id !== 'string' || !data.job_id) throw new Error('Сервер не вернул job_id.')
  return data.job_id
}

export async function fetchJob(baseUrl: string, apiKey: string, jobId: string, signal?: AbortSignal): Promise<AnalysisJob> {
  const res = await fetch(`${normalizeBaseUrl(baseUrl)}/api/v1/jobs/${encodeURIComponent(jobId)}`, {
    headers: { 'X-API-Key': apiKey },
    signal,
  })
  const data = await checkedJson(res) as AnalysisJob
  if (!data || !['processing', 'done', 'failed'].includes(data.status)) throw new Error('Неизвестный формат статуса от Backend.')
  if (data.status === 'done' && (!['violation', 'no_violation', 'requires_review'].includes(data.verdict || '') || !Array.isArray(data.violations))) {
    throw new Error('Неверная структура результата: отсутствует verdict или violations.')
  }
  if (data.status === 'done') {
    const response = await fetch(
      `${normalizeBaseUrl(baseUrl)}/api/v1/jobs/${encodeURIComponent(jobId)}/report`,
      {
        headers: { 'X-API-Key': apiKey },
        signal,
      }
    )

    const report = await checkedJson(response) as {
      events?: Array<{
        event_type?: string
        start_sec?: number
        description?: string
        title?: string
      }>
    }

    const events = Array.isArray(report.events) ? report.events : []

    data.violations = events.map(event => ({
      type: event.title || event.event_type || 'Observation',
      timecode_sec: event.start_sec ?? 0,
      description: event.description || 'Требуется ручная проверка.',
      evidence_frames: [],
    }))

    data.verdict = 'requires_review'
  }

  return data
}

export async function waitForJob(baseUrl: string, apiKey: string, jobId: string, signal: AbortSignal, onPoll: () => void): Promise<AnalysisJob> {
  const started = Date.now()
  while (Date.now() - started < 290_000) {
    if (signal.aborted) throw new DOMException('Операция отменена', 'AbortError')
    const job = await fetchJob(baseUrl, apiKey, jobId, signal)
    onPoll()
    if (job.status === 'done') return job
    if (job.status === 'failed') throw new Error(job.error || 'AI-модель сообщила об ошибке обработки.')
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        signal.removeEventListener('abort', abort)
        resolve()
      }, 2200)
      const abort = () => {
        clearTimeout(timer)
        reject(new DOMException('Операция отменена', 'AbortError'))
      }
      signal.addEventListener('abort', abort, { once: true })
    })
  }
  throw new Error('Сервер не завершил анализ за отведённое время (5 минут).')
}
