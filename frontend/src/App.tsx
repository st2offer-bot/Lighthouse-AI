import { useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent, ReactNode } from 'react'
import {
  Activity, ArrowRight, ArrowUpRight, Check, CheckCircle2,
  ChevronRight, CircleAlert, Clock3, CloudUpload, ExternalLink,
  Eye, FileVideo2, History, Info, LayoutDashboard, Link2, LockKeyhole,
  Menu, Play, Plus, Radar, ScanLine, Settings2, ShieldCheck,
  Sparkles, Timer, Upload, Video, X, Zap,
} from 'lucide-react'
import { formatTime, safeEvidenceSrc, uploadVideo, waitForJob } from './api'
import type { AnalysisJob, CompletedScan, NormativeRule, Violation } from './types'

type View = 'dashboard' | 'history' | 'about'
type Phase = 'idle' | 'uploading' | 'analyzing' | 'done' | 'failed'
const MAX_UI_FILE_SIZE = 250 * 1024 * 1024

function clampText(input: string, limit = 100): string {
  return input.length > limit ? `${input.slice(0, limit - 1)}…` : input
}

function fileSize(n: number): string {
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} КБ`
  return `${(n / (1024 * 1024)).toFixed(1)} МБ`
}

function normativeData(event: Violation): NormativeRule | null {
  const raw = event.rule ?? event.normative_source
  if (!raw) return null
  if (typeof raw === 'string') return { title: raw }
  if (!raw || typeof raw !== 'object') return null
  return raw
}

function formatDuration(sec: number): string {
  return sec < 60 ? `${Math.max(1, Math.round(sec))} сек` : `${Math.floor(sec / 60)} мин ${Math.round(sec % 60)} сек`
}

function IconTile({ children, accent = false }: { children: ReactNode; accent?: boolean }) {
  return <span className={`icon-tile ${accent ? 'accent' : ''}`}>{children}</span>
}

function SceneIllustration() {
  return (
    <div className="scene" aria-hidden="true">
      <div className="scene-top"><span><i className="dot live"/> CAM_01 / MONITORING</span><span>00:44:02 <i className="dot pale"/></span></div>
      <svg className="scene-art" viewBox="0 0 488 245" role="img" aria-label="Абстрактная визуализация видеомониторинга">
        <defs>
          <linearGradient id="deskGrad" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#67707a" stopOpacity=".45"/><stop offset="1" stopColor="#1e2937" stopOpacity=".26"/></linearGradient>
          <radialGradient id="sceneGlow"><stop stopColor="#a55158" stopOpacity=".23"/><stop offset="1" stopColor="#a55158" stopOpacity="0"/></radialGradient>
        </defs>
        <circle cx="278" cy="160" r="215" fill="url(#sceneGlow)"/>
        <g stroke="#627183" strokeOpacity=".17" strokeWidth="1">
          {Array.from({ length: 14 }).map((_, i) => <path key={`v${i}`} d={`M${i * 38} 0 V245`}/>)}
          {Array.from({ length: 8 }).map((_, i) => <path key={`h${i}`} d={`M0 ${i * 36} H488`}/>)}
        </g>
        <path d="M88 158 L239 115 L390 165 L263 221 Z" fill="url(#deskGrad)" stroke="#89949e" strokeOpacity=".32"/>
        <path d="M116 164 L235 136 L357 169 L256 204Z" fill="#1f2935" stroke="#7e8999" strokeOpacity=".2" />
        <rect x="215" y="140" width="62" height="34" rx="5" transform="rotate(6 215 140)" fill="#9b8a80" opacity=".8" />
        <path d="M224 148 L268 152 M223 157 L267 160 M223 164 L264 168" stroke="#d1b9a9" opacity=".72" strokeWidth="2"/>
        <circle cx="143" cy="62" r="24" fill="#556274" stroke="#8295ac" strokeOpacity=".34"/>
        <path d="M104 142 Q107 92 143 92 Q178 92 183 144 L172 171 L120 160 Z" fill="#47566a" stroke="#7e8999" strokeOpacity=".25" />
        <path d="M171 123 L221 145" stroke="#8395ab" strokeWidth="18" strokeLinecap="round"/><path d="M212 142 L234 145" stroke="#c4a68e" strokeWidth="15" strokeLinecap="round" />
        <circle cx="363" cy="65" r="24" fill="#4b5868" stroke="#8295ac" strokeOpacity=".34"/>
        <path d="M320 139 Q326 90 363 94 Q397 95 414 147 L391 160 L340 164 Z" fill="#394a5e" stroke="#7e8999" strokeOpacity=".25" />
        <path d="M331 124 L278 146" stroke="#768a9d" strokeWidth="17" strokeLinecap="round"/><path d="M286 144 L264 151" stroke="#c4a68e" strokeWidth="15" strokeLinecap="round" />
        <rect x="193" y="126" width="108" height="69" rx="8" fill="none" stroke="#fa8789" strokeDasharray="6 5" strokeWidth="2"/>
        <path d="M193 141v-15h17 M284 126h17v15 M193 180v15h17 M284 195h17v-15" stroke="#ff9a9a" strokeWidth="2.6" fill="none" />
        <circle cx="248" cy="157" r="24" stroke="#ffb3ae" fill="#e26267" fillOpacity=".09" strokeOpacity=".55" />
        <path d="M237 157h22 M248 146v22" stroke="#ffc5bf" strokeWidth="1.4"/>
        <rect x="303" y="115" width="107" height="23" rx="4" fill="#aa363f"/><text x="312" y="131" fontFamily="monospace" fontWeight="700" fontSize="10" fill="#fff">EVENT DETECTED</text>
        <path d="M302 127h-22" stroke="#e78a8b" strokeWidth="1.5"/>
        <path d="M19 22v15 M19 22h15 M469 22h-15 M469 22v15 M19 221v-15 M19 221h15 M469 221h-15 M469 221v-15" fill="none" stroke="#e4e8ef" strokeOpacity=".45" strokeWidth="1.5"/>
      </svg>
      <div className="scene-bottom"><span><i className="dot accent-dot"/> OBJECT TRACKING ACTIVE</span><span>DEMO VISUAL</span></div>
    </div>
  )
}

function Brand({ compact = false }: { compact?: boolean }) {
  return <div className={`brand ${compact ? 'compact' : ''}`}>
    <span className="brand-emblem"><ScanLine size={23} strokeWidth={2.5}/></span>
    <span className="brand-wordmark"><strong>FORTE<span>VISION</span></strong><small>AI VIDEO INTELLIGENCE</small></span>
  </div>
}

function App() {
  const [view, setView] = useState<View>('dashboard')
  const [mobileNav, setMobileNav] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [baseUrl, setBaseUrl] = useState(import.meta.env.VITE_API_BASE_URL || '')
  const [apiKey, setApiKey] = useState('')
  const [demo, setDemo] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [job, setJob] = useState<AnalysisJob | null>(null)
  const [phase, setPhase] = useState<Phase>('idle')
  const [error, setError] = useState('')
  const [elapsed, setElapsed] = useState(0)
  const [startedAt, setStartedAt] = useState(0)
  const [history, setHistory] = useState<CompletedScan[]>([])
  const [, setActiveScan] = useState<string | null>(null)
  const [lightbox, setLightbox] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const abortRef = useRef<AbortController | null>(null)
  const [polls, setPolls] = useState(0)

  useEffect(() => {
    if (!selectedFile) { setPreviewUrl(null); return }
    const url = URL.createObjectURL(selectedFile)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [selectedFile])

  useEffect(() => {
    if (phase !== 'uploading' && phase !== 'analyzing') return
    const interval = setInterval(() => setElapsed((Date.now() - startedAt) / 1000), 250)
    return () => clearInterval(interval)
  }, [phase, startedAt])

  useEffect(() => {
    if (!settingsOpen && !lightbox) return
    const close = (e: KeyboardEvent) => { if (e.key === 'Escape') { setSettingsOpen(false); setLightbox(null) } }
    window.addEventListener('keydown', close)
    return () => window.removeEventListener('keydown', close)
  }, [settingsOpen, lightbox])

  useEffect(() => () => abortRef.current?.abort(), [])

  const scansReal = history.filter(s => !s.demo)
  const numberViolations = scansReal.reduce((sum, item) => sum + (item.job.violations?.length || 0), 0)
  const averageTime = scansReal.length ? scansReal.reduce((sum, item) => sum + item.elapsedSec, 0) / scansReal.length : 0
  const processing = phase === 'uploading' || phase === 'analyzing'
  const displayedViolations = useMemo(() => job?.violations || [], [job])

  function selectFile(f: File | null) {
    if (!f) return
    if (!f.name.toLowerCase().endsWith('.mov')) { setError('Поддерживается формат .mov. Выберите подходящее видео.'); return }
    if (f.size > MAX_UI_FILE_SIZE) { setError('Максимальный размер в интерфейсе — 250 МБ. При необходимости измените лимит в исходном коде.'); return }
    if (f.size === 0) { setError('Файл пустой. Выберите другую видеозапись.'); return }
    abortRef.current?.abort()
    setSelectedFile(f)
    setActiveScan(null)
    setJob(null)
    setError('')
    setPhase('idle')
    setElapsed(0)
    setPolls(0)
    setView('dashboard')
  }

  function onDrop(e: DragEvent<HTMLElement>) {
    e.preventDefault()
    setDragging(false)
    selectFile(e.dataTransfer.files.item(0))
  }

  function onFileInput(e: ChangeEvent<HTMLInputElement>) {
    selectFile(e.target.files?.item(0) || null)
    e.target.value = ''
  }

  function addScan(newJob: AnalysisJob, isDemo: boolean, scanFile: File | null, spent: number) {
    const record: CompletedScan = {
      id: newJob.job_id,
      name: scanFile?.name || 'Пример интерфейса',
      timestamp: new Date(),
      elapsedSec: spent,
      size: scanFile?.size || 0,
      job: newJob,
      demo: isDemo,
      file: scanFile || undefined,
    }
    setHistory(old => [record, ...old].slice(0, 12))
    setActiveScan(record.id)
  }

  function showDemo() {
    abortRef.current?.abort()
    setDemo(true)
    setPhase('done')
    setError('')
    const sample: AnalysisJob = {
      job_id: `demo-${Date.now()}`,
      status: 'done',
      verdict: 'violation',
      violations: [{
        type: 'Демонстрационное событие',
        timecode_sec: 44.2,
        description: 'Пример оформления результатов. Событие и время созданы только для демонстрации интерфейса; нейросеть не анализировала видео.',
        evidence_frames: [],
      }],
    }
    setJob(sample)
    addScan(sample, true, null, 0)
    setView('dashboard')
    window.setTimeout(() => document.getElementById('results')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80)
  }

  async function analyze() {
    if (!selectedFile) { setError('Сначала выберите видеофайл в формате .mov.'); return }
    if (!baseUrl.trim() || !apiKey.trim()) { setSettingsOpen(true); setError('Для анализа укажите адрес Backend и API-ключ.'); return }
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    const start = Date.now()
    setStartedAt(start)
    setElapsed(0)
    setPolls(0)
    setError('')
    setJob(null)
    setDemo(false)
    setPhase('uploading')
    try {
      const id = await uploadVideo(baseUrl, apiKey, selectedFile, controller.signal)
      if (controller.signal.aborted) return
      setPhase('analyzing')
      const finished = await waitForJob(baseUrl, apiKey, id, controller.signal, () => setPolls(p => p + 1))
      if (controller.signal.aborted) return
      const seconds = (Date.now() - start) / 1000
      setElapsed(seconds)
      setJob(finished)
      setPhase('done')
      addScan(finished, false, selectedFile, seconds)
      window.setTimeout(() => document.getElementById('results')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80)
    } catch (err) {
      if (controller.signal.aborted) return
      setError(err instanceof Error ? err.message : 'Неизвестная ошибка обработки.')
      setPhase('failed')
    }
  }

  function cancelAnalysis() {
    abortRef.current?.abort()
    setPhase('idle')
    setError('Запрос остановлен в интерфейсе. Сервер может продолжать обработку уже загруженного видео.')
  }

  function openScan(scan: CompletedScan) {
    setJob(scan.job)
    setSelectedFile(scan.file || null)
    setDemo(scan.demo)
    setPhase('done')
    setActiveScan(scan.id)
    setError('')
    setView('dashboard')
    setMobileNav(false)
  }

  function goToTime(seconds: number) {
    setView('dashboard')
    const video = videoRef.current
    if (video && Number.isFinite(seconds) && seconds >= 0) {
      const seek = () => { video.currentTime = Math.min(seconds, Number.isFinite(video.duration) ? video.duration : seconds); video.play().catch(() => {}) }
      if (video.readyState >= 1) seek()
      else video.addEventListener('loadedmetadata', seek, { once: true })
      video.scrollIntoView({ behavior: 'smooth', block: 'center' })
    } else if (!demo) setError('Видео недоступно в браузере. Кадры-доказательства можно посмотреть ниже.')
  }

  const nav = (
    <>
      <div className="sidebar-block-label">РАБОЧЕЕ ПРОСТРАНСТВО</div>
      <button className={`nav-item ${view === 'dashboard' ? 'active' : ''}`} onClick={() => { setView('dashboard'); setMobileNav(false) }}><LayoutDashboard size={18}/> Анализ видео <ChevronRight className="nav-chevron" size={16}/></button>
      <button className={`nav-item ${view === 'history' ? 'active' : ''}`} onClick={() => { setView('history'); setMobileNav(false) }}><History size={18}/> История сессии <span className="nav-count">{history.length}</span></button>
      <button className={`nav-item ${view === 'about' ? 'active' : ''}`} onClick={() => { setView('about'); setMobileNav(false) }}><Info size={18}/> О системе</button>
      <div className="sidebar-block-label second-label">НАСТРОЙКИ</div>
      <button className="nav-item" onClick={() => { setSettingsOpen(true); setMobileNav(false) }}><Settings2 size={18}/> Подключение к API <ArrowUpRight className="nav-chevron" size={15}/></button>
    </>
  )

  return (
    <div className="app-shell">
      {mobileNav && <button className="mobile-overlay" onClick={() => setMobileNav(false)} aria-label="Закрыть меню"/>}
      <aside className={`sidebar ${mobileNav ? 'sidebar-open' : ''}`}>
        <div className="sidebar-brand"><Brand/><button className="mobile-close" onClick={() => setMobileNav(false)} aria-label="Закрыть меню"><X size={20}/></button></div>
        <div className="sidebar-inner">{nav}</div>
        <div className="sidebar-bottom">
          <div className="side-status"><span className="side-status-icon"><ShieldCheck size={18}/></span><span><strong>Приватность данных</strong><small>AI работает на вашем сервере</small></span></div>
          <div className="sidebar-footer"><div className="footer-orbit"><span/></div><div><strong>Datathon 26</strong><small>ForteBank AI Challenge</small></div><span className="v-label">v1.0</span></div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left"><button className="mobile-menu" aria-label="Открыть меню" onClick={() => setMobileNav(true)}><Menu size={22}/></button><span className="crumb">WORKSPACE</span><ChevronRight size={13} className="crumb-arrow"/><span className="crumb-current">{view === 'dashboard' ? 'Проверка видео' : view === 'history' ? 'История сессии' : 'О платформе'}</span></div>
          <div className="topbar-right"><span className="topbar-product"><span className="tiny-signal"/> Локальный AI · API-интеграция</span><button className="icon-button settings-shortcut" aria-label="Настройки" title="Настройки API" onClick={() => setSettingsOpen(true)}><Settings2 size={18}/></button><span className="vertical-divider"/><div className="team-avatar">F<span>V</span></div></div>
        </header>

        <div className="page-container">
          {view === 'dashboard' && <>
            <section className="welcome-strip"><div className="section-breadcrumb">ПЛАТФОРМА ВИДЕОАНАЛИТИКИ <span> / </span> КАССОВЫЕ ОПЕРАЦИИ</div><div className="welcome-heading"><div><h1>Центр видеоконтроля<span className="heading-point">.</span></h1><p>Умный анализ кассовых операций. Каждое событие — с точным временем и доказательствами.</p></div><button className="outlined-action" onClick={() => setSettingsOpen(true)}><Settings2 size={16}/> Настройки API</button></div></section>

            <section className="hero-card">
              <div className="hero-content"><div className="hero-eyebrow"><span className="hero-sparkle"><Sparkles size={14}/></span> INTELLIGENT VIDEO MONITORING <span className="eyebrow-bar"/></div><h2>Замечаем детали.<br/><em>Защищаем доверие.</em></h2><p>Локальный AI-анализ видеозаписей с фиксацией потенциальных нарушений, точными таймкодами и доказательствами для экспертной проверки.</p><div className="hero-actions"><button className="primary-button hero-button" onClick={() => { setView('dashboard'); inputRef.current?.click() }}><Plus size={18}/> Новая проверка <ArrowRight size={17}/></button><button className="quiet-link" onClick={showDemo}>Посмотреть демо <ArrowUpRight size={16}/></button></div></div>
              <div className="hero-visual"><span className="visual-circle circle-one"/><span className="visual-circle circle-two"/><SceneIllustration/><div className="visual-overlay-label"><Radar size={14}/> AI SCAN / PREVIEW</div></div>
            </section>

            <section className="stats-row" aria-label="Статистика текущей сессии">
              <div className="stat-card"><div className="stat-top"><span>Проверено видео</span><IconTile><Video size={19}/></IconTile></div><strong className="stat-value">{String(scansReal.length).padStart(2, '0')}</strong><div className="stat-foot"><span className="mini-sparkline">━━ ━━ ━━━</span> за эту сессию</div></div>
              <div className="stat-card"><div className="stat-top"><span>Найдено событий</span><IconTile accent><Radar size={19}/></IconTile></div><strong className="stat-value">{String(numberViolations).padStart(2, '0')}</strong><div className="stat-foot"><span className="mini-marker red"/> только реальные проверки</div></div>
              <div className="stat-card"><div className="stat-top"><span>Среднее время</span><IconTile><Timer size={19}/></IconTile></div><strong className="stat-value">{scansReal.length ? formatDuration(averageTime) : '—'}</strong><div className="stat-foot"><span className="mini-marker green"/> локальная обработка</div></div>
            </section>

            <section className="workflow-grid" id="upload">
              <div className="upload-panel panel"><div className="panel-header"><div><span className="overline">01 / ИСТОЧНИК ДАННЫХ</span><h3>Загрузить видеозапись</h3><p>Выберите файл для анализа кассовых операций</p></div><div className="panel-header-icon"><CloudUpload size={22}/></div></div>
                <input ref={inputRef} className="sr-only" type="file" accept=".mov,video/quicktime" onChange={onFileInput} aria-label="Выбрать видео MOV" />
                <div className={`drop-zone ${dragging ? 'drag-active' : ''} ${selectedFile ? 'file-chosen' : ''}`} role="button" tabIndex={0} onClick={() => !processing && inputRef.current?.click()} onKeyDown={e => { if ((e.key === 'Enter' || e.key === ' ') && !processing) { e.preventDefault(); inputRef.current?.click() } }} onDragOver={e => {e.preventDefault(); if (!processing) setDragging(true)}} onDragLeave={e => {e.preventDefault(); setDragging(false)}} onDrop={e => { if (!processing) onDrop(e); else e.preventDefault() }}>
                  {!selectedFile ? <><span className="upload-icon-wrap"><Upload size={24}/></span><strong>Перетащите видео сюда</strong><p>или <span className="red-underlined">выберите файл</span> на компьютере</p><span className="file-hint">MOV <i/> до 250 МБ в интерфейсе</span></> : <><span className="selected-file-icon"><FileVideo2 size={26}/></span><strong className="selected-name" title={selectedFile.name}>{selectedFile.name}</strong><p>{fileSize(selectedFile.size)} · MOV · Готов к анализу</p><span className="replace-file-link">Нажмите, чтобы заменить файл</span></>}
                </div>
                {error && <div className="error-alert" role="alert"><CircleAlert size={18}/><span>{error}</span><button aria-label="Закрыть ошибку" onClick={() => setError('')}><X size={16}/></button></div>}
                <div className="upload-bottom"><span><LockKeyhole size={14}/> Файл отправляется только на ваш Backend</span><button className="primary-button analyze-button" disabled={processing} onClick={analyze}>{processing ? <><span className="loader-white"/> Выполняется...</> : <><ScanLine size={17}/> Запустить анализ <ArrowRight size={16}/></>}</button></div>
              </div>

              <div className="process-panel panel"><div className="panel-header"><div><span className="overline">02 / ПРОЦЕСС</span><h3>Как работает анализ</h3><p>От загрузки до проверяемого результата</p></div><div className="panel-header-icon secondary"><Activity size={22}/></div></div>
                <div className="stepper"><div className="step"><div className="step-index">01</div><div><strong>Загрузка и проверка</strong><p>Безопасная передача видео по API</p></div><CheckCircle2 size={17} className="step-icon"/></div><div className="step-line"/><div className="step"><div className="step-index">02</div><div><strong>AI-анализ событий</strong><p>Локальная обработка на вашем сервере</p></div><Zap size={17} className="step-icon"/></div><div className="step-line"/><div className="step"><div className="step-index">03</div><div><strong>Отчёт с доказательствами</strong><p>Таймкоды, кадры и описание событий</p></div><Eye size={17} className="step-icon"/></div></div>
                <div className="process-callout"><ShieldCheck size={19}/><div><strong>Конфиденциальность — по умолчанию</strong><span>Нет обращений к облачным AI-сервисам из интерфейса.</span></div></div>
              </div>
            </section>

            {processing && <section className="processing-card panel" role="status"><div className="processing-circle"><Radar size={26}/></div><div className="processing-copy"><div className="status-heading"><strong>{phase === 'uploading' ? 'Загрузка видео на сервер' : 'AI анализирует запись'}</strong><span className="processing-badge"><span className="pulsing"/> В ПРОЦЕССЕ</span></div><p>{phase === 'uploading' ? 'Ожидаем подтверждения загрузки и идентификатор задачи.' : 'Сервер обрабатывает видео. Таймкоды появятся после завершения.'}</p><div className="processing-meta"><Clock3 size={14}/> Прошло: {formatTime(elapsed)} {polls > 0 && <span>· Проверок статуса: {polls}</span>}</div></div><button className="secondary-button" onClick={cancelAnalysis}>Остановить ожидание</button></section>}

            {job?.status === 'done' && <section className="results-section" id="results"><div className="results-title-row"><div><span className="overline">03 / РЕЗУЛЬТАТ</span><h3>Результаты проверки</h3><p>Данные получены {demo ? 'из демонстрационного примера' : 'из вашего Backend'}.</p></div>{demo && <span className="demo-banner-tag"><Info size={14}/> ДЕМО · НЕ РЕАЛЬНЫЙ АНАЛИЗ</span>}</div>
              <div className={`verdict-card ${job.verdict === 'violation' ? 'warning' : 'clean'}`}><div className="verdict-icon">{job.verdict === 'violation' ? <Radar size={25}/> : <CheckCircle2 size={25}/>}</div><div><span className="verdict-kicker">ЗАКЛЮЧЕНИЕ AI-СИСТЕМЫ</span><h4>{job.verdict === 'violation' ? 'Обнаружены события для проверки' : 'Подтверждённых нарушений нет — требуется проверка'}</h4><p>{demo ? 'Симуляция интерфейса: результат не основан на анализе видео.' : 'Результат автоматического анализа требует проверки уполномоченным специалистом.'}</p></div><div className="verdict-number"><strong>{displayedViolations.length}</strong><small>событий</small></div></div>

              <div className="results-body"><div className="events-area"><div className="results-subtitle"><h4>Обнаруженные события <span>{displayedViolations.length}</span></h4><small>Временные метки и пояснения</small></div>{displayedViolations.length === 0 ? <div className="empty-events"><CheckCircle2 size={30}/><strong>Автоматические события не обнаружены</strong><span>Модель не выявила событий-кандидатов. Это не подтверждает соблюдение правил.</span></div> : displayedViolations.map((event, idx) => { const rule = normativeData(event); return <div className="event-card" key={`${idx}-${event.timecode_sec}`}><div className="event-top"><div className="event-index">{String(idx+1).padStart(2,'0')}</div><div className="event-top-main"><strong>{event.type}</strong><span>{demo ? 'Демонстрационный пример' : 'Визуально наблюдаемое событие'}</span></div><button className="timecode-pill" onClick={() => goToTime(event.timecode_sec)} title="Перейти к моменту в видеоплеере"><Play size={13} fill="currentColor"/>{formatTime(event.timecode_sec)}</button></div><p className="event-description">{event.description}</p><div className="evidence-area"><div className="evidence-label"><Eye size={15}/> КАДРЫ-ДОКАЗАТЕЛЬСТВА</div><div className="evidence-frames">{event.evidence_frames?.length ? event.evidence_frames.slice(0, 3).map((f, i) => { const src = safeEvidenceSrc(f.image_base64); return src ? <button key={`${i}-${f.timestamp_sec}`} className="evidence-thumb" onClick={() => setLightbox(src)}><img src={src} alt={`Доказательный кадр ${i+1} в ${formatTime(f.timestamp_sec)}`}/><span>{formatTime(f.timestamp_sec)} <ArrowUpRight size={13}/></span></button> : <span className="evidence-missing" key={i}>Кадр недоступен</span> }) : <div className="evidence-placeholder"><Eye size={18}/><span>{demo ? 'В демо нет настоящих кадров' : 'Backend не передал кадры'}</span></div>}</div></div><div className="rule-area"><span><Link2 size={15}/> Нормативный источник</span>{rule ? <div className="rule-details"><strong>{rule.title || rule.document || 'Документ без названия'}</strong>{(rule.section || rule.clause) && <small>{rule.section || rule.clause}</small>}{rule.url && /^https:\/\//i.test(rule.url) && <a href={rule.url} target="_blank" rel="noopener noreferrer">Открыть источник <ExternalLink size={12}/></a>}</div> : <small>Нормативная ссылка не передана сервером — требуется подтверждение источника</small>}</div></div> })}</div>
                <div className="video-column"><div className="video-card panel"><div className="side-card-heading"><span><Video size={18}/> Исходная видеозапись</span><small>{selectedFile ? 'Локальный просмотр' : 'Недоступно'}</small></div>{previewUrl ? <video ref={videoRef} controls playsInline preload="metadata" src={previewUrl} className="video-player">Ваш браузер не поддерживает воспроизведение этого MOV-файла.</video> : <div className="no-video"><Play size={30}/><span>Видеофайл не выбран</span><small>Откройте новый анализ для просмотра</small></div>}<div className="video-footer"><span><FileVideo2 size={14}/>{selectedFile ? clampText(selectedFile.name, 28) : 'Нет файла'}</span><span>{selectedFile ? fileSize(selectedFile.size) : '—'}</span></div></div><div className="info-card"><Info size={17}/><span>AI показывает признаки возможных нарушений. Итоговое решение принимает специалист после проверки видеозаписи и нормативной базы.</span></div></div></div>
            </section>}

            <div className="workspace-footer"><span><span className="footer-pulse"/> FORTEVISION · AI VIDEO INTELLIGENCE</span><span>Designed for ForteBank Datathon 26 <span className="footer-dot">•</span> Локальная инфраструктура</span></div>
          </>}

          {view === 'history' && <section className="standalone-page"><div className="standalone-intro"><div className="overline">АКТИВНОСТЬ</div><h1>История проверок<span className="heading-point">.</span></h1><p>Результаты текущей сессии. После перезагрузки страницы история очищается.</p></div>{history.length ? <div className="history-list panel"><div className="history-table-head"><span>ФАЙЛ / ЗАДАЧА</span><span>РЕЗУЛЬТАТ</span><span>СОБЫТИЯ</span><span>ДЕЙСТВИЕ</span></div>{history.map(scan => <div className="history-row" key={scan.id}><div className="history-file"><span className="history-file-icon"><FileVideo2 size={19}/></span><div><strong>{clampText(scan.name, 28)}</strong><small>{scan.timestamp.toLocaleString('ru-RU')} · {scan.demo ? 'Демонстрация' : formatDuration(scan.elapsedSec)}</small></div></div><span className={`history-verdict ${scan.demo ? 'demo' : scan.job.verdict === 'violation' ? 'warning' : 'clean'}`}>{scan.demo ? 'Демо' : scan.job.verdict === 'violation' ? 'Есть признаки' : 'Без нарушений'}</span><strong className="history-number">{scan.job.violations?.length ?? 0}</strong><button className="text-action" onClick={() => openScan(scan)}>Открыть <ArrowRight size={15}/></button></div>)}</div> : <div className="empty-history panel"><History size={32}/><h3>Пока нет проверок</h3><p>Загрузите первое видео или посмотрите демонстрацию интерфейса.</p><button className="primary-button" onClick={() => setView('dashboard')}>Начать проверку <ArrowRight size={16}/></button></div>}</section>}

          {view === 'about' && <section className="standalone-page"><div className="standalone-intro"><span className="overline">О ПРОЕКТЕ</span><h1>AI, которому можно задать вопрос<span className="heading-point">.</span></h1><p>ForteVision — концепт пользовательского интерфейса для ForteBank Datathon 26. Он помогает сотруднику увидеть момент события, его описание и кадры для проверки.</p></div><div className="about-grid"><div className="about-card panel"><IconTile accent><ShieldCheck size={21}/></IconTile><h3>Обоснованные выводы</h3><p>Сервис показывает признаки возможных нарушений, не подменяя решение специалиста и не выдавая непроверенные нормативные ссылки.</p></div><div className="about-card panel"><IconTile><Radar size={21}/></IconTile><h3>Точный контекст</h3><p>Таймкоды, кадры-доказательства и быстрый переход к нужному участку видеозаписи.</p></div><div className="about-card panel"><IconTile><LockKeyhole size={21}/></IconTile><h3>Работа с локальным AI</h3><p>Frontend общается только с настроенным вами Backend. Видео не передаётся в сторонние AI API самим интерфейсом.</p></div></div><div className="api-contract panel"><h3>Контракт жюри</h3><p>Frontend настроен на спецификацию ForteBank:</p><code>POST /api/v1/analyze</code><code>GET /api/v1/jobs/{'{job_id}'}</code><span>Передача ключа: <strong>X-API-Key</strong>. Формат: <strong>MOV</strong>.</span></div></section>}
        </div>
      </main>

      {settingsOpen && <div className="modal-backdrop" onMouseDown={e => { if (e.currentTarget === e.target) setSettingsOpen(false) }}><div className="modal settings-modal" role="dialog" aria-modal="true" aria-labelledby="settings-title"><div className="modal-top"><span className="settings-icon"><Settings2 size={22}/></span><button className="modal-close" onClick={() => setSettingsOpen(false)} aria-label="Закрыть настройки"><X size={20}/></button></div><h3 id="settings-title">Подключение к Backend</h3><p>Укажите адрес сервера ваших коллег и API-ключ. Ключ хранится только в памяти страницы и сбрасывается после перезагрузки.</p><label className="field-label" htmlFor="backend-url">АДРЕС API СЕРВЕРА</label><div className="input-wrap"><Link2 size={18}/><input id="backend-url" placeholder="http://127.0.0.1:8000" autoComplete="url" value={baseUrl} onChange={e => setBaseUrl(e.target.value)}/></div><label className="field-label" htmlFor="backend-key">X-API-KEY</label><div className="input-wrap"><LockKeyhole size={18}/><input id="backend-key" type="password" placeholder="Секретный ключ от Backend" autoComplete="off" value={apiKey} onChange={e => setApiKey(e.target.value)}/></div><div className="modal-note"><ShieldCheck size={17}/><span>Для браузерной работы Backend должен разрешать CORS-источник вашего Frontend. Используйте HTTPS для публичного развёртывания.</span></div><button className="primary-button modal-save" onClick={() => { setBaseUrl(baseUrl.trim().replace(/\/+$/, '')); setSettingsOpen(false) }}><Check size={18}/> Сохранить настройки</button></div></div>}
      {lightbox && <div className="modal-backdrop lightbox" onMouseDown={e => {if (e.currentTarget === e.target) setLightbox(null)}} role="dialog" aria-modal="true" aria-label="Кадр-доказательство"><button className="lightbox-close" onClick={() => setLightbox(null)} aria-label="Закрыть кадр"><X size={23}/></button><img src={lightbox} alt="Увеличенный кадр-доказательство"/></div>}
    </div>
  )
}

export default App
