'use client'

import { useEffect, useState, useRef, useCallback } from 'react'
import { Loader2, Copy, Check, Trash2 } from 'lucide-react'
import { api } from '@/lib/api'
import { useStore } from '@/lib/store'
import { formatDuration, cn } from '@/lib/utils'
import { ScoreCard } from './ScoreCard'
import { PostMortem } from './PostMortem'
import { Timeline } from './Timeline'
import type { RunSummary, TraceEvent, ApiCallLog } from '@/lib/types'

type Tab = 'overview' | 'timeline' | 'calls' | 'enforcer'

interface RunDetailProps {
  runId: string
}

function deriveApiCalls(events: TraceEvent[]): ApiCallLog[] {
  const calls: ApiCallLog[] = []
  const starts: Map<number, TraceEvent> = new Map()

  for (let i = 0; i < events.length; i++) {
    const e = events[i]
    if (e.event_type === 'tool_start' && e.payload?.tool_name === 'tripletex_request') {
      const args = e.payload.arguments ?? {}
      calls.push({
        method: args.method ?? '?',
        path: args.path ?? '?',
        status_code: 0,
        ok: true,
      })
      starts.set(calls.length - 1, e)
    }
    if (e.event_type === 'tool_result' && e.payload?.tool_name === 'tripletex_request') {
      const result = e.payload.result ?? {}
      if (calls.length > 0) {
        const lastUnresolved = calls.findIndex((c, idx) => c.status_code === 0 && idx >= 0)
        if (lastUnresolved >= 0) {
          calls[lastUnresolved].status_code = result.status_code ?? (result.ok ? 200 : 400)
          calls[lastUnresolved].ok = result.ok ?? true
        }
      }
    }
  }
  return calls
}

export function RunDetail({ runId }: RunDetailProps) {
  const [summary, setSummary] = useState<RunSummary | null>(null)
  const [events, setEvents] = useState<TraceEvent[]>([])
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState<Tab>('overview')
  const [copied, setCopied] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const pollRef = useRef<NodeJS.Timeout | null>(null)
  const runs = useStore(s => s.runs)
  const removeRun = useStore(s => s.removeRun)

  const handleCopyFilename = useCallback(() => {
    if (!summary?.filename) return
    navigator.clipboard.writeText(summary.filename).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }, [summary?.filename])

  const handleDelete = useCallback(async () => {
    setDeleting(true)
    try {
      await api.deleteRun(runId)
      removeRun(runId)
    } catch (e) { console.error(e) }
    setDeleting(false)
  }, [runId, removeRun])

  const fetchDetail = (showLoader: boolean) => {
    if (showLoader) setLoading(true)
    api.getRunDetail(runId).then(data => {
      if (data?.summary) {
        setSummary(data.summary)
        setEvents(data.events ?? [])
      }
      if (showLoader) setLoading(false)
    }).catch(() => {
      if (showLoader) setLoading(false)
    })
  }

  useEffect(() => {
    fetchDetail(true)
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [runId])

  // Auto-refresh for running runs
  useEffect(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }

    if (summary?.status === 'running') {
      pollRef.current = setInterval(() => fetchDetail(false), 3000)
    }

    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [summary?.status, runId])

  // Re-fetch when WebSocket snapshot updates this run's status
  const storeRun = runs.find(r => r.run_id === runId)
  const storeStatus = storeRun?.status
  const storeScore = storeRun?.competition_score
  useEffect(() => {
    if (storeStatus && summary && storeStatus !== summary.status) {
      fetchDetail(false)
    }
    if (storeScore && summary && !summary.competition_score) {
      fetchDetail(false)
    }
  }, [storeStatus, storeScore])

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-[var(--text2)]" />
      </div>
    )
  }

  if (!summary) {
    return <div className="flex h-full items-center justify-center text-sm text-[var(--text2)]">Run not found</div>
  }

  const s = summary

  // Derive API calls from events if summary call_log is empty
  const apiCalls = s.tripletex_call_log.length > 0 ? s.tripletex_call_log : deriveApiCalls(events)
  const apiCallCount = apiCalls.length || s.tripletex_call_count

  const isRecent = s.started_at ? (Date.now() - new Date(s.started_at).getTime()) < 10 * 60 * 1000 : false
  const isAwaitingScore = s.source === 'competition' && s.status === 'completed' && !s.competition_score && isRecent

  const tabs: { key: Tab; label: string }[] = [
    { key: 'overview', label: 'Overview' },
    { key: 'timeline', label: `Timeline (${events.length})` },
    { key: 'calls', label: `API Calls (${apiCallCount})` },
    { key: 'enforcer', label: `Enforcer (${s.enforcer_rejections.length + s.semantic_rejections.length})` },
  ]

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="shrink-0 border-b border-[var(--border)] bg-[var(--surface)] px-4 py-3">
        <div className="flex items-center gap-2 mb-1">
          <span className={cn('inline-block h-2.5 w-2.5 rounded-full',
            s.status === 'completed' ? 'bg-[var(--green)]' :
            s.status === 'error' ? 'bg-[var(--red)]' :
            s.status === 'running' ? 'bg-[var(--blue)] animate-pulse' : 'bg-zinc-500'
          )} />
          <span className="text-sm font-semibold">{s.task_type || 'unknown'}</span>
          <span className="text-xs text-[var(--text2)]">{s.run_id.slice(0, 12)}</span>
          {isAwaitingScore && (
            <span className="flex items-center gap-1 text-[11px] text-[var(--yellow)]">
              <Loader2 className="h-3 w-3 animate-spin" /> awaiting score
            </span>
          )}
          <span className="ml-auto flex items-center gap-2">
            <span className="text-xs text-[var(--text2)]">{formatDuration(s.duration_seconds)}</span>
            <button type="button" onClick={handleDelete} disabled={deleting}
              className="p-1 rounded text-[var(--text2)] hover:text-[var(--red)] transition" title="Delete run">
              {deleting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
            </button>
          </span>
        </div>
        <p className="text-xs text-[var(--text2)] leading-relaxed line-clamp-2">{s.prompt || s.goal}</p>

        {s.filename && (
          <div className="flex items-center gap-1.5 mt-1.5">
            <span className="text-[10px] font-mono text-[var(--text2)] truncate">{s.filename}</span>
            <button type="button" onClick={handleCopyFilename}
              className="p-0.5 rounded text-[var(--text2)] hover:text-[var(--text)] transition shrink-0" title="Copy filename">
              {copied ? <Check className="h-3 w-3 text-[var(--green)]" /> : <Copy className="h-3 w-3" />}
            </button>
          </div>
        )}

        <div className="flex gap-1 mt-3">
          {tabs.map(({ key, label }) => (
            <button
              key={key}
              type="button"
              onClick={() => setTab(key)}
              className={cn(
                'rounded px-2.5 py-1 text-[11px] font-medium transition',
                tab === key ? 'bg-[var(--blue)] text-white' : 'text-[var(--text2)] hover:text-[var(--text)] hover:bg-[var(--surface2)]',
              )}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {tab === 'overview' && (
          <>
            {s.competition_score && s.competition_score.score_raw != null && s.competition_score.status === 'completed' && (
              <ScoreCard score={s.competition_score} />
            )}

            {s.source === 'competition' && s.status !== 'running' && (
              <PostMortem events={events} runId={runId} />
            )}

            <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
              <h3 className="text-xs font-semibold uppercase text-[var(--text2)] mb-3">Run Details</h3>
              <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-xs">
                <dt className="text-[var(--text2)]">Status</dt><dd>{s.status}</dd>
                <dt className="text-[var(--text2)]">Source</dt><dd>{s.source}</dd>
                <dt className="text-[var(--text2)]">Task Type</dt><dd>{s.task_type || '-'}</dd>
                <dt className="text-[var(--text2)]">Duration</dt><dd>{formatDuration(s.duration_seconds)}</dd>
                <dt className="text-[var(--text2)]">API Calls</dt><dd>{apiCallCount}</dd>
                <dt className="text-[var(--text2)]">API Errors</dt><dd className={s.tripletex_error_count > 0 ? 'text-[var(--red)]' : ''}>{s.tripletex_error_count}</dd>
                <dt className="text-[var(--text2)]">Tool Calls</dt><dd>{s.tool_call_count}</dd>
                <dt className="text-[var(--text2)]">Events</dt><dd>{s.event_count}</dd>
                <dt className="text-[var(--text2)]">Files</dt><dd>{s.file_count}</dd>
              </dl>
            </div>

            {s.goal && (
              <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
                <h3 className="text-xs font-semibold uppercase text-[var(--text2)] mb-2">Goal</h3>
                <p className="text-xs leading-relaxed">{s.goal}</p>
              </div>
            )}

            {s.error_message && (
              <div className="rounded-lg border border-red-500/30 bg-red-500/5 p-4">
                <h3 className="text-xs font-semibold text-[var(--red)] mb-1">Error</h3>
                <p className="text-xs text-[var(--text2)] font-mono">{s.error_message}</p>
              </div>
            )}

            {s.summary && (
              <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
                <h3 className="text-xs font-semibold uppercase text-[var(--text2)] mb-2">Summary</h3>
                <p className="text-xs leading-relaxed">{s.summary}</p>
              </div>
            )}

            {Object.keys(s.planner_payload).length > 0 && (
              <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
                <h3 className="text-xs font-semibold uppercase text-[var(--text2)] mb-2">Planner</h3>
                <pre className="text-[11px] text-[var(--text2)] font-mono whitespace-pre-wrap break-all max-h-48 overflow-y-auto">
                  {JSON.stringify(s.planner_payload, null, 2)}
                </pre>
              </div>
            )}
          </>
        )}

        {tab === 'timeline' && <Timeline events={events} />}

        {tab === 'calls' && (
          <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[var(--border)] text-[var(--text2)]">
                  <th className="text-left px-3 py-2 font-medium">Method</th>
                  <th className="text-left px-3 py-2 font-medium">Path</th>
                  <th className="text-left px-3 py-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {apiCalls.map((call, i) => (
                  <tr key={i} className="border-b border-[var(--border)] last:border-0 hover:bg-[var(--surface2)]">
                    <td className={cn('px-3 py-1.5 font-mono font-medium',
                      call.method === 'GET' ? 'text-[var(--green)]' :
                      call.method === 'POST' ? 'text-[var(--blue)]' :
                      call.method === 'PUT' ? 'text-[var(--yellow)]' :
                      call.method === 'DELETE' ? 'text-[var(--red)]' : ''
                    )}>{call.method}</td>
                    <td className="px-3 py-1.5 font-mono text-[var(--text2)] truncate max-w-xs">{call.path}</td>
                    <td className={cn('px-3 py-1.5 font-mono',
                      call.status_code === 0 ? 'text-[var(--text2)]' :
                      call.status_code < 300 ? 'text-[var(--green)]' :
                      call.status_code < 400 ? 'text-[var(--yellow)]' : 'text-[var(--red)]'
                    )}>
                      {call.status_code === 0 ? '-' : call.status_code}
                    </td>
                  </tr>
                ))}
                {apiCalls.length === 0 && (
                  <tr><td colSpan={3} className="px-3 py-4 text-center text-[var(--text2)]">No API calls recorded</td></tr>
                )}
              </tbody>
            </table>
          </div>
        )}

        {tab === 'enforcer' && (
          <div className="space-y-3">
            {s.enforcer_rejections.length === 0 && s.semantic_rejections.length === 0 && s.enforcer_overrides.length === 0 && (
              <div className="text-center text-xs text-[var(--text2)] py-6">No enforcer activity</div>
            )}
            {s.enforcer_rejections.map((e, i) => (
              <div key={`rej-${i}`} className="rounded border border-orange-500/30 bg-orange-500/5 p-3">
                <div className="text-[11px] font-semibold text-orange-400 mb-1">Enforcer Rejected</div>
                <div className="text-xs text-[var(--text2)]">{e.reason}</div>
                {e.suggestion && <div className="text-xs text-[var(--text2)] mt-1 italic">{e.suggestion}</div>}
              </div>
            ))}
            {s.semantic_rejections.map((e, i) => (
              <div key={`sem-${i}`} className="rounded border border-amber-500/30 bg-amber-500/5 p-3">
                <div className="text-[11px] font-semibold text-amber-400 mb-1">Semantic Rejection</div>
                <div className="text-xs text-[var(--text2)]">{e.reason}</div>
              </div>
            ))}
            {s.enforcer_overrides.map((e, i) => (
              <div key={`ovr-${i}`} className="rounded border border-yellow-500/30 bg-yellow-500/5 p-3">
                <div className="text-[11px] font-semibold text-yellow-400 mb-1">Override</div>
                <div className="text-xs text-[var(--text2)]">{e.reason}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
