'use client'

import { useEffect, useRef, useState } from 'react'
import { Loader2, Send, Play, Pause, XCircle, Trophy, CheckCircle, Hash, TrendingUp } from 'lucide-react'
import { api } from '@/lib/api'
import type { Submission } from '@/lib/types'

type BatchProgress = {
  status?: string
  completed: number
  total: number
  running: number
  concurrency: number
}

type BatchResult = {
  run?: number
  submission_id?: string
  status?: string
  error?: string
}

export function CompetitionView() {
  const [submissions, setSubmissions] = useState<Submission[]>([])
  const [summary, setSummary] = useState<{ best_normalized_score: number; perfect_runs: number; total: number; check_pass_rate: number } | null>(null)
  const [loading, setLoading] = useState(true)

  const [endpointUrl, setEndpointUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [submitResult, setSubmitResult] = useState<string | null>(null)

  const [batchCount, setBatchCount] = useState(10)
  const [batchConcurrency, setBatchConcurrency] = useState(8)
  const [batchActive, setBatchActive] = useState(false)
  const [batchProgress, setBatchProgress] = useState<BatchProgress | null>(null)
  const [batchResults, setBatchResults] = useState<BatchResult[]>([])
  const [batchLoading, setBatchLoading] = useState(false)

  const batchPollRef = useRef<number | null>(null)
  const wasBatchActiveRef = useRef(false)

  useEffect(() => {
    fetchSubmissions()
  }, [])

  useEffect(() => {
    void refreshBatchStatus()
  }, [])

  useEffect(() => {
    if (!batchActive) {
      if (batchPollRef.current) {
        window.clearInterval(batchPollRef.current)
        batchPollRef.current = null
      }
      return
    }

    if (batchPollRef.current) {
      window.clearInterval(batchPollRef.current)
    }

    batchPollRef.current = window.setInterval(() => {
      void refreshBatchStatus()
    }, 3000)

    return () => {
      if (batchPollRef.current) {
        window.clearInterval(batchPollRef.current)
        batchPollRef.current = null
      }
    }
  }, [batchActive])

  async function fetchSubmissions() {
    setLoading(true)
    try {
      const data = await api.getSubmissions()
      setSubmissions(data.submissions ?? [])
      setSummary(data.summary ?? null)
    } catch (e) { console.error(e) }
    setLoading(false)
  }

  async function handleSubmit() {
    setSubmitting(true)
    setSubmitResult(null)
    try {
      const res = await api.submit(endpointUrl, apiKey)
      if (res.error) setSubmitResult(`Error: ${res.error}`)
      else setSubmitResult(`Submitted! ID: ${res.submission_id}`)
      fetchSubmissions()
    } catch (e) {
      setSubmitResult('Failed to submit')
    }
    setSubmitting(false)
  }

  async function handleStartBatch() {
    setBatchLoading(true)
    try {
      const count = Math.max(1, Math.min(batchCount, 100))
      const concurrency = Math.max(1, Math.min(batchConcurrency, 10))
      await api.startBatch(count, Math.min(concurrency, count))
      setBatchActive(true)
      setBatchResults([])
      setBatchProgress({
        completed: 0,
        total: count,
        running: 0,
        concurrency: Math.min(concurrency, count),
        status: `starting batch: ${count} submissions`,
      })
      void refreshBatchStatus()
    } catch (e) { console.error(e) }
    setBatchLoading(false)
  }

  async function handleStopBatch() {
    setBatchLoading(true)
    try {
      await api.stopBatch()
      await refreshBatchStatus()
    } catch (e) { console.error(e) }
    setBatchLoading(false)
  }

  async function refreshBatchStatus() {
    try {
      const status = await api.getBatchStatus()
      const active = Boolean(status?.active)

      setBatchActive(active)
      setBatchProgress(status?.progress ? {
        status: status.progress.status,
        completed: Number(status.progress.completed ?? 0),
        total: Number(status.progress.total ?? 0),
        running: Number(status.progress.running ?? 0),
        concurrency: Number(status.progress.concurrency ?? 0),
      } : null)
      setBatchResults(Array.isArray(status?.results) ? status.results : [])

      if (wasBatchActiveRef.current && !active) {
        void fetchSubmissions()
      }
      wasBatchActiveRef.current = active
    } catch (e) {
      console.error(e)
    }
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {summary && (
        <div className="grid grid-cols-4 gap-3">
          {[
            { icon: Trophy, label: 'Best Score', value: summary.best_normalized_score?.toFixed(2) ?? '-', color: 'text-[var(--yellow)]' },
            { icon: CheckCircle, label: 'Perfect Runs', value: String(summary.perfect_runs), color: 'text-[var(--green)]' },
            { icon: Hash, label: 'Total', value: String(summary.total), color: 'text-[var(--blue)]' },
            { icon: TrendingUp, label: 'Check Pass Rate', value: summary.check_pass_rate != null ? (summary.check_pass_rate * 100).toFixed(1) + '%' : '-', color: 'text-[var(--purple)]' },
          ].map(({ icon: Icon, label, value, color }) => (
            <div key={label} className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
              <div className="flex items-center gap-2 mb-1">
                <Icon className={`h-4 w-4 ${color}`} />
                <span className="text-xs text-[var(--text2)]">{label}</span>
              </div>
              <div className={`text-xl font-bold ${color}`}>{value}</div>
            </div>
          ))}
        </div>
      )}

      <div className="grid grid-cols-2 gap-4">
        <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4 space-y-3">
          <h3 className="text-sm font-semibold">Submit Endpoint</h3>
          <input
            type="text"
            placeholder="https://your-agent.run.app/solve"
            value={endpointUrl}
            onChange={(e) => setEndpointUrl(e.target.value)}
            className="w-full rounded border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-xs focus:border-[var(--blue)] focus:outline-none"
          />
          <input
            type="password"
            placeholder="API Key"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            className="w-full rounded border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-xs focus:border-[var(--blue)] focus:outline-none"
          />
          <button
            type="button"
            onClick={handleSubmit}
            disabled={submitting || !endpointUrl}
            className="flex items-center gap-2 rounded bg-[var(--blue)] px-4 py-2 text-xs font-medium text-white transition hover:opacity-90 disabled:opacity-50"
          >
            {submitting ? <Loader2 className="h-3 w-3 animate-spin" /> : <Send className="h-3 w-3" />}
            Submit
          </button>
          {submitResult && <p className="text-xs text-[var(--text2)]">{submitResult}</p>}
        </div>

        <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4 space-y-3">
          <h3 className="text-sm font-semibold">Batch Runner</h3>
          {batchActive && batchProgress ? (
            <div className="space-y-3">
              <div className="text-xs font-medium text-[var(--text)] flex items-center gap-2">
                <Loader2 className="h-3.5 w-3.5 animate-spin text-[var(--blue)]" />
                Batch Running: {batchProgress.completed}/{batchProgress.total} complete, {batchProgress.running} workers active
              </div>

              <div className="space-y-1.5 text-xs text-[var(--text2)]">
                <div className="h-2 rounded-full bg-[var(--surface2)] overflow-hidden">
                  <div
                    className="h-full bg-[var(--blue)] rounded-full transition-all"
                    style={{ width: `${batchProgress.total > 0 ? (batchProgress.completed / batchProgress.total) * 100 : 0}%` }}
                  />
                </div>
                <div>{batchProgress.completed}/{batchProgress.total}</div>
                <div>Workers: {batchProgress.running}/{batchProgress.concurrency} active</div>
                <div>Status: {batchProgress.status ?? 'running'}</div>
              </div>

              <div className="rounded border border-[var(--border)] bg-[var(--bg)]">
                <div className="px-2.5 py-2 text-[11px] text-[var(--text2)] border-b border-[var(--border)]">Results</div>
                <div className="max-h-36 overflow-y-auto p-2 text-xs space-y-1">
                  {batchResults.length === 0 ? (
                    <div className="text-[var(--text2)]">No completed submissions yet…</div>
                  ) : (
                    batchResults.map((result, idx) => {
                      const isOk = result.status === 'done'
                      const isErr = result.status === 'error' || result.status === 'timeout' || result.status === 'cancelled'
                      return (
                        <div key={`${result.run ?? idx}-${result.submission_id ?? 'none'}-${idx}`} className="font-mono text-[11px]">
                          #{result.run ?? idx + 1}:{' '}
                          {isOk ? '✓' : isErr ? '✗' : '…'}{' '}
                          {result.submission_id ? `submission ${result.submission_id}` : 'submission n/a'}
                          {result.error ? ` — error: ${result.error}` : ''}
                          {!result.error && result.status && result.status !== 'done' ? ` — ${result.status}` : ''}
                        </div>
                      )
                    })
                  )}
                </div>
              </div>

              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={handleStopBatch}
                  disabled={batchLoading}
                  className="flex items-center gap-1 rounded bg-amber-500 px-3 py-2 text-xs font-medium text-white disabled:opacity-50"
                >
                  <Pause className="h-3 w-3" /> Pause
                </button>
                <button
                  type="button"
                  onClick={handleStopBatch}
                  disabled={batchLoading}
                  className="flex items-center gap-1 rounded bg-[var(--red)] px-3 py-2 text-xs font-medium text-white disabled:opacity-50"
                >
                  <XCircle className="h-3 w-3" /> Cancel
                </button>
              </div>
            </div>
          ) : (
            <>
              <div className="flex gap-2">
                <div className="flex-1">
                  <label className="text-[11px] text-[var(--text2)]">Count</label>
                  <input type="number" value={batchCount} onChange={(e) => setBatchCount(Number(e.target.value))} min={1} max={100}
                    className="w-full rounded border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-xs focus:border-[var(--blue)] focus:outline-none" />
                </div>
                <div className="flex-1">
                  <label className="text-[11px] text-[var(--text2)]">Concurrency</label>
                  <input type="number" value={batchConcurrency} onChange={(e) => setBatchConcurrency(Number(e.target.value))} min={1} max={10}
                    className="w-full rounded border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-xs focus:border-[var(--blue)] focus:outline-none" />
                </div>
              </div>
              <button
                type="button"
                onClick={handleStartBatch}
                disabled={batchLoading}
                className="flex items-center gap-1 rounded bg-[var(--green)] px-3 py-2 text-xs font-medium text-white disabled:opacity-50"
              >
                {batchLoading ? <Loader2 className="h-3 w-3 animate-spin" /> : <Play className="h-3 w-3" />} Start Batch
              </button>
            </>
          )}
        </div>
      </div>

      <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
        <h3 className="text-sm font-semibold px-4 py-3 border-b border-[var(--border)]">Submissions ({submissions.length})</h3>
        {loading ? (
          <div className="flex justify-center py-8"><Loader2 className="h-5 w-5 animate-spin text-[var(--text2)]" /></div>
        ) : (
          <div className="overflow-x-auto max-h-96 overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-[var(--surface)]">
                <tr className="border-b border-[var(--border)] text-[var(--text2)]">
                  <th className="text-left px-3 py-2 font-medium">Time</th>
                  <th className="text-left px-3 py-2 font-medium">Score</th>
                  <th className="text-left px-3 py-2 font-medium">Normalized</th>
                  <th className="text-left px-3 py-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {submissions.map((sub) => (
                  <tr key={sub.id} className="border-b border-[var(--border)] last:border-0 hover:bg-[var(--surface2)]">
                    <td className="px-3 py-1.5 font-mono text-[var(--text2)]">{new Date(sub.created_at).toLocaleString()}</td>
                    <td className="px-3 py-1.5">{sub.score_raw?.toFixed(1) ?? '-'}/{sub.score_max?.toFixed(1) ?? '-'}</td>
                    <td className="px-3 py-1.5 font-semibold">{sub.normalized_score != null ? sub.normalized_score.toFixed(2) : '-'}</td>
                    <td className="px-3 py-1.5">
                      <span className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-medium ${
                        sub.status === 'completed' ? 'bg-green-500/20 text-green-300' :
                        sub.status === 'error' ? 'bg-red-500/20 text-red-300' :
                        'bg-zinc-500/20 text-zinc-400'
                      }`}>{sub.status}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
