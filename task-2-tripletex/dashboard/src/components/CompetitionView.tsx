'use client'

import { useEffect, useState } from 'react'
import { Loader2, Send, Play, Square, Trophy, CheckCircle, Hash, TrendingUp } from 'lucide-react'
import { api } from '@/lib/api'
import type { Submission } from '@/lib/types'

export function CompetitionView() {
  const [submissions, setSubmissions] = useState<Submission[]>([])
  const [summary, setSummary] = useState<{ best_normalized_score: number; perfect_runs: number; total: number; check_pass_rate: number } | null>(null)
  const [loading, setLoading] = useState(true)

  const [endpointUrl, setEndpointUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [submitResult, setSubmitResult] = useState<string | null>(null)

  const [batchCount, setBatchCount] = useState(10)
  const [batchDelay, setBatchDelay] = useState(2)
  const [batchActive, setBatchActive] = useState(false)
  const [batchProgress, setBatchProgress] = useState<{ completed: number; total: number } | null>(null)

  useEffect(() => {
    fetchSubmissions()
  }, [])

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
    try {
      await api.startBatch(batchCount, batchDelay)
      setBatchActive(true)
      pollBatch()
    } catch (e) { console.error(e) }
  }

  async function handleStopBatch() {
    try {
      await api.stopBatch()
      setBatchActive(false)
      setBatchProgress(null)
    } catch (e) { console.error(e) }
  }

  async function pollBatch() {
    const interval = setInterval(async () => {
      try {
        const status = await api.getBatchStatus()
        setBatchActive(status.active)
        setBatchProgress(status.progress ? { completed: status.progress.completed, total: status.progress.total } : null)
        if (!status.active) {
          clearInterval(interval)
          fetchSubmissions()
        }
      } catch { clearInterval(interval) }
    }, 3000)
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
          <div className="flex gap-2">
            <div className="flex-1">
              <label className="text-[11px] text-[var(--text2)]">Count</label>
              <input type="number" value={batchCount} onChange={(e) => setBatchCount(Number(e.target.value))} min={1} max={100}
                className="w-full rounded border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-xs focus:border-[var(--blue)] focus:outline-none" />
            </div>
            <div className="flex-1">
              <label className="text-[11px] text-[var(--text2)]">Delay (s)</label>
              <input type="number" value={batchDelay} onChange={(e) => setBatchDelay(Number(e.target.value))} min={0} max={60}
                className="w-full rounded border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-xs focus:border-[var(--blue)] focus:outline-none" />
            </div>
          </div>
          <div className="flex gap-2">
            <button type="button" onClick={handleStartBatch} disabled={batchActive}
              className="flex items-center gap-1 rounded bg-[var(--green)] px-3 py-2 text-xs font-medium text-white disabled:opacity-50">
              <Play className="h-3 w-3" /> Start
            </button>
            <button type="button" onClick={handleStopBatch} disabled={!batchActive}
              className="flex items-center gap-1 rounded bg-[var(--red)] px-3 py-2 text-xs font-medium text-white disabled:opacity-50">
              <Square className="h-3 w-3" /> Stop
            </button>
          </div>
          {batchProgress && (
            <div className="text-xs text-[var(--text2)]">
              Progress: {batchProgress.completed}/{batchProgress.total}
              <div className="h-1.5 rounded-full bg-[var(--surface2)] mt-1 overflow-hidden">
                <div className="h-full bg-[var(--blue)] rounded-full transition-all" style={{ width: `${(batchProgress.completed / batchProgress.total) * 100}%` }} />
              </div>
            </div>
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
