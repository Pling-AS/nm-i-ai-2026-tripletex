'use client'

import { useState } from 'react'
import { Loader2, Search, AlertTriangle, CheckCircle, Lightbulb } from 'lucide-react'
import { api } from '@/lib/api'
import type { TraceEvent, PostMortemPayload } from '@/lib/types'

interface PostMortemProps {
  events: TraceEvent[]
  runId: string
  hasScore?: boolean
}

export function PostMortem({ events, runId, hasScore }: PostMortemProps) {
  const [loading, setLoading] = useState(false)
  const [localPm, setLocalPm] = useState<PostMortemPayload | null>(null)

  const pmEvent = events.find(e => e.event_type === 'post_mortem')
  const pm: PostMortemPayload | null = localPm ?? (pmEvent ? pmEvent.payload as PostMortemPayload : null)
  const awaitingScore = !hasScore && !pm

  async function handleGenerate() {
    setLoading(true)
    try {
      const res = await api.generatePostMortem(runId)
      if (res.post_mortem) setLocalPm(res.post_mortem)
    } catch (e) {
      console.error('Post-mortem generation failed', e)
    } finally {
      setLoading(false)
    }
  }

  if (!pm) {
    return (
      <div className="rounded-lg border border-dashed border-[var(--border)] p-6 text-center">
        {awaitingScore ? (
          <>
            <Loader2 className="h-8 w-8 mx-auto mb-2 text-[var(--yellow)] animate-spin" />
            <p className="text-sm text-[var(--yellow)] mb-1">Waiting for score enrichment</p>
            <p className="text-xs text-[var(--text2)]">Post-mortem analysis will be available after scoring completes</p>
          </>
        ) : (
          <>
            <Search className="h-8 w-8 mx-auto mb-2 text-[var(--text2)]" />
            <p className="text-sm text-[var(--text2)] mb-3">No post-mortem analysis yet</p>
            <button
              type="button"
              onClick={handleGenerate}
              disabled={loading}
              className="rounded-md bg-[var(--purple)] px-4 py-2 text-sm font-medium text-white transition hover:opacity-90 disabled:opacity-50"
            >
              {loading ? <Loader2 className="h-4 w-4 animate-spin inline mr-1" /> : null}
              {loading ? 'Analyzing...' : 'Generate Analysis'}
            </button>
          </>
        )}
      </div>
    )
  }

  const confidenceColor = pm.confidence === 'high' ? 'text-[var(--green)]' : pm.confidence === 'medium' ? 'text-[var(--yellow)]' : 'text-[var(--text2)]'

  return (
    <div className="rounded-lg border border-purple-500/25 bg-gradient-to-br from-purple-500/5 to-indigo-500/5 p-4 space-y-3">
      <div className="flex items-center gap-2">
        <Search className="h-4 w-4 text-[var(--purple)]" />
        <h3 className="text-sm font-semibold text-[var(--purple)]">Post-Mortem Analysis</h3>
        <span className={`ml-auto text-[10px] font-medium uppercase ${confidenceColor}`}>{pm.confidence}</span>
        <span className="text-[10px] rounded bg-[var(--surface2)] px-1.5 py-0.5 text-[var(--text2)]">{pm.root_cause_category}</span>
      </div>

      <p className="text-sm font-medium">{pm.headline}</p>

      {pm.what_went_wrong?.length > 0 && (
        <div>
          <div className="flex items-center gap-1 text-[11px] font-semibold text-[var(--red)] uppercase mb-1">
            <AlertTriangle className="h-3 w-3" /> Issues
          </div>
          {pm.what_went_wrong.map((item, i) => (
            <div key={i} className="text-xs text-[var(--text2)] py-1 pl-3 border-l-2 border-[var(--red)]">{item}</div>
          ))}
        </div>
      )}

      {pm.what_worked?.length > 0 && (
        <div>
          <div className="flex items-center gap-1 text-[11px] font-semibold text-[var(--green)] uppercase mb-1">
            <CheckCircle className="h-3 w-3" /> Worked
          </div>
          {pm.what_worked.map((item, i) => (
            <div key={i} className="text-xs text-[var(--text2)] py-1 pl-3 border-l-2 border-[var(--green)]">{item}</div>
          ))}
        </div>
      )}

      {pm.what_to_try_next?.length > 0 && (
        <div>
          <div className="flex items-center gap-1 text-[11px] font-semibold text-[var(--blue)] uppercase mb-1">
            <Lightbulb className="h-3 w-3" /> Recommendations
          </div>
          {pm.what_to_try_next.map((item, i) => (
            <div key={i} className="text-xs text-[var(--text2)] py-1 pl-3 border-l-2 border-[var(--blue)]">{item}</div>
          ))}
        </div>
      )}

      {(pm.api_call_count !== undefined || pm.unnecessary_calls?.length) && (
        <div className="text-[11px] text-[var(--text2)] flex gap-4 pt-1 border-t border-[var(--border)]">
          {pm.api_call_count !== undefined && <span>API calls: {pm.api_call_count}</span>}
          {pm.error_call_count !== undefined && <span className="text-[var(--red)]">Errors: {pm.error_call_count}</span>}
        </div>
      )}
    </div>
  )
}
