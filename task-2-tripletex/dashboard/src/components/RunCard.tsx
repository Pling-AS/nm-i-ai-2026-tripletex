'use client'

import { useState, useEffect } from 'react'
import { Loader2, Trash2 } from 'lucide-react'
import { cn, formatDuration, formatTimeAgo, truncate } from '@/lib/utils'
import { api } from '@/lib/api'
import { useStore } from '@/lib/store'
import type { RunSummary } from '@/lib/types'

function useElapsed(startedAt: string | undefined, isRunning: boolean) {
  const [elapsed, setElapsed] = useState<number | null>(null)
  useEffect(() => {
    if (!isRunning || !startedAt) { setElapsed(null); return }
    const start = new Date(startedAt).getTime()
    const tick = () => setElapsed(Math.floor((Date.now() - start) / 1000))
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [startedAt, isRunning])
  return elapsed
}

const statusColors: Record<string, string> = {
  running: 'bg-[var(--blue)]',
  completed: 'bg-[var(--green)]',
  error: 'bg-[var(--red)]',
  incomplete: 'bg-[var(--yellow)]',
  unknown: 'bg-zinc-500',
}

interface RunCardProps {
  run: RunSummary
  isSelected: boolean
  onClick: () => void
}

export function RunCard({ run, isSelected, onClick }: RunCardProps) {
  const { removeRun } = useStore()
  const [deleting, setDeleting] = useState(false)
  const score = run.competition_score
  const isRunning = run.status === 'running'
  const elapsed = useElapsed(run.started_at, isRunning)
  const isRecent = run.started_at ? (Date.now() - new Date(run.started_at).getTime()) < 10 * 60 * 1000 : false
  const isAwaitingScore = run.source === 'competition' && run.status === 'completed' && !score && isRecent
  const scoreStatus = score?.status
  const isScoring = scoreStatus === 'in_progress' || scoreStatus === 'pending'

  async function handleDelete(e: React.MouseEvent) {
    e.stopPropagation()
    setDeleting(true)
    try {
      await api.deleteRun(run.run_id)
      removeRun(run.run_id)
    } catch (err) { console.error(err) }
    setDeleting(false)
  }

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'group w-full text-left px-3 py-2.5 border-b border-[var(--border)] transition-colors border-l-2',
        isSelected
          ? 'bg-blue-500/10 border-l-[var(--blue)]'
          : 'hover:bg-[var(--surface2)] border-l-transparent',
        run.status === 'running' && 'bg-blue-500/5 border-l-[var(--blue)]',
        run.status === 'error' && !isSelected && 'border-l-[var(--red)]/50',
      )}
    >
      <div className="flex items-center gap-2 mb-1">
        <span className={cn('inline-block h-2 w-2 rounded-full shrink-0', statusColors[run.status] ?? 'bg-zinc-500', run.status === 'running' && 'animate-pulse')} />
        {run.status === 'running' && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-300 font-semibold uppercase tracking-wider animate-pulse">running</span>
        )}
        <span className="text-xs font-medium text-[var(--text)]">{run.task_type || 'unknown'}</span>
        {run.metadata?.hostname && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/15 text-cyan-300 font-medium truncate max-w-[60px]" title={run.metadata.hostname}>
            {run.metadata.hostname.split('.')[0]}
          </span>
        )}
        <span className={cn('ml-auto text-[10px] px-1.5 py-0.5 rounded font-medium',
          run.source === 'competition' ? 'bg-purple-500/20 text-purple-300' : 'bg-zinc-500/20 text-zinc-400'
        )}>
          {run.source}
        </span>
        <button
          type="button"
          onClick={handleDelete}
          disabled={deleting}
          className="hidden group-hover:block p-0.5 rounded text-[var(--text2)] hover:text-[var(--red)] transition"
        >
          {deleting ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
        </button>
      </div>

      <div className="text-xs text-[var(--text2)] mb-1.5 leading-relaxed">
        {truncate(run.prompt || run.goal || '-', 80)}
      </div>

      <div className="flex items-center gap-3 text-[11px] text-[var(--text2)]">
        <span className={isRunning ? 'text-[var(--blue)] tabular-nums' : 'tabular-nums'}>
          {isRunning && elapsed != null ? formatDuration(elapsed) : formatDuration(run.duration_seconds)}
        </span>
        <span className={cn(isRunning && 'text-[var(--blue)]')}>{run.tripletex_call_count} calls</span>
        {run.tripletex_error_count > 0 && (
          <span className="text-[var(--red)]">{run.tripletex_error_count} err</span>
        )}

        <span className="ml-auto flex items-center gap-1">
          {score && score.status === 'completed' && (
            <>
              <span className="text-[var(--text2)]">{score.score_raw?.toFixed(1) ?? '-'}/{score.score_max?.toFixed(1) ?? '-'}</span>
              <span className={cn('font-bold',
                (score.normalized_score ?? 0) >= 3.0 ? 'text-[var(--green)]' :
                (score.normalized_score ?? 0) >= 1.5 ? 'text-[var(--yellow)]' : 'text-[var(--red)]'
              )}>
                {score.normalized_score != null ? score.normalized_score.toFixed(2) : '-'}
              </span>
            </>
          )}
          {(isAwaitingScore || isScoring) && (
            <span className="flex items-center gap-1 text-[var(--yellow)]">
              <Loader2 className="h-3 w-3 animate-spin" />
              {isScoring ? 'scoring' : 'awaiting'}
            </span>
          )}
        </span>

        <span className="text-[10px]">{formatTimeAgo(run.started_at)}</span>
      </div>
    </button>
  )
}
