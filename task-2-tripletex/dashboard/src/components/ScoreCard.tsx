'use client'

import { Check, X } from 'lucide-react'
import type { CompetitionScore } from '@/lib/types'

interface ScoreCardProps {
  score: CompetitionScore
}

export function ScoreCard({ score }: ScoreCardProps) {
  const pct = (score.normalized_score * 100).toFixed(1)
  const checksPct = score.checks_total > 0 ? (score.checks_passed / score.checks_total) * 100 : 0

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
      <div className="flex items-center gap-4 mb-4">
        <div>
          <div className="text-[11px] text-[var(--text2)] uppercase font-medium mb-1">Score</div>
          <div className={`text-4xl font-black tabular-nums ${
            (score.normalized_score ?? 0) >= 3.0 ? 'text-[var(--green)]' :
            (score.normalized_score ?? 0) >= 1.5 ? 'text-[var(--yellow)]' : 'text-[var(--red)]'
          }`}>{score.normalized_score != null ? score.normalized_score.toFixed(2) : '-'}</div>
        </div>
        <div className="ml-auto text-right">
          <div className="text-[11px] text-[var(--text2)] uppercase font-medium mb-1">Raw Score</div>
          <div className="text-2xl font-bold tabular-nums">
            {score.score_raw?.toFixed(1) ?? '-'}
            <span className="text-[var(--text2)] text-lg"> / {score.score_max?.toFixed(1) ?? '-'}</span>
          </div>
        </div>
      </div>

      <div className="mb-3">
        <div className="flex justify-between text-xs text-[var(--text2)] mb-1">
          <span>Checks: {score.checks_passed}/{score.checks_total}</span>
          <span>{checksPct.toFixed(0)}%</span>
        </div>
        <div className="h-1.5 rounded-full bg-[var(--surface2)] overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${checksPct === 100 ? 'bg-[var(--green)]' : checksPct > 50 ? 'bg-[var(--yellow)]' : 'bg-[var(--red)]'}`}
            style={{ width: `${checksPct}%` }}
          />
        </div>
      </div>

      {score.checks.length > 0 && (
        <div className="space-y-1 max-h-48 overflow-y-auto">
          {score.checks.map((check, i) => {
            const passed = check.toLowerCase().includes('passed') || check.toLowerCase().includes('✓') || check.toLowerCase().includes('correct')
            return (
              <div key={i} className="flex items-start gap-2 text-xs">
                {passed ? (
                  <Check className="h-3.5 w-3.5 shrink-0 text-[var(--green)] mt-0.5" />
                ) : (
                  <X className="h-3.5 w-3.5 shrink-0 text-[var(--red)] mt-0.5" />
                )}
                <span className="text-[var(--text2)]">{check}</span>
              </div>
            )
          })}
        </div>
      )}

      {score.comment && (
        <div className="mt-3 rounded bg-[var(--surface2)] p-2 text-xs text-[var(--text2)]">{score.comment}</div>
      )}
    </div>
  )
}
