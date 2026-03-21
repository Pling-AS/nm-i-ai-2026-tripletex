'use client'

import { useState } from 'react'
import { ChevronRight, ChevronDown } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { TraceEvent } from '@/lib/types'

const typeColors: Record<string, string> = {
  init: 'bg-white',
  planner: 'bg-cyan-400',
  thinking: 'bg-purple-400',
  assistant_reasoning: 'bg-purple-300',
  tool_start: 'bg-blue-400',
  tool_result: 'bg-emerald-400',
  error: 'bg-red-500',
  done: 'bg-green-500',
  enforcer_rejected: 'bg-orange-400',
  semantic_enforcer_rejected: 'bg-orange-300',
  enforcer_override: 'bg-yellow-400',
  competition_scoring: 'bg-yellow-500',
  error_summary: 'bg-red-400',
  post_mortem: 'bg-purple-500',
  execution_brief: 'bg-sky-400',
  final_payload: 'bg-green-400',
  metadata_update: 'bg-zinc-400',
}

interface TimelineProps {
  events: TraceEvent[]
}

function TimelineEvent({ event }: { event: TraceEvent }) {
  const [expanded, setExpanded] = useState(false)
  const dotColor = typeColors[event.event_type] ?? 'bg-zinc-500'
  const time = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : ''

  const isError = event.event_type === 'tool_result' && event.payload?.result && !event.payload.result.ok
  const actualDotColor = isError ? 'bg-red-500' : dotColor

  return (
    <div className="group">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-2 py-1 text-left hover:bg-[var(--surface2)] rounded transition"
      >
        <span className={cn('h-2 w-2 rounded-full shrink-0', actualDotColor)} />
        <span className="text-[11px] text-[var(--text2)] font-mono w-20 shrink-0">{time}</span>
        <span className="text-xs font-medium text-[var(--text)] truncate">{event.event_type}</span>
        {event.payload?.tool_name && (
          <span className="text-[11px] text-[var(--text2)] font-mono truncate">{event.payload.tool_name}</span>
        )}
        <span className="ml-auto text-[var(--text2)]">
          {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        </span>
      </button>

      {expanded && (
        <div className="ml-4 mr-2 mb-1 overflow-auto rounded bg-[var(--bg)] border border-[var(--border)] p-2 max-h-64">
          <pre className="text-[11px] text-[var(--text2)] font-mono whitespace-pre-wrap break-all">
            {JSON.stringify(event.payload, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

export function Timeline({ events }: TimelineProps) {
  const [showAll, setShowAll] = useState(false)
  const visible = showAll ? events : events.slice(0, 100)

  return (
    <div className="space-y-0.5">
      {visible.map((event, i) => (
        <TimelineEvent key={i} event={event} />
      ))}
      {!showAll && events.length > 100 && (
        <button
          type="button"
          onClick={() => setShowAll(true)}
          className="w-full text-center py-2 text-xs text-[var(--blue)] hover:underline"
        >
          Show all {events.length} events
        </button>
      )}
    </div>
  )
}
