'use client'

import { useState, useRef, useEffect } from 'react'
import { ChevronRight, ChevronDown, Loader2 } from 'lucide-react'
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

const methodColors: Record<string, string> = {
  GET: 'text-[var(--green)]',
  POST: 'text-[var(--blue)]',
  PUT: 'text-[var(--yellow)]',
  DELETE: 'text-[var(--red)]',
  PATCH: 'text-[var(--purple)]',
}

function statusColor(code: number): string {
  if (code < 300) return 'text-[var(--green)]'
  if (code < 400) return 'text-[var(--yellow)]'
  return 'text-[var(--red)]'
}

function InlineToolInfo({ event }: { event: TraceEvent }) {
  const p = event.payload
  if (event.event_type === 'tool_start' && p?.tool_name === 'tripletex_request') {
    const args = p.arguments ?? {}
    const bodyStr = JSON.stringify(args.json_body ?? args.params ?? {})
    const hasRefs = bodyStr.includes('$REF:')
    const refCount = (bodyStr.match(/\$REF:/g) || []).length
    return (
      <span className="flex items-center gap-1.5 text-[11px] font-mono truncate">
        <span className={cn('font-semibold', methodColors[args.method] ?? '')}>{args.method}</span>
        <span className="text-[var(--text2)] truncate">{args.path}</span>
        {hasRefs && <span className="text-[var(--cyan)] shrink-0">{refCount} $REF</span>}
      </span>
    )
  }
  if (event.event_type === 'tool_result' && p?.tool_name === 'tripletex_request') {
    const result = p.result ?? {}
    const code = result.status_code
    const ok = result.ok
    const refs = result.available_refs
    const refCount = refs ? Object.keys(refs).length : 0
    return (
      <span className="flex items-center gap-1.5 text-[11px] font-mono">
        {code && <span className={cn('font-semibold', statusColor(code))}>{code}</span>}
        <span className={ok ? 'text-[var(--green)]' : 'text-[var(--red)]'}>{ok ? 'OK' : 'FAIL'}</span>
        {result.summary && <span className="text-[var(--text2)] truncate max-w-[200px]">{result.summary}</span>}
        {!ok && result.error && <span className="text-[var(--red)] truncate max-w-[150px]">{typeof result.error === 'string' ? result.error : 'error'}</span>}
        {refCount > 0 && <span className="text-[var(--cyan)] shrink-0">{refCount} refs</span>}
      </span>
    )
  }
  if (event.event_type === 'tool_start' && p?.tool_name) {
    return <span className="text-[11px] text-[var(--text2)] font-mono">{p.tool_name}</span>
  }
  if (event.event_type === 'attachments_prepared' && p?.pdf_extractions?.length > 0) {
    return (
      <span className="flex items-center gap-1.5 text-[11px]">
        {p.pdf_extractions.map((ext: { filename: string; method: string; cached: boolean; char_count: number }, i: number) => (
          <span key={i} className="flex items-center gap-1">
            <span className="text-[var(--text2)]">{ext.filename}</span>
            <span className={ext.method === 'datalab' ? 'text-[var(--green)]' : 'text-[var(--yellow)]'}>{ext.method}</span>
            {ext.cached && <span className="text-[var(--cyan)]">cached</span>}
            <span className="text-[var(--text2)]">{ext.char_count}ch</span>
          </span>
        ))}
      </span>
    )
  }
  if (event.event_type === 'thinking' && p?.text) {
    return <span className="text-[11px] text-[var(--text2)] italic truncate max-w-[300px]">{p.text.slice(0, 80)}</span>
  }
  if (event.event_type === 'planner' && p?.task_type) {
    return <span className="text-[11px] text-[var(--text2)]">{p.task_type}: {p.goal?.slice(0, 60)}</span>
  }
  if (event.event_type === 'error' && p?.message) {
    return <span className="text-[11px] text-[var(--red)] truncate max-w-[300px]">{p.message.slice(0, 80)}</span>
  }
  return null
}

function TimelineEvent({ event }: { event: TraceEvent }) {
  const [expanded, setExpanded] = useState(false)
  const dotColor = typeColors[event.event_type] ?? 'bg-zinc-500'
  const time = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : ''

  const isError = event.event_type === 'tool_result' && event.payload?.result && !event.payload.result.ok
  const actualDotColor = isError ? 'bg-red-500' : dotColor

  return (
    <div>
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-2 py-1.5 text-left hover:bg-[var(--surface2)] rounded transition"
      >
        <span className={cn('h-2 w-2 rounded-full shrink-0', actualDotColor)} />
        <span className="text-[10px] text-[var(--text2)] font-mono w-16 shrink-0">{time}</span>
        <span className="text-[11px] font-medium text-[var(--text)] shrink-0 whitespace-nowrap">{event.event_type}</span>
        <span className="min-w-0 flex-1 truncate"><InlineToolInfo event={event} /></span>
        <span className="ml-auto text-[var(--text2)] shrink-0">
          {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        </span>
      </button>

      {expanded && (
        <div className="ml-4 mr-2 mb-1 overflow-auto rounded bg-[var(--bg)] border border-[var(--border)] p-2 max-h-80">
          <pre className="text-[11px] text-[var(--text2)] font-mono whitespace-pre-wrap break-all">
            {JSON.stringify(event.payload, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

interface TimelineProps {
  events: TraceEvent[]
  isRunning?: boolean
}

export function Timeline({ events, isRunning }: TimelineProps) {
  const [showAll, setShowAll] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const visible = showAll ? events : events.slice(0, 200)

  useEffect(() => {
    if (isRunning && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [events.length, isRunning])

  return (
    <div className="space-y-0.5">
      {visible.map((event, i) => (
        <TimelineEvent key={`${event.timestamp}-${i}`} event={event} />
      ))}
      {isRunning && (() => {
        const last = events[events.length - 1]
        const et = last?.event_type ?? ''
        const p = last?.payload ?? {}
        const label =
          et === 'thinking' ? 'Thinking...' :
          et === 'assistant_reasoning' ? 'Reasoning...' :
          et === 'planner' ? 'Executing plan...' :
          et === 'tool_start' && p.tool_name === 'tripletex_request' ? `Waiting for ${p.arguments?.method ?? ''} ${p.arguments?.path ?? ''}...` :
          et === 'tool_start' ? `Waiting for ${p.tool_name}...` :
          et === 'tool_result' ? 'Thinking about next step...' :
          et === 'init' ? 'Creating plan...' :
          et === 'attachments_prepared' ? 'Creating plan...' :
          et === 'execution_brief' ? 'Starting execution...' :
          et === 'api_advisor_query' ? 'Waiting for API advisor...' :
          et === 'api_advisor_response' ? 'Thinking about next step...' :
          et === 'enforcer_passed' ? 'Executing API call...' :
          et === 'metadata_update' ? 'Creating plan...' :
          'Working...'
        return (
          <div className="flex items-center gap-2 px-2 py-2 text-xs text-[var(--blue)] animate-pulse">
            <Loader2 className="h-3 w-3 animate-spin" />
            {label}
          </div>
        )
      })()}
      {!showAll && events.length > 200 && (
        <button type="button" onClick={() => setShowAll(true)}
          className="w-full text-center py-2 text-xs text-[var(--blue)] hover:underline">
          Show all {events.length} events
        </button>
      )}
      <div ref={bottomRef} />
    </div>
  )
}
