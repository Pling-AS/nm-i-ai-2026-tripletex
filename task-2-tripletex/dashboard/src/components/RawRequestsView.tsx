'use client'

import { useEffect, useState } from 'react'
import { Loader2, RefreshCw, ChevronRight, ChevronDown } from 'lucide-react'
import { api } from '@/lib/api'
import { cn } from '@/lib/utils'

interface RawRequest {
  timestamp: string
  method: string
  path: string
  status_code: number
  duration_ms?: number
  request_body?: unknown
  response_body?: unknown
  [key: string]: unknown
}

export function RawRequestsView() {
  const [requests, setRequests] = useState<RawRequest[]>([])
  const [loading, setLoading] = useState(true)
  const [expanded, setExpanded] = useState<Set<number>>(new Set())

  async function fetchRequests() {
    setLoading(true)
    try {
      const data = await api.getRawRequests()
      setRequests(data.requests ?? [])
    } catch (e) { console.error(e) }
    setLoading(false)
  }

  useEffect(() => { fetchRequests() }, [])

  function toggleExpand(i: number) {
    setExpanded(prev => {
      const next = new Set(prev)
      if (next.has(i)) next.delete(i)
      else next.add(i)
      return next
    })
  }

  const methodColors: Record<string, string> = {
    GET: 'text-[var(--green)]',
    POST: 'text-[var(--blue)]',
    PUT: 'text-[var(--yellow)]',
    DELETE: 'text-[var(--red)]',
    PATCH: 'text-[var(--purple)]',
  }

  if (loading) {
    return <div className="flex justify-center py-12"><Loader2 className="h-6 w-6 animate-spin text-[var(--text2)]" /></div>
  }

  return (
    <div className="max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold">Raw HTTP Requests ({requests.length})</h2>
        <button type="button" onClick={fetchRequests}
          className="flex items-center gap-1 rounded border border-[var(--border)] bg-[var(--surface2)] px-3 py-1.5 text-xs text-[var(--text2)] hover:text-[var(--text)] transition">
          <RefreshCw className="h-3 w-3" /> Refresh
        </button>
      </div>

      <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-[var(--border)] text-[var(--text2)]">
              <th className="w-6"></th>
              <th className="text-left px-3 py-2 font-medium">Time</th>
              <th className="text-left px-3 py-2 font-medium">Method</th>
              <th className="text-left px-3 py-2 font-medium">Path</th>
              <th className="text-left px-3 py-2 font-medium">Status</th>
              <th className="text-left px-3 py-2 font-medium">Duration</th>
            </tr>
          </thead>
          <tbody>
            {requests.map((req, i) => (
              <tr key={i} className="border-b border-[var(--border)] last:border-0">
                <td>
                  <button type="button" onClick={() => toggleExpand(i)} className="p-1 text-[var(--text2)] hover:text-[var(--text)]">
                    {expanded.has(i) ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                  </button>
                </td>
                <td className="px-3 py-1.5 font-mono text-[var(--text2)]">{req.timestamp ? new Date(req.timestamp).toLocaleTimeString() : '-'}</td>
                <td className={cn('px-3 py-1.5 font-mono font-medium', methodColors[req.method] ?? '')}>{req.method}</td>
                <td className="px-3 py-1.5 font-mono text-[var(--text2)] truncate max-w-md">{req.path}</td>
                <td className={cn('px-3 py-1.5 font-mono',
                  req.status_code < 300 ? 'text-[var(--green)]' :
                  req.status_code < 400 ? 'text-[var(--yellow)]' : 'text-[var(--red)]'
                )}>{req.status_code}</td>
                <td className="px-3 py-1.5 font-mono text-[var(--text2)]">{req.duration_ms != null ? `${req.duration_ms}ms` : '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {requests.length === 0 && (
          <div className="py-8 text-center text-xs text-[var(--text2)]">No raw requests recorded</div>
        )}
      </div>
    </div>
  )
}
