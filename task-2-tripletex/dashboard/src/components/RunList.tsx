'use client'

import { Search } from 'lucide-react'
import { cn } from '@/lib/utils'
import { RunCard } from './RunCard'
import type { RunSummary } from '@/lib/types'

interface RunListProps {
  runs: RunSummary[]
  selectedRunId: string | null
  onSelectRun: (id: string) => void
  searchQuery: string
  onSearchChange: (q: string) => void
  statusFilter: 'all' | 'running' | 'completed' | 'error'
  onStatusFilterChange: (f: 'all' | 'running' | 'completed' | 'error') => void
  sourceFilter: 'all' | 'competition' | 'simulation'
  onSourceFilterChange: (f: 'all' | 'competition' | 'simulation') => void
}

const statusOptions: { value: 'all' | 'running' | 'completed' | 'error'; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'running', label: 'Running' },
  { value: 'completed', label: 'Done' },
  { value: 'error', label: 'Error' },
]

const sourceOptions: { value: 'all' | 'competition' | 'simulation'; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'competition', label: 'Comp' },
  { value: 'simulation', label: 'Sim' },
]

export function RunList({
  runs, selectedRunId, onSelectRun, searchQuery, onSearchChange,
  statusFilter, onStatusFilterChange, sourceFilter, onSourceFilterChange,
}: RunListProps) {
  const filteredRuns = runs.filter((run) => {
    const matchesStatus = statusFilter === 'all' || run.status === statusFilter || (statusFilter === 'error' && run.status === 'incomplete')
    const matchesSource = sourceFilter === 'all' || run.source === sourceFilter
    const q = searchQuery.trim().toLowerCase()
    const matchesQuery = !q || [run.prompt, run.task_type, run.run_id, run.goal].some(v => (v ?? '').toLowerCase().includes(q))
    return matchesStatus && matchesSource && matchesQuery
  })

  return (
    <aside className="flex w-80 shrink-0 flex-col border-r border-[var(--border)] bg-[var(--surface)]">
      <div className="p-2 space-y-2">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-[var(--text2)]" />
          <input
            type="text"
            placeholder="Search runs..."
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            className="w-full rounded-md border border-[var(--border)] bg-[var(--bg)] py-1.5 pl-8 pr-3 text-xs text-[var(--text)] placeholder:text-[var(--text2)] focus:border-[var(--blue)] focus:outline-none"
          />
        </div>

        <div className="flex gap-1">
          {statusOptions.map(({ value, label }) => (
            <button
              key={value}
              type="button"
              onClick={() => onStatusFilterChange(value)}
              className={cn(
                'flex-1 rounded px-2 py-1 text-[11px] font-medium transition',
                statusFilter === value ? 'bg-[var(--blue)] text-white' : 'bg-[var(--surface2)] text-[var(--text2)] hover:text-[var(--text)]',
              )}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="flex gap-1">
          {sourceOptions.map(({ value, label }) => (
            <button
              key={value}
              type="button"
              onClick={() => onSourceFilterChange(value)}
              className={cn(
                'flex-1 rounded px-2 py-1 text-[11px] font-medium transition',
                sourceFilter === value ? 'bg-purple-500/30 text-purple-300' : 'bg-[var(--surface2)] text-[var(--text2)] hover:text-[var(--text)]',
              )}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="text-[11px] text-[var(--text2)]">{filteredRuns.length} runs</div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {filteredRuns.map((run) => (
          <RunCard key={run.run_id} run={run} isSelected={run.run_id === selectedRunId} onClick={() => onSelectRun(run.run_id)} />
        ))}
        {filteredRuns.length === 0 && (
          <div className="p-4 text-center text-xs text-[var(--text2)]">No runs match filters</div>
        )}
      </div>
    </aside>
  )
}
