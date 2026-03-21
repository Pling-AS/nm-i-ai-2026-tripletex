'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { Header, type DashboardView } from '@/components/Header'
import { RunList } from '@/components/RunList'
import { RunDetail } from '@/components/RunDetail'
import { CompetitionView } from '@/components/CompetitionView'
import { RawRequestsView } from '@/components/RawRequestsView'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useSounds } from '@/hooks/useSounds'
import { useStore } from '@/lib/store'

export default function Home() {
  const { connectionStatus } = useWebSocket()
  const { playStart, playComplete, playError } = useSounds()
  const { runs, activeCount, totalCount, settings, soundEnabled, bestScore } = useStore()

  const [activeView, setActiveView] = useState<DashboardView>('runs')
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<'all' | 'running' | 'completed' | 'error'>('all')
  const [sourceFilter, setSourceFilter] = useState<'all' | 'competition' | 'simulation'>('all')

  const prevRunStatusesRef = useRef<Map<string, string>>(new Map())
  const initializedRef = useRef(false)

  useEffect(() => {
    if (!initializedRef.current) {
      prevRunStatusesRef.current = new Map(runs.map((r) => [r.run_id, r.status]))
      initializedRef.current = true
      return
    }

    const prev = prevRunStatusesRef.current
    for (const run of runs) {
      const previousStatus = prev.get(run.run_id)
      if (!previousStatus && run.status === 'running') {
        playStart()
      }
      if (previousStatus === 'running' && run.status === 'completed') {
        playComplete()
      }
      if ((previousStatus === 'running' || !previousStatus) && run.status === 'error') {
        playError()
      }
    }
    prevRunStatusesRef.current = new Map(runs.map((r) => [r.run_id, r.status]))
  }, [runs, playComplete, playError, playStart])

  const filteredRuns = useMemo(() => {
    const query = searchQuery.trim().toLowerCase()
    return runs.filter((run) => {
      const matchesStatus =
        statusFilter === 'all' ? true : statusFilter === 'error' ? run.status === 'error' || run.status === 'incomplete' : run.status === statusFilter
      const matchesSource = sourceFilter === 'all' ? true : run.source === sourceFilter
      const matchesQuery =
        query.length === 0
          ? true
          : [run.prompt, run.task_type, run.run_id, run.goal].some((value) =>
              (value ?? '').toLowerCase().includes(query),
            )

      return matchesStatus && matchesSource && matchesQuery
    })
  }, [runs, searchQuery, sourceFilter, statusFilter])

  useEffect(() => {
    if (!selectedRunId && filteredRuns.length > 0) {
      setSelectedRunId(filteredRuns[0].run_id)
    }
    if (selectedRunId && !runs.some((r) => r.run_id === selectedRunId)) {
      setSelectedRunId(filteredRuns[0]?.run_id ?? null)
    }
  }, [filteredRuns, runs, selectedRunId])

  return (
    <div className="flex h-screen min-h-screen w-full flex-col bg-[var(--bg)] text-[var(--text)]">
      <Header
        activeView={activeView}
        onViewChange={setActiveView}
        plannerModel={settings?.planner_model ?? ''}
        tier1Model={settings?.tier1_executor_model ?? ''}
        tier2Model={settings?.tier2_executor_model ?? ''}
        tier3Model={settings?.tier3_executor_model ?? ''}
        activeCount={activeCount}
        totalCount={totalCount}
        bestScore={bestScore}
        connectionStatus={connectionStatus}
        soundEnabled={soundEnabled}
      />

      <main className="flex min-h-0 flex-1 overflow-hidden">
        {activeView === 'runs' && (
          <>
            <RunList
              runs={runs}
              selectedRunId={selectedRunId}
              onSelectRun={setSelectedRunId}
              searchQuery={searchQuery}
              onSearchChange={setSearchQuery}
              statusFilter={statusFilter}
              onStatusFilterChange={setStatusFilter}
              sourceFilter={sourceFilter}
              onSourceFilterChange={setSourceFilter}
            />
            <section className="min-w-0 flex-1 overflow-hidden border-l border-[var(--border)] bg-[var(--bg)]">
              {selectedRunId ? (
                <RunDetail runId={selectedRunId} />
              ) : (
                <div className="flex h-full items-center justify-center text-sm text-[var(--text2)]">
                  Select a run to inspect details
                </div>
              )}
            </section>
          </>
        )}

        {activeView === 'competition' && (
          <section className="min-w-0 flex-1 overflow-auto p-4">
            <CompetitionView />
          </section>
        )}

        {activeView === 'raw-requests' && (
          <section className="min-w-0 flex-1 overflow-auto p-4">
            <RawRequestsView />
          </section>
        )}
      </main>
    </div>
  )
}
