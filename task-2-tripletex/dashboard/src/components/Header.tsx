'use client'

import { useState } from 'react'
import { Activity, Database, Trophy, Volume2, VolumeX, Wifi, WifiOff, Plus, FlaskConical, Loader2 } from 'lucide-react'
import { shortModel, cn } from '@/lib/utils'
import { useStore } from '@/lib/store'
import { api } from '@/lib/api'

export type DashboardView = 'runs' | 'competition' | 'raw-requests'

interface HeaderProps {
  activeView: DashboardView
  onViewChange: (view: DashboardView) => void
  plannerModel: string
  tier1Model: string
  tier2Model: string
  tier3Model: string
  activeCount: number
  totalCount: number
  bestScore: number
  connectionStatus: 'connecting' | 'live' | 'reconnecting' | 'offline'
  soundEnabled: boolean
}

function connectionColor(status: HeaderProps['connectionStatus']) {
  if (status === 'live') return 'bg-[var(--green)]'
  if (status === 'reconnecting') return 'bg-[var(--yellow)]'
  if (status === 'offline') return 'bg-[var(--red)]'
  return 'bg-zinc-500'
}

export function Header({
  activeView, onViewChange, plannerModel, tier1Model, tier2Model, tier3Model,
  activeCount, totalCount, bestScore, connectionStatus, soundEnabled,
}: HeaderProps) {
  const { toggleSound } = useStore()
  const [simLoading, setSimLoading] = useState(false)
  const [submitLoading, setSubmitLoading] = useState(false)
  const [batchMenu, setBatchMenu] = useState(false)
  const [batchCount, setBatchCount] = useState(5)

  async function handleSimulate() {
    setSimLoading(true)
    try {
      await api.simulateRun()
    } catch (e) { console.error(e) }
    setTimeout(() => setSimLoading(false), 2000)
  }

  async function handleQuickSubmit() {
    setSubmitLoading(true)
    try {
      await api.submit()
    } catch (e) { console.error(e) }
    setTimeout(() => setSubmitLoading(false), 2000)
  }

  async function handleBatchSubmit(count: number) {
    setBatchMenu(false)
    setSubmitLoading(true)
    try {
      await api.startBatch(count, 2)
    } catch (e) { console.error(e) }
    setTimeout(() => setSubmitLoading(false), 2000)
  }

  return (
    <header className="border-b border-[var(--border)] bg-[var(--surface)] px-4 py-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-md bg-[var(--surface2)] text-[var(--blue)]">
            <Activity className="h-5 w-5" />
          </div>
          <div>
            <div className="text-sm font-semibold tracking-wide">Tripletex Agent</div>
            <div className="mt-1 flex flex-wrap gap-1.5 text-[11px]">
              <span className="rounded border border-[var(--border)] bg-[var(--surface2)] px-2 py-0.5 text-cyan-300">
                P: {shortModel(plannerModel)}
              </span>
              <span className="rounded border border-[var(--border)] bg-[var(--surface2)] px-2 py-0.5 text-violet-300">
                T1: {shortModel(tier1Model)}
              </span>
              <span className="rounded border border-[var(--border)] bg-[var(--surface2)] px-2 py-0.5 text-fuchsia-300">
                T2: {shortModel(tier2Model)}
              </span>
              <span className="rounded border border-[var(--border)] bg-[var(--surface2)] px-2 py-0.5 text-indigo-300">
                T3: {shortModel(tier3Model)}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <div className="relative">
            <button
              type="button"
              onClick={handleQuickSubmit}
              onContextMenu={(e) => { e.preventDefault(); setBatchMenu(!batchMenu) }}
              disabled={submitLoading}
              className="flex items-center gap-1 rounded-md border border-[var(--green)]/50 bg-green-500/10 px-2.5 py-1.5 text-[var(--green)] font-medium transition hover:bg-green-500/20 disabled:opacity-50"
              title="Click: single run · Right-click: batch"
            >
              {submitLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
              Run
            </button>
            {batchMenu && (
              <div className="absolute top-full right-0 mt-1 z-50 rounded-lg border border-[var(--border)] bg-[var(--surface)] p-3 shadow-xl min-w-[180px]">
                <div className="text-[11px] text-[var(--text2)] font-medium mb-2">Batch Submit</div>
                <div className="flex items-center gap-2 mb-2">
                  <input
                    type="number"
                    value={batchCount}
                    onChange={(e) => setBatchCount(Number(e.target.value))}
                    min={1} max={50}
                    className="w-16 rounded border border-[var(--border)] bg-[var(--bg)] px-2 py-1 text-xs focus:border-[var(--blue)] focus:outline-none"
                  />
                  <span className="text-xs text-[var(--text2)]">runs</span>
                </div>
                <div className="flex gap-1">
                  <button type="button" onClick={() => handleBatchSubmit(batchCount)}
                    className="flex-1 rounded bg-[var(--green)] px-2 py-1 text-xs font-medium text-white">
                    Start
                  </button>
                  <button type="button" onClick={() => setBatchMenu(false)}
                    className="rounded border border-[var(--border)] px-2 py-1 text-xs text-[var(--text2)]">
                    Cancel
                  </button>
                </div>
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={handleSimulate}
            disabled={simLoading}
            className="flex items-center gap-1 rounded-md border border-[var(--purple)]/50 bg-purple-500/10 px-2.5 py-1.5 text-[var(--purple)] font-medium transition hover:bg-purple-500/20 disabled:opacity-50"
          >
            {simLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <FlaskConical className="h-3.5 w-3.5" />}
            Sim
          </button>

          <div className="w-px h-5 bg-[var(--border)] mx-1" />

          <div className="flex items-center gap-1 rounded-md border border-[var(--border)] bg-[var(--surface2)] px-2 py-1">
            <Activity className={cn('h-3.5 w-3.5', activeCount > 0 ? 'text-[var(--green)]' : 'text-[var(--text2)]')} />
            <span className={cn(activeCount > 0 && 'animate-pulse text-[var(--green)]')}>{activeCount} active</span>
          </div>
          <div className="flex items-center gap-1 rounded-md border border-[var(--border)] bg-[var(--surface2)] px-2 py-1">
            <Database className="h-3.5 w-3.5 text-[var(--text2)]" />
            <span>{totalCount}</span>
          </div>
          <div className="flex items-center gap-1 rounded-md border border-[var(--border)] bg-[var(--surface2)] px-2 py-1 text-[var(--yellow)]">
            <Trophy className="h-3.5 w-3.5" />
            <span>{bestScore.toFixed(2)}</span>
          </div>
          <button type="button" onClick={toggleSound}
            className="rounded-md border border-[var(--border)] bg-[var(--surface2)] p-1.5 text-[var(--text2)] transition hover:text-[var(--text)]">
            {soundEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
          </button>
          <div className="flex items-center gap-1 rounded-md border border-[var(--border)] bg-[var(--surface2)] px-2 py-1">
            {connectionStatus === 'offline' ? (
              <WifiOff className="h-3.5 w-3.5 text-[var(--red)]" />
            ) : (
              <Wifi className="h-3.5 w-3.5 text-[var(--text2)]" />
            )}
            <span className={cn('inline-block h-2 w-2 rounded-full', connectionColor(connectionStatus))} />
            <span>{connectionStatus}</span>
          </div>
        </div>
      </div>

      <nav className="mt-3 flex items-center gap-2">
        {([
          ['runs', 'Runs'],
          ['competition', 'Competition'],
          ['raw-requests', 'Raw Requests'],
        ] as [DashboardView, string][]).map(([view, label]) => (
          <button key={view} type="button" onClick={() => onViewChange(view)}
            className={cn(
              'rounded-md border px-3 py-1.5 text-xs font-medium transition',
              activeView === view
                ? 'border-[var(--blue)] bg-blue-500/20 text-blue-300'
                : 'border-[var(--border)] bg-[var(--surface2)] text-[var(--text2)] hover:text-[var(--text)]',
            )}>
            {label}
          </button>
        ))}
      </nav>
    </header>
  )
}
