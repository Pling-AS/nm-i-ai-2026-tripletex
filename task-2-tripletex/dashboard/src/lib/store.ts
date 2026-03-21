import { create } from 'zustand'
import { RunSummary, Settings, Submission } from './types'

interface DashboardState {
  runs: RunSummary[]
  activeCount: number
  totalCount: number
  settings: Settings | null
  connectionStatus: 'connecting' | 'live' | 'reconnecting' | 'offline'
  soundEnabled: boolean
  bestScore: number
  
  setRuns: (runs: RunSummary[], activeCount: number, totalCount: number) => void
  updateRun: (runId: string, updates: Partial<RunSummary>) => void
  removeRun: (runId: string) => void
  addPendingRun: (submissionId: string) => void
  setSettings: (settings: Settings) => void
  setConnectionStatus: (status: 'connecting' | 'live' | 'reconnecting' | 'offline') => void
  toggleSound: () => void
  setBestScore: (score: number) => void
}

export const useStore = create<DashboardState>((set) => ({
  runs: [],
  activeCount: 0,
  totalCount: 0,
  settings: null,
  connectionStatus: 'connecting',
  soundEnabled: true,
  bestScore: 0,

  setRuns: (runs, activeCount, totalCount) => set((state) => {
    const pendingRuns = state.runs.filter(r => r.run_id.startsWith('pending-'))
    const realSubIds = new Set(runs.map(r => r.metadata?.submission_id).filter(Boolean))
    const survivingPending = pendingRuns.filter(r => {
      const subId = r.run_id.replace('pending-', '')
      return !realSubIds.has(subId)
    })
    return {
      runs: [...survivingPending, ...runs],
      activeCount: activeCount + survivingPending.length,
      totalCount: totalCount + survivingPending.length,
    }
  }),
  updateRun: (runId, updates) => set((state) => ({
    runs: state.runs.map(r => r.run_id === runId ? { ...r, ...updates } : r)
  })),
  removeRun: (runId) => set((state) => ({
    runs: state.runs.filter(r => r.run_id !== runId),
    totalCount: state.totalCount - 1,
  })),
  addPendingRun: (submissionId) => set((state) => ({
    runs: [{
      run_id: `pending-${submissionId}`,
      filename: '',
      started_at: new Date().toISOString(),
      prompt: 'Submitting to competition...',
      task_type: '',
      goal: '',
      status: 'running' as const,
      source: 'competition' as const,
      duration_seconds: null,
      tripletex_call_count: 0,
      tripletex_error_count: 0,
      tripletex_call_log: [],
      tool_call_count: 0,
      tool_error_count: 0,
      file_count: 0,
      summary: '',
      error_message: '',
      metadata: { submission_id: submissionId },
      event_count: 0,
      enforcer_rejections: [],
      semantic_rejections: [],
      enforcer_overrides: [],
      preflight_rejections: [],
      preflight_rejection_count: 0,
      auto_stripped: [],
      planner_payload: {},
      execution_brief: {},
      executor_system_prompt: '',
    } as RunSummary, ...state.runs],
    activeCount: state.activeCount + 1,
    totalCount: state.totalCount + 1,
  })),
  setSettings: (settings) => set({ settings }),
  setConnectionStatus: (status) => set({ connectionStatus: status }),
  toggleSound: () => set((state) => ({ soundEnabled: !state.soundEnabled })),
  setBestScore: (score) => set({ bestScore: score }),
}))
