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

  setRuns: (runs, activeCount, totalCount) => set({ runs, activeCount, totalCount }),
  updateRun: (runId, updates) => set((state) => ({
    runs: state.runs.map(r => r.run_id === runId ? { ...r, ...updates } : r)
  })),
  removeRun: (runId) => set((state) => ({
    runs: state.runs.filter(r => r.run_id !== runId),
    totalCount: state.totalCount - 1,
  })),
  setSettings: (settings) => set({ settings }),
  setConnectionStatus: (status) => set({ connectionStatus: status }),
  toggleSound: () => set((state) => ({ soundEnabled: !state.soundEnabled })),
  setBestScore: (score) => set({ bestScore: score }),
}))
