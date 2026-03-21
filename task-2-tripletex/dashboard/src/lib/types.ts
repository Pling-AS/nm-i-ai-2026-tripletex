export interface RunSummary {
  run_id: string
  filename: string
  started_at: string
  prompt: string
  task_type: string
  goal: string
  status: 'running' | 'completed' | 'error' | 'incomplete' | 'unknown'
  source: 'competition' | 'simulation'
  duration_seconds: number | null
  tripletex_call_count: number
  tripletex_error_count: number
  tripletex_call_log: ApiCallLog[]
  tool_call_count: number
  tool_error_count: number
  file_count: number
  summary: string
  error_message: string
  metadata: Record<string, any>
  event_count: number
  competition_score?: CompetitionScore | null
  enforcer_rejections: EnforcerEvent[]
  semantic_rejections: EnforcerEvent[]
  enforcer_overrides: EnforcerEvent[]
  preflight_rejections: PreflightEvent[]
  preflight_rejection_count: number
  auto_stripped: AutoStripped[]
  planner_payload: Record<string, any>
  execution_brief: Record<string, any>
  executor_system_prompt: string
}

export interface CompetitionScore {
  score_raw: number
  score_max: number
  normalized_score: number
  checks_passed: number
  checks_total: number
  comment: string
  checks: string[]
  submission_id: string
  status: string
}

export interface PostMortemPayload {
  headline: string
  what_went_wrong: string[]
  what_worked: string[]
  what_to_try_next: string[]
  confidence: 'high' | 'medium' | 'low'
  root_cause_category: string
  api_call_count?: number
  error_call_count?: number
  unnecessary_calls?: string[]
}

export interface TraceEvent {
  timestamp: string
  event_type: string
  payload: Record<string, any>
}

export interface ApiCallLog {
  method: string
  path: string
  status_code: number
  ok: boolean
}

export interface EnforcerEvent {
  reason: string
  suggestion?: string
  tool_name?: string
  [key: string]: any
}

export interface PreflightEvent {
  error: string
  fields: string[]
  tool_name: string
}

export interface AutoStripped {
  fields: string[]
  tool_name: string
}

export interface Submission {
  id: string
  status: string
  score_raw: number
  score_max: number
  normalized_score: number
  created_at: string
  feedback?: {
    checks: string[]
    comment: string
  }
  [key: string]: any
}

export interface BatchResult {
  score: number
  errors: number
  calls: number
}

export interface Settings {
  model: string
  planner_model: string
  tier1_executor_model: string
  tier2_executor_model: string
  tier3_executor_model: string
  max_steps: number
  temperature: number
}

export type WsMessage =
  | { type: 'snapshot'; payload: { runs: RunSummary[]; active_count: number; total_count: number } }
  | { type: 'settings'; payload: Settings }
  | { type: 'run_event'; payload: { run_id: string; event_type: string; timestamp: string } }
  | { type: 'heartbeat' }
  | { type: 'pong' }
