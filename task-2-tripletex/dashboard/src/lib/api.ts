const API_BASE = ''

export const api = {
  getRuns: () => fetch(`${API_BASE}/api/runs`).then(r => r.json()),
  getRunDetail: (runId: string) => fetch(`${API_BASE}/api/runs/${runId}`).then(r => r.json()),
  enrichRuns: () => fetch(`${API_BASE}/api/runs/enrich`, { method: 'POST' }).then(r => r.json()),
  deleteRun: (runId: string) => fetch(`${API_BASE}/api/runs/${runId}`, { method: 'DELETE' }).then(r => r.json()),
  generatePostMortem: (runId: string) => fetch(`${API_BASE}/api/runs/${runId}/postmortem`, { method: 'POST' }).then(r => r.json()),
  simulateRun: () => fetch(`${API_BASE}/api/runs/simulate`, { method: 'POST' }).then(r => r.json()),
  getSettings: () => fetch(`${API_BASE}/api/settings`).then(r => r.json()),
  getSubmissions: () => fetch(`${API_BASE}/api/competition/submissions`).then(r => r.json()),
  submit: (endpointUrl?: string, apiKey?: string) => fetch(`${API_BASE}/api/competition/submit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ...(endpointUrl ? { endpoint_url: endpointUrl } : {}),
      ...(apiKey ? { endpoint_api_key: apiKey } : {}),
    })
  }).then(r => r.json()),
  startBatch: (count: number, concurrency: number) => fetch(`${API_BASE}/api/competition/batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ count, concurrency })
  }).then(r => r.json()),
  stopBatch: () => fetch(`${API_BASE}/api/competition/batch/stop`, { method: 'POST' }).then(r => r.json()),
  getBatchStatus: () => fetch(`${API_BASE}/api/competition/batch/status`).then(r => r.json()),
  getRawRequests: () => fetch(`${API_BASE}/api/raw-requests`).then(r => r.json()),
}
