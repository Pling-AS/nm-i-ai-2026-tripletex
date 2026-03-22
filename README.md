# NM i AI 2026 — Tripletex (AI Accounting Agent)

**Team: Er det kå i?**

Our Task 2 submission for **NM i AI 2026** (Norwegian Championship in AI) — a 69-hour AI hackathon with a 1,000,000 NOK prize pool.

Task 1 (NorgesGruppen) and Task 3 (Astar Island) are in a [separate repo](https://github.com/Pling-AS/nm-i-ai-2026-norgesgruppen-astar).

---

## What It Does

An AI agent that completes accounting tasks in [Tripletex](https://www.tripletex.no/). It receives a natural-language prompt describing an accounting task (in one of 7 languages), uses the Tripletex REST API to execute it, and returns `{"status": "completed"}`.

**How the competition works:**
1. We submit our HTTPS endpoint URL to the platform
2. The platform provisions a fresh Tripletex sandbox account
3. It sends a random accounting task to our `/solve` endpoint
4. Our agent reads the prompt, processes any attachments (PDFs, images), and calls the Tripletex API to complete the task
5. The platform verifies the result field-by-field against expected values

**Task types include:** creating employees, customers, products, invoices, travel expenses, projects, departments, registering payments, issuing credit notes, and multi-step workflows.

## Scoring

- **30 task types** × **56 variants** (7 languages × 8 datasets) = 1,680 unique prompts
- **Field-by-field correctness** normalized to 0–1, multiplied by tier (×1, ×2, or ×3)
- **Efficiency bonus** (up to 2×) for perfect submissions with fewer API calls and zero errors
- **Score range:** 0.0–6.0 per task. Best score per task is kept — bad runs never lower your score
- **Leaderboard:** sum of best scores across all 30 task types

## Architecture

```
POST /solve
  → Parse prompt + attachments (PDF/image extraction)
  → Planner LLM → multi-step execution plan
  → Executor loop (up to 40 steps):
      → Just-in-time API spec lookup (spec_index.py)
      → Tripletex API call via authenticated proxy
      → Result validation + error recovery
      → Next step or complete
  → JSONL trace written to runs/
  → Response: { "status": "completed" }
```

**Key design decisions:**
- **Multi-model routing:** Different LLMs for planning vs execution vs enforcement, tiered by task complexity
- **Bounded tool layer:** All Tripletex API calls go through a typed wrapper with error handling and compact results
- **Just-in-time spec search:** Instead of stuffing the full API spec into context, we search `api_spec.json` on demand
- **Trace logging:** Every `/solve` call writes a full JSONL trace for debugging

## Project Layout

```
task-2-tripletex/
├── src/tripletex_agent/
│   ├── main.py                 # FastAPI app (POST /solve, GET /health)
│   ├── agent.py                # Core agent loop (planner → executor)
│   ├── config.py               # Pydantic settings from .env
│   ├── prompts.py              # System prompts for planner/executor
│   ├── executor_knowledge.py   # Tripletex API rules & constraints
│   ├── openrouter.py           # LLM client (OpenRouter / Azure / Vertex AI)
│   ├── tripletex.py            # Tripletex API tool layer
│   ├── spec_index.py           # API spec search
│   ├── schemas.py              # Pydantic models
│   └── files.py                # PDF/image attachment extraction
├── dashboard/                  # Next.js monitoring UI
├── tests/                      # Pytest suite
├── task_docs/                  # Tripletex API spec (api_spec.json)
├── runs/                       # JSONL trace files per /solve call
├── Dockerfile                  # Python 3.11 + uv + uvicorn
└── .env.example                # Configuration template
```

## Setup

```bash
cd task-2-tripletex
uv sync
cp .env.example .env
# Fill in OPENROUTER_API_KEY at minimum
```

### Run Locally

```bash
uvicorn tripletex_agent.main:app --reload --host 0.0.0.0 --port 8000
```


## Environment Variables

### Required

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | LLM API access via [OpenRouter](https://openrouter.ai/) |
| `TRIPLETEX_SANDBOX_API_URL` | Your Tripletex sandbox URL (e.g. `https://xxx.tripletex.dev/v2`) |
| `TRIPLETEX_SANDBOX_API_SESSION_TOKEN` | Session token for your sandbox account |
| `TRIPLETEX_SANDBOX_LOGIN_EMAIL` | Email used for sandbox login |
| `LOCAL_SOLVE_URL` | URL where your agent is running (e.g. `http://127.0.0.1:8000/solve`) |

Get sandbox credentials by creating a free Tripletex test account at [tripletex.no](https://www.tripletex.no/). The `LOCAL_SOLVE_URL` is used by the dashboard's batch runner to test your agent locally.

### Optional — LLM Providers

The agent routes LLM calls through multiple providers. Without Azure or Vertex AI configured, all calls go through OpenRouter.

| Variable | Default | Purpose |
|----------|---------|---------|
| `VERTEX_AI_PROJECT_ID` | _(empty = disabled)_ | GCP Vertex AI — set to enable as primary Claude backend |
| `VERTEX_AI_REGION` | `europe-west1` | Vertex AI region |
| `VERTEX_AI_ENABLED` | `true` | Master switch (needs PROJECT_ID to actually activate) |
| `AZURE_API_KEY` | _(empty = disabled)_ | Azure AI — used as fallback for Claude/OpenAI models |
| `AZURE_OPENAI_BASE_URL` | _(preset)_ | Azure OpenAI endpoint |
| `AZURE_ANTHROPIC_BASE_URL` | _(preset)_ | Azure Anthropic endpoint |

### Optional — Model Routing

| Variable | Default | Purpose |
|----------|---------|---------|
| `PLANNER_MODEL` | `gpt-5.4` | Model for planning step |
| `TIER1_EXECUTOR_MODEL` | `claude-sonnet-4-6` | Simple task executor |
| `TIER2_EXECUTOR_MODEL` | `claude-opus-4-6` | Medium task executor |
| `TIER3_EXECUTOR_MODEL` | `gpt-5.4` | Complex task executor |
| `AGENT_MAX_STEPS` | `40` | Max executor loop iterations |

### Optional — PDF Extraction

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATALAB_API_KEY` | _(empty = disabled)_ | [Datalab Marker API](https://www.datalab.to/) for high-quality PDF extraction (preserves æøå). Falls back to pymupdf/pypdf without it. |

### Optional — Endpoint Protection

| Variable | Default | Purpose |
|----------|---------|---------|
| `APP_API_KEY` | _(empty = disabled)_ | Bearer token to protect your `/solve` endpoint |

## Endpoint Contract

### `POST /solve`

**Request:**
```json
{
  "prompt": "Opprett en ansatt med navn Ola Nordmann, ola@example.org.",
  "files": [
    { "filename": "faktura.pdf", "content_base64": "JVBERi0...", "mime_type": "application/pdf" }
  ],
  "tripletex_credentials": {
    "base_url": "https://tx-proxy.ainm.no/v2",
    "session_token": "abc123"
  }
}
```

**Response:**
```json
{ "status": "completed" }
```

The agent authenticates with the Tripletex API using Basic Auth: username `0`, password = the provided `session_token`.

## Debug Traces

Each `/solve` call writes a JSONL trace to `runs/`, capturing the prompt, planner output, tool calls, results, and final stats. Useful for understanding why the agent made specific API calls.
