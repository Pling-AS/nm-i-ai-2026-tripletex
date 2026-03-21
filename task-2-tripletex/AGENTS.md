# AI ACCOUNTING AGENT — Tripletex

FastAPI service exposing `POST /solve` that completes accounting tasks in Tripletex sandbox environments. OpenRouter-powered planner + tool-calling executor with bounded Tripletex API access.

## STRUCTURE

```
task-2-tripletex/
├── src/tripletex_agent/           # Main agent package
│   ├── main.py                    #   FastAPI app (POST /solve, GET /health)
│   ├── agent.py                   #   Core agent loop (planner → executor)
│   ├── config.py                  #   Settings from env vars
│   ├── prompts.py                 #   System prompts for planner/executor
│   ├── executor_knowledge.py      #   Tripletex API rules & constraints (CRITICAL)
│   ├── openrouter.py              #   OpenRouter API client
│   ├── tripletex.py               #   Tripletex API tool layer
│   ├── spec_index.py              #   API spec search (task_docs/api_spec.json)
│   ├── schemas.py                 #   Pydantic models
│   ├── files.py                   #   Attachment preprocessing (PDF, images, text)
│   ├── trace.py                   #   JSONL trace writer → runs/
│   ├── run_store.py               #   Run history storage
│   ├── dashboard.py               #   Dashboard route handler
│   └── postmortem.py              #   Error analysis
├── dashboard/                     # Next.js monitoring UI (real-time + history)
├── tests/                         # Pytest suite (5 test files)
├── docs/                          # Task specification
├── task_docs/                     # Tripletex API spec (api_spec.json)
├── general_docs/                  # Additional reference docs
├── runs/                          # JSONL trace files (per /solve call)
├── Dockerfile                     # Python 3.11 slim + uv + uvicorn
├── .env.example                   # Required: OPENROUTER_API_KEY
└── pyproject.toml                 # FastAPI, httpx, openai SDK
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Agent loop | `src/tripletex_agent/agent.py` | Planner → tool calls → verify |
| API rules | `src/tripletex_agent/executor_knowledge.py` | Critical DO/DON'T for 422 prevention |
| Prompts | `src/tripletex_agent/prompts.py` | System prompts, few-shot examples |
| Tripletex tools | `src/tripletex_agent/tripletex.py` | Bounded API wrapper with error handling |
| API spec search | `src/tripletex_agent/spec_index.py` | Just-in-time endpoint discovery |
| File processing | `src/tripletex_agent/files.py` | PDF, image, text attachment handling |
| Trace inspection | `runs/*.jsonl` | Tool calls, reasoning, errors per run |
| Dashboard | `dashboard/` | Real-time monitoring, batch runner, scores |
| Scoring rules | `docs/scoring.md` | Field-by-field checks, 0.0-6.0 per task |
| Endpoint spec | `docs/endpoint-spec.md` | POST /solve request/response contract |

## ARCHITECTURE

```
Request → FastAPI /solve
  → Parse prompt + attachments (files.py)
  → Planner (OpenRouter) → execution plan
  → Executor loop:
      → Tool selection (spec_index.py lookup)
      → Tripletex API call (tripletex.py)
      → Result validation
      → Next step or complete
  → Trace written to runs/
  → Response: { "status": "completed" }
```

## SCORING

- 30 task types × 56 variants (7 languages × 8 datasets)
- Field-by-field correctness checks
- Efficiency bonus for fewer API calls
- Score range: 0.0-6.0 per task
- 5 minute timeout per submission

## ANTI-PATTERNS

- **NEVER** invent data — only use what's in prompt/attachments
- **ALWAYS** set BOTH amountGross AND amountGrossCurrency
- **DO NOT** include postings with row=0
- **DO NOT** re-create entities already created successfully
- **ALWAYS** verify final state after mutations
- **DO NOT** hardcode Tripletex URLs — use provided base_url
- See `executor_knowledge.py` for exhaustive rules

## SECRETS (from .env.example)

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENROUTER_API_KEY` | Yes | AI model access |
| `APP_API_KEY` | No | Endpoint bearer token protection |
| `OPENROUTER_MODEL` | No | Model override |
| `AGENT_MAX_STEPS` | No | Loop depth limit (≥28 recommended) |

## COMMANDS

```bash
# Run locally
uvicorn tripletex_agent.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
uv run pytest tests/

# Validate
python3 -m compileall src tests

# Docker build + deploy
docker build -t tripletex-agent .
docker run -p 8000:8000 --env-file .env tripletex-agent

# Dashboard
cd dashboard && npm run dev
```
