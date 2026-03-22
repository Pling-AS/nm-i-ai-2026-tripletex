# NM i AI 2026 — Tripletex (AI Accounting Agent)

**Team: Er det kå i?**

Our Task 2 submission for **NM i AI 2026** (Norwegian Championship in AI) — a 69-hour AI hackathon with a 1,000,000 NOK prize pool.

Task 1 (NorgesGruppen) and Task 3 (Astar Island) are in a [separate repo](https://github.com/Pling-AS/nm-i-ai-2026-norgesgruppen-astar).

---

## Tripletex — AI Accounting Agent

An HTTPS endpoint that receives accounting task prompts and completes them in a Tripletex sandbox environment using API calls.

- **Approach:** LLM planner + tool-calling executor with bounded Tripletex API access
- **Stack:** FastAPI, OpenRouter/Vertex AI, Pydantic
- **Scoring:** 30 task types × 56 variants, field-by-field correctness + efficiency bonus

### Setup

```bash
cd task-2-tripletex
uv sync
cp .env.example .env
# Fill in OPENROUTER_API_KEY at minimum
```

### Run

```bash
uvicorn tripletex_agent.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t tripletex-agent .
docker run -p 8000:8000 --env-file .env tripletex-agent
```

### Environment Variables

See `task-2-tripletex/.env.example` for all variables. Key ones:

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENROUTER_API_KEY` | Yes | LLM API access |
| `VERTEX_AI_PROJECT_ID` | No | GCP Vertex AI (primary backend) |
| `AZURE_API_KEY` | No | Azure fallback |
| `APP_API_KEY` | No | Protect your endpoint with bearer token |

### Architecture

```
Request → FastAPI /solve
  → Parse prompt + attachments
  → Planner (LLM) → execution plan
  → Executor loop:
      → Tool selection (API spec lookup)
      → Tripletex API call
      → Result validation
      → Next step or complete
  → Response: { "status": "completed" }
```

See `task-2-tripletex/AGENTS.md` for detailed architecture notes, scoring breakdowns, and implementation details.
