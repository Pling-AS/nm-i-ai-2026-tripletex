# PROJECT KNOWLEDGE BASE

**Competition:** NM i AI 2026 (Norwegian Championship in AI)
**Dates:** March 19 18:00 CET — March 22 15:00 CET (69 hours)
**Prize pool:** 1,000,000 NOK
**Platform:** https://app.ainm.no

## MCP DOCUMENTATION SERVER

A shared MCP server provides **all competition documentation** for AI-assisted development:

```
URL: https://mcp-docs.ainm.no/mcp
Transport: HTTP
```

**OpenCode** — configured in root `opencode.json` as `nmaimcp`:
```json
{ "mcp": { "nmaimcp": { "type": "remote", "url": "https://mcp-docs.ainm.no/mcp" } } }
```

**Claude Code** — add manually:
```bash
claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp
```

### What the MCP serves

The MCP exposes the full docs tree at `app.ainm.no/docs`:

| Section | Topics |
|---------|--------|
| **Getting Started** | Competition overview, team setup, Vipps verification |
| **Google Cloud** | Overview, Account Setup, Deploy on Cloud Run, Services & Tools |
| **NorgesGruppen Data** | Overview, Submission format, Scoring, Examples |
| **Tripletex** | Overview, Sandbox Account, Endpoint Spec, Scoring, Examples |
| **Astar Island** | Overview, Simulation Mechanics, API Endpoints, Scoring, Quickstart |

Use this MCP to look up any task's rules, API specs, scoring details, or examples without leaving your editor.

## REPO STRUCTURE

```
nm-i-ai/
├── opencode.json                  # Root MCP config (shared docs server)
├── getting-started.md             # Competition overview (local copy)
├── google-cloud-docs/             # GCP deployment guides (local copies)
├── groccery-practice-task/        # Warm-up: grocery store bot (WebSocket)
│   └── AGENTS.md                  #   Detailed knowledge base
├── task-1-norgesgruppen/          # Object detection on grocery shelves
│   └── AGENTS.md                  #   Pipeline, training, submission details
├── task-2-tripletex/              # AI accounting agent
│   └── AGENTS.md                  #   Agent architecture, API, deployment
└── task-3-astar_island/           # Norse civilisation prediction
    └── AGENTS.md                  #   Model architecture, score history, GCP autorun
```

## THE THREE TASKS (33% each)

### Task 1 — NorgesGruppen Data (Object Detection)

| Aspect | Detail |
|--------|--------|
| Type | Grocery shelf product detection |
| Submission | ZIP upload (code + model weights) |
| Execution | Sandboxed Docker, NVIDIA L4 GPU, no network |
| Data | 248 images, ~22,700 COCO annotations, 356 categories |
| Scoring | 70% detection (mAP) + 30% classification |
| Details | `task-1-norgesgruppen/AGENTS.md` |

### Task 2 — Tripletex (AI Accounting Agent)

| Aspect | Detail |
|--------|--------|
| Type | HTTPS endpoint that completes accounting tasks |
| Submission | Submit endpoint URL at app.ainm.no |
| Tasks | 30 task types, 56 variants each (7 languages × 8 datasets) |
| Timeout | 5 minutes per submission |
| Scoring | Field-by-field checks + efficiency bonus, 0.0-6.0 per task |
| Details | `task-2-tripletex/AGENTS.md` |

### Task 3 — Astar Island (Norse World Prediction)

| Aspect | Detail |
|--------|--------|
| Type | Predict terrain probability distributions |
| API | REST at `https://api.ainm.no/astar-island/` |
| Map | 40×40 grid, 6 prediction classes |
| Budget | 50 viewport queries per round (max 15×15 each), 5 seeds |
| Scoring | Entropy-weighted KL divergence, score = 100 × exp(-3 × wKL) |
| Status | Running autonomously on GCP, best 94.1 (R8), rank #11 |
| Details | `task-3-astar_island/AGENTS.md` |

## CONVENTIONS

- **Package manager:** `uv` (all tasks)
- **Python:** 3.11+ (tasks 1-3), 3.13 (grocery bot)
- **Auth:** JWT tokens from app.ainm.no (cookie or Bearer header)
- **Secrets:** `.env` files per task — never commit
- **No shared code** between tasks — each is fully isolated
- **No CI/CD** — custom scripts per task

## COMMANDS

```bash
# Astar Island
cd task-3-astar_island && uv run run.py

# Grocery Bot (warm-up)
cd groccery-practice-task && uv run main.py easy

# NorgesGruppen (task 1) — see task-1-norgesgruppen/AGENTS.md
cd task-1-norgesgruppen

# Tripletex (task 2)
cd task-2-tripletex
uvicorn tripletex_agent.main:app --reload --host 0.0.0.0 --port 8000
```
