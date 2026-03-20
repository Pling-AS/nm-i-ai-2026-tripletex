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
│   ├── google-cloud.md
│   ├── account-setup.md
│   ├── deploy-on-cloud-run.md
│   └── services-and-tools.md
├── groccery-practice-task/        # Warm-up: grocery store bot (WebSocket)
│   ├── main.py                    #   Bot entry point
│   ├── bot/                       #   Core bot logic (planner, pathfinding, traffic)
│   ├── docs/                      #   Game API spec, getting started
│   ├── AGENTS.md                  #   Detailed project knowledge base
│   └── opencode.json              #   Task-level MCP config
├── task-1-norgesgruppen/          # Object detection on grocery shelves
│   ├── docs/                      #   Overview, submission, scoring, examples
│   ├── submission/                #   Submission code (run.py + model)
│   ├── training_data/             #   COCO dataset
│   ├── tools/                     #   Helper scripts
│   └── work/                      #   YOLO training outputs, predictions
├── task-2-tripletex/              # AI accounting agent
│   └── docs/                      #   Overview, sandbox, endpoint spec, scoring, examples
└── task-3-astar_island/           # Norse civilisation prediction
    ├── docs/                      #   Overview, mechanics, API, scoring, quickstart
    ├── client.py                  #   REST API client
    ├── run.py                     #   Main pipeline (query -> predict -> submit)
    ├── predictor.py               #   Dirichlet posterior predictor
    ├── query_strategy.py          #   Entropy-aware viewport tiling
    ├── observation_store.py       #   Observation accumulator (save/load)
    ├── features.py                #   Cell archetypes, priority masks
    ├── analyze.py                 #   Post-round analysis, calibration generation
    └── calibration.json           #   Calibrated priors from completed rounds
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
| Local docs | `task-1-norgesgruppen/docs/` |

### Task 2 — Tripletex (AI Accounting Agent)

| Aspect | Detail |
|--------|--------|
| Type | HTTPS endpoint that completes accounting tasks |
| Submission | Submit endpoint URL at app.ainm.no |
| Tasks | 30 task types, 56 variants each (7 languages x 8 datasets) |
| Timeout | 5 minutes per submission |
| Scoring | Field-by-field checks + efficiency bonus, 0.0-6.0 per task |
| Local docs | `task-2-tripletex/docs/` |

### Task 3 — Astar Island (Norse World Prediction)

| Aspect | Detail |
|--------|--------|
| Type | Predict terrain probability distributions |
| API | REST at `https://api.ainm.no/astar-island/` |
| Map | 40x40 grid, 6 prediction classes |
| Budget | 50 viewport queries per round (max 15x15 each), 5 seeds |
| Prediction | H x W x 6 probability tensor per seed |
| Scoring | Entropy-weighted KL divergence, score = 100 * exp(-3 * wKL) |
| Local docs | `task-3-astar_island/docs/` |

## CONVENTIONS

- **Package manager:** `uv` (all tasks)
- **Python:** 3.11+ (task-3), 3.13 (grocery bot)
- **Auth:** JWT tokens from app.ainm.no (cookie or Bearer header)
- **Secrets:** `.env` files per task — never commit

## COMMANDS

```bash
# Astar Island
cd task-3-astar_island
uv run run.py                      # Full pipeline: query + predict + submit
uv run run.py --predict-only       # Re-predict from saved observations
uv run analyze.py                  # Post-round analysis + calibration
uv run analyze.py --all            # Analyze all completed rounds

# Grocery Bot (warm-up)
cd groccery-practice-task
uv run main.py easy                # Run bot on easy difficulty

# NorgesGruppen (task 1)
cd task-1-norgesgruppen
# See docs/submission.md for ZIP upload format
```
