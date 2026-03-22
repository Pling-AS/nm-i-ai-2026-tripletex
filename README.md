# NM i AI 2026

**Team: Er det kå i?**

Our submission for **NM i AI 2026** (Norwegian Championship in AI) — a 69-hour AI hackathon with a 1,000,000 NOK prize pool.

The competition consisted of three scored tasks plus a practice warm-up, each requiring a different AI approach.

## Repo Structure

```
nm-i-ai/
├── task-1-norgesgruppen/      # Object detection on grocery shelves
├── task-2-tripletex/          # AI accounting agent
├── task-3-astar_island/       # Terrain probability prediction
└── groccery-practice-task/    # Warm-up: grocery warehouse bot
```

Each task is fully isolated with its own dependencies, `.env`, and `AGENTS.md` knowledge base.

## Tech Stack

- **Python 3.11** (tasks 1–3), **Python 3.13** (practice task)
- **uv** — package manager for all tasks
- **Key libraries:** ultralytics (YOLO), FastAPI, numpy, scipy, requests, websockets

---

## Task 1 — NorgesGruppen (Object Detection)

Detect and classify products on grocery store shelves from images.

- **Approach:** YOLO object detection with k-fold cross-validation + DINOv2-based product classifier + Weighted Box Fusion
- **Pipeline:** Dataset prep → k-fold splits → YOLO training → classifier training → hybrid inference
- **Submission:** ZIP with inference code + model weights, executed in sandboxed Docker with NVIDIA L4 GPU

### Setup

```bash
cd task-1-norgesgruppen
uv sync
```

### Training

```bash
bash train.sh
```

### Key Files

| File | Purpose |
|------|---------|
| `tools/prepare_yolo_dataset.py` | Convert COCO annotations to YOLO format |
| `tools/create_kfold_splits.py` | Create 5-fold CV splits |
| `tools/train_fold.py` | Train single YOLO fold |
| `tools/train_cls_refmix.py` | Train product classifier |
| `tools/export_submission.py` | Package submission ZIP |
| `tools/score_hybrid.py` | Local scoring |
| `submission/` | Inference code (run.py, fusion, detection, classification) |

---

## Task 2 — Tripletex (AI Accounting Agent)

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

---

## Task 3 — Astar Island (Terrain Prediction)

Predict 6-class terrain probability distributions on a 40×40 grid for a Norse civilisation simulation.

- **Approach:** Spatial Bayesian Diffusion with Dirichlet posteriors, adaptive priors, entropy-weighted scoring
- **Pipeline:** Fetch round → viewport queries → observation accumulation → Dirichlet prediction → submit
- **Budget:** 50 viewport queries per round, 5 seeds

### Setup

```bash
cd task-3-astar_island
uv sync
cp .env.example .env
# Fill in ACCESS_TOKEN from app.ainm.no
```

### Run

```bash
uv run run.py                     # Full pipeline: query + predict + submit
uv run run.py --predict-only      # Re-predict from saved observations
uv run simulate_round.py          # Backtest against known round data
uv run analyze.py --all           # Post-round calibration analysis
```

### Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `ACCESS_TOKEN` | Yes | JWT token from app.ainm.no |
| `API_BASE` | No | API base URL (default: `https://api.ainm.no`) |

---

## Practice Task — Grocery Bot

WebSocket-based bot controlling warehouse workers to fulfill grocery orders. Used as a warm-up before the main competition.

### Setup

```bash
cd groccery-practice-task
uv sync
cp .env.example .env
# Fill in WS_ENDPOINT_* URLs from app.ainm.no/challenge
```

### Run

```bash
uv run main.py easy    # or: medium, hard, expert, nightmare
```

---

## General Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Clone the repo
3. `cd` into the task you want to run
4. `uv sync` to install dependencies
5. Copy `.env.example` to `.env` and fill in your credentials
6. Follow the task-specific run instructions above

Each task directory contains an `AGENTS.md` with detailed architecture notes, scoring breakdowns, and implementation details.
