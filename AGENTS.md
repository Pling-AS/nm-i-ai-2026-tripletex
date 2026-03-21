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
    ├── solution_spatial.py        #   Spatial Bayesian Diffusion predictor
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

**STATUS: RUNNING AUTONOMOUSLY | Best: 94.1 (R8) | Rank: #11 | Autorun on GCP**

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
uv run simulate_round.py           # Verify solution against Round 2 ground truth
uv run analyze.py                  # Post-round analysis + calibration
uv run analyze.py --all            # Analyze all completed rounds

# Grocery Bot (warm-up)
cd groccery-practice-task
uv run main.py easy                # Run bot on easy difficulty

# NorgesGruppen (task 1)
cd task-1-norgesgruppen
# See docs/submission.md for ZIP upload format
```

## ASTAR ISLAND — DEEP KNOWLEDGE (Task 3)

### Architecture
- **Dirichlet posterior**: p_k = (n_k + τ·m_k) / (N + τ)
- **Adaptive τ per archetype**: JSD-based, range [12, 40], round-level τ is weighted median
- **SHRINKAGE_KAPPA = 10.0** for prior blending (λ = arch_n/(arch_n + κ))
- **Same-terrain neighbor pseudo-counts**: λ=0.5, cap=2.0 — borrows neighbor observations
- **Terrain-aware spatial smoothing**: β=0.15, always-on, only same-terrain neighbors
- **Coverage**: 3×3 tiling (9 viewports × 5 seeds = 45 queries) + 5 VOI repeats
- **Interleaved queries** across seeds so partial budget gives partial coverage of ALL seeds
- **Leave-one-out**: Target cell's counts subtracted from archetype pool before building prior
- **Archetype backoff**: Full 4-tuple → drop coastal → drop dist → terrain only
- **Floor**: ε=0.001 additive floor, then renormalize

### Score History
| Round | Raw Score | Rank | Weighted (×1.05^r) | Map Type |
|-------|-----------|------|---------------------|----------|
| R1 | missed | — | 0 | — |
| R2 | 74.40 | #41/153 | 82.0 | — |
| R3 | 88.24 | #2/100 | 102.2 | moderate |
| R4 | 90.53 | #10/86 | 110.0 | easy |
| R5 | missed | — | 0 | — |
| R6 | 86.90 | #4/186 | 116.5 | moderate |
| R7 | 66.38 | #26/199 | 93.4 | hard |
| R8 | 93.90 | #5/214 | 138.8 | easy |
| R9 | 92.90 | #9/221 | 144.1 | easy |
| R10 | 88.96 | ?/? | 144.9 | moderate |

### Error Profiles (vary per round — no single fix works)
| Round | Empty | Settlement | Port | Ruin | Forest | Dominant |
|-------|-------|------------|------|------|--------|----------|
| R3 | 12% | 8% | -1% | 6% | 75% | Forest |
| R4 | 36% | 23% | 12% | 15% | 14% | Empty |
| R6 | 16% | 24% | 16% | 16% | 29% | Forest |
| R7 | -13% | 78% | 15% | 9% | 10% | Settlement |
| R8 | 17% | 48% | 5% | 24% | 5% | Settlement |
| R9 | 47% | -12% | 12% | 9% | 43% | Empty |

### CRITICAL FINDINGS
1. **Predictions are 97% prior-dominated** — with τ≈36 and N≈1.5, observations get only 2.7% weight
2. **Shrinkage blend already adapts priors perfectly** — λ=0.997 on R8 means prior is 99.7% from current round data
3. **ALL cells are observed** (0 unobserved) — error is 100% from model inaccuracy, not missing data
4. **Within-archetype variance is the bottleneck** — cells in same archetype have different true distributions
5. **Leaderboard uses MAX scoring** — best(round_score × 1.05^round_number)
6. **Coastal-wrap bug was false alarm** — all maps are islands, edges ALL ocean
7. **Settlement dynamics vary 100× across rounds** — d=2 settlement rate: 0.001 (R3) to 0.266 (R6)

### WHAT WAS TRIED AND FAILED (DO NOT RETRY)
| Approach | Result | Why it failed |
|----------|--------|---------------|
| Spatial residual field | Neutral | N=1 observation noise overwhelms spatial signal |
| Global class tilt | -1 to -3 pts | Over-corrects well-calibrated archetype priors |
| Surprise-based τ | +1.2 hard / -0.3 easy | Net negative under MAX scoring |
| Empirical Bayes τ (MML) | -0.7 to +1.6 | Degenerate with N≈1.5 observations |
| Terrain-level hedge | Zero effect | Redundant with Dirichlet prior mass |
| Fixed low TAU_OBS | -1 to -6 pts easy | Amplifies noise on easy maps |

### WHAT WORKS (currently deployed)
| Feature | Impact on backtests |
|---------|---------------------|
| Same-terrain neighbor pseudo-counts (λ=0.5) | R4 +0.07, R6 +0.30, R7 +1.14 |
| Terrain-aware spatial smooth (β=0.15) | R4 +0.04, R6 +0.03, R7 +0.06 |

### GCP VM
- Name: `astar-autorun`, zone: `europe-north1-b`, type: `e2-small`
- Python 3.11 via uv, tmux session: `autorun`
- Code: `~/astar/`, log: `~/autorun.log`
- gcloud: `/opt/homebrew/share/google-cloud-sdk/bin/gcloud`
- Project: `ainm26osl-753`

### Autorun Controls
| Action | Command |
|--------|---------|
| Pause (before next round) | `gcloud compute ssh astar-autorun --zone=europe-north1-b --command="touch ~/astar/PAUSE"` |
| Resume | `gcloud compute ssh astar-autorun --zone=europe-north1-b --command="rm ~/astar/PAUSE"` |
| Check log | `gcloud compute ssh astar-autorun --zone=europe-north1-b --command="tail -20 ~/autorun.log"` |
| Deploy file | `gcloud compute scp --zone=europe-north1-b <local> astar-autorun:~/astar/<remote>` |
| Restart autorun | `gcloud compute ssh astar-autorun --zone=europe-north1-b --command="tmux kill-session -t autorun; cd ~/astar && tmux new-session -d -s autorun 'export PATH=\"/home/m/.local/bin:\$PATH\" && uv run autorun.py 2>&1 \| tee ~/autorun.log'"` |

### Spatial Residual Field Design (SAVED, NOT DEPLOYED)
Full design saved in `.sisyphus/plans/spatial-residual-field.md` — tested and found neutral.

### Remaining Ideas (untested, lower confidence)
1. **Query strategy optimization** — variable viewport sizes for repeats
2. **Use settlement stats from simulate API** — population, food, wealth, defense, owner_id
3. **Finer archetypes** — add dist_to_ruin, direction features
4. **Per-cell logistic regression** from calibration data (continuous instead of bucketed)
