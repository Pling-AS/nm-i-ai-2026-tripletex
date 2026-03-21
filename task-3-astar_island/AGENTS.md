# TERRAIN PREDICTION — Astar Island

Predict 6-class probability distributions for Norse civilisation terrain evolution on 40×40 grids. Spatial Bayesian Diffusion model with Dirichlet posteriors, adaptive priors, and entropy-weighted scoring.

**STATUS: RUNNING AUTONOMOUSLY | Best: 94.1 (R8) | Rank: #11 | Autorun on GCP**

## STRUCTURE

```
task-3-astar_island/
├── run.py                # Main pipeline: setup → query → predict → submit
├── client.py             # REST API client (AstarClient)
├── predictor.py          # Active predictor (Dirichlet posterior, calibration)
├── solution_spatial.py   # Legacy predictor (spatial variant, not imported by run.py)
├── query_strategy.py     # Viewport planning (coverage + VOI repeats)
├── observation_store.py  # Per-cell observation counts + archetype pools
├── features.py           # SeedAnalysis: archetypes, masks, distances
├── utils.py              # Constants, terrain mappings, floor normalization
├── analyze.py            # Post-round KL analysis → calibration.json
├── autorun.py            # GCP autonomous round handler
├── simulate_round.py     # Backtest against R2 ground truth
├── calibration.json      # Calibrated priors from completed rounds
├── data/                 # Saved observations (obs_*.npz + .json)
└── docs/                 # Task specification
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Full pipeline | `run.py` | Phases: setup, query, predict, submit |
| API calls | `client.py` | simulate(), submit(), get_rounds() |
| Prediction math | `predictor.py` | Dirichlet posterior + calibration + spatial |
| Viewport planning | `query_strategy.py` | 3×3 tiling + entropy-weighted VOI repeats |
| Feature extraction | `features.py` | CellArchetype, SeedAnalysis, distance/coastal masks |
| Observation storage | `observation_store.py` | Save/load per-round, archetype pool aggregation |
| Constants | `utils.py` | TERRAIN_*, CLASS_*, floor/normalize |
| Calibration | `analyze.py` | Post-round: generate_calibrated_priors() |
| Autonomous ops | `autorun.py` | Monitors rounds, runs pipeline, resubmits |
| Backtesting | `simulate_round.py` | End-to-end validation against known R2 truth |

## DATA FLOW

```
run.py
  → client.get_round_detail()              # Fetch round metadata
  → features.SeedAnalysis()                 # Precompute terrain, archetypes, distances
  → query_strategy.plan_coverage_queries()  # 9 viewports × 5 seeds (interleaved)
  → client.simulate()                       # Execute viewport queries (50 budget)
  → observation_store.add()                 # Accumulate stochastic observation counts
  → query_strategy.plan_repeat_queries()    # VOI-based repeats (remaining budget)
  → predictor.predict_full_grid()           # Dirichlet posterior → H×W×6 tensor
  → client.submit()                         # Submit predictions per seed
```

## ARCHITECTURE

- **Dirichlet posterior**: p_k = (n_k + τ·m_k) / (N + τ)
- **Adaptive τ per archetype**: JSD-based, range [12, 40], round-level τ is weighted median
- **SHRINKAGE_KAPPA = 10.0**: Prior blend λ = arch_n/(arch_n + κ)
- **Same-terrain neighbor pseudo-counts**: λ=0.5, cap=2.0
- **Terrain-aware spatial smoothing**: β=0.15, always-on, same-terrain neighbors only
- **Coverage**: 3×3 tiling (9 viewports × 5 seeds = 45 queries) + 5 VOI repeats
- **Interleaved queries**: Partial budget gives partial coverage of ALL seeds
- **Leave-one-out**: Target cell counts subtracted from archetype pool
- **Archetype backoff**: Full 4-tuple → drop coastal → drop dist → terrain only
- **Floor**: ε=0.001 additive, then renormalize

## SCORE HISTORY

| Round | Score | Rank | Weighted (×1.05^r) | Map Type |
|-------|-------|------|---------------------|----------|
| R2 | 74.40 | #41/153 | 82.0 | — |
| R3 | 88.24 | #2/100 | 102.2 | moderate |
| R4 | 90.53 | #10/86 | 110.0 | easy |
| R6 | 86.90 | #4/186 | 116.5 | moderate |
| R7 | 66.38 | #26/199 | 93.4 | hard |
| R8 | 93.90 | #5/214 | 138.8 | easy |
| R9 | 92.90 | #9/221 | 144.1 | easy |
| R10 | 88.96 | ?/? | 144.9 | moderate |

## ERROR PROFILES (vary per round — no single fix works)

| Round | Empty | Settlement | Port | Ruin | Forest | Dominant |
|-------|-------|------------|------|------|--------|----------|
| R3 | 12% | 8% | -1% | 6% | 75% | Forest |
| R4 | 36% | 23% | 12% | 15% | 14% | Empty |
| R6 | 16% | 24% | 16% | 16% | 29% | Forest |
| R7 | -13% | 78% | 15% | 9% | 10% | Settlement |
| R8 | 17% | 48% | 5% | 24% | 5% | Settlement |
| R9 | 47% | -12% | 12% | 9% | 43% | Empty |

## CRITICAL FINDINGS

1. **97% prior-dominated** — τ≈36, N≈1.5 → observations get 2.7% weight
2. **Shrinkage already adapts** — λ=0.997 on R8 → prior 99.7% from current round
3. **ALL cells observed** (0 unobserved) — error is model inaccuracy, not missing data
4. **Within-archetype variance is the bottleneck** — same archetype, different distributions
5. **MAX scoring** — leaderboard uses best(round_score × 1.05^round_number)
6. **Settlement dynamics vary 100×** — d=2 rate: 0.001 (R3) to 0.266 (R6)

## WHAT FAILED (DO NOT RETRY)

| Approach | Result | Why |
|----------|--------|-----|
| Spatial residual field | Neutral | N=1 noise overwhelms signal |
| Global class tilt | -1 to -3 pts | Over-corrects calibrated priors |
| Surprise-based τ | +1.2 hard / -0.3 easy | Net negative under MAX scoring |
| Empirical Bayes τ (MML) | -0.7 to +1.6 | Degenerate with N≈1.5 |
| Terrain-level hedge | Zero effect | Redundant with Dirichlet mass |
| Fixed low TAU_OBS | -1 to -6 pts easy | Amplifies noise on easy maps |

## WHAT WORKS (deployed)

| Feature | Backtest Impact |
|---------|-----------------|
| Same-terrain neighbor pseudo-counts (λ=0.5) | R4 +0.07, R6 +0.30, R7 +1.14 |
| Terrain-aware spatial smooth (β=0.15) | R4 +0.04, R6 +0.03, R7 +0.06 |

## GCP VM

- Instance: `astar-autorun`, zone: `europe-north1-b`, type: `e2-small`
- Python 3.11 via uv, tmux session: `autorun`
- Code: `~/astar/`, log: `~/autorun.log`
- gcloud: `/opt/homebrew/share/google-cloud-sdk/bin/gcloud`
- Project: `ainm26osl-753`

| Action | Command |
|--------|---------|
| Pause | `gcloud compute ssh astar-autorun --zone=europe-north1-b --command="touch ~/astar/PAUSE"` |
| Resume | `gcloud compute ssh astar-autorun --zone=europe-north1-b --command="rm ~/astar/PAUSE"` |
| Check log | `gcloud compute ssh astar-autorun --zone=europe-north1-b --command="tail -20 ~/autorun.log"` |
| Deploy file | `gcloud compute scp --zone=europe-north1-b <local> astar-autorun:~/astar/<remote>` |
| Restart | `gcloud compute ssh astar-autorun --zone=europe-north1-b --command="tmux kill-session -t autorun; cd ~/astar && tmux new-session -d -s autorun 'export PATH=\"/home/m/.local/bin:\$PATH\" && uv run autorun.py 2>&1 \| tee ~/autorun.log'"` |

## REMAINING IDEAS (untested)

1. Variable viewport sizes for VOI repeats
2. Use settlement stats from simulate API (population, food, wealth, defense)
3. Finer archetypes: add dist_to_ruin, direction features
4. Per-cell logistic regression from calibration data

## COMMANDS

```bash
uv run run.py                      # Full pipeline: query + predict + submit
uv run run.py --predict-only       # Re-predict from saved observations
uv run simulate_round.py           # Backtest against R2 ground truth
uv run analyze.py                  # Post-round analysis + calibration
uv run analyze.py --all            # Analyze all completed rounds
```
