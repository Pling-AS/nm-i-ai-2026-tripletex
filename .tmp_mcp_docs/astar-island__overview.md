# Astar Island â Viking Civilisation Prediction

## What is this?

Astar Island is a machine learning challenge where you observe a black-box Norse civilisation simulator through a limited viewport and predict the final world state. The simulator runs a procedurally generated Norse world for 50 years â settlements grow, factions clash, trade routes form, alliances shift, forests reclaim ruins, and harsh winters reshape entire civilisations.

Your goal: **observe, learn the world's hidden rules, and predict the probability distribution of terrain types across the entire map.**

- **Task type**: Observation + probabilistic prediction
- **Platform**: [app.ainm.no](https://app.ainm.no)
- **API**: REST endpoints at `api.ainm.no/astar-island/`

![Initial Astar Island Map](/docs/astar-island/map-initial.png)

## How It Works

1. **A round starts** â the admin creates a round with a fixed map, many hidden parameters, and 5 random seeds
2. **Observe through a viewport** â call `POST /astar-island/simulate` with viewport coordinates to observe one stochastic run through a window (max 15Ã15 cells). You have 50 queries total per round, shared across all 5 seeds.
3. **Learn the hidden rules** â analyze viewport observations to understand the forces that govern the world
4. **Generate predictions** â use your understanding to build probability distributions for the full map
5. **Submit predictions** â for each of the 5 seeds, submit a WÃHÃ6 probability tensor predicting terrain type probabilities per cell
6. **Scoring** â your prediction is compared against the ground truth using entropy-weighted KL divergence

## The Core Challenge

The simulation is **stochastic** â the same map and parameters produce different outcomes every run. With only **50 queries** shared across **5 seeds**, and each query only revealing a **15Ã15 viewport** of the 40Ã40 map, you must be strategic about what you observe and how you use that information.

![Same map, different stochastic outcomes](/docs/astar-island/stochastic-outcomes.png)

![Viewport: you only see a small window of the full map](/docs/astar-island/viewport-demo.png)

The world is governed by many hidden forces that interact in complex ways. Teams that understand these interactions can build accurate models and generate predictions far beyond what raw observation provides.

## Quick Start

1. Sign in at [app.ainm.no](https://app.ainm.no) with Google
2. Create or join a team
3. Go to the Astar Island page
4. When a round is active, use the API to observe the simulator
5. Analyze results, build your model, submit predictions for all 5 seeds

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Map seed** | Determines terrain layout (fixed per seed, visible to you) |
| **Sim seed** | Random seed for each simulation run (different every query) |
| **Hidden parameters** | Values controlling the world's behavior (same for all seeds in a round) |
| **50 queries** | Your budget per round, shared across all 5 seeds |
| **Viewport** | Each query reveals a max 15Ã15 window of the map |
| **WÃHÃ6 tensor** | Your prediction â probability of each of 6 terrain classes per cell |
| **50 years** | Each simulation runs for 50 time steps |
