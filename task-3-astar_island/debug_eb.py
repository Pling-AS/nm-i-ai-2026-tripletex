#!/usr/bin/env python3
"""Debug: print per-archetype EB τ vs JSD τ for R4."""

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from client import AstarClient
from features import SeedAnalysis
from observation_store import ObservationStore
import predictor as pred_module
from predictor import (
    _log_marginal_likelihood,
    _get_calibrated_prior,
    _initial_terrain_prior,
    _TAU_GRID,
    _jsd,
    TAU_MIN,
    TAU_MAX,
)
from utils import is_static, NUM_CLASSES

c = AstarClient()
rid = "8e839974-b13b-407b-a5e7-fc749d877195"
detail = c.get_round_detail(rid)
obs_path = f"data/obs_{rid.split('-')[0]}"
store = ObservationStore.load(obs_path)
all_sa = [SeedAnalysis(s["grid"], s["settlements"]) for s in detail.initial_states]

arch_cells = {}
for si, sa in enumerate(all_sa):
    counts_grid = store.get_seed_counts(si)
    obs_grid = store.get_seed_obs_counts(si)
    for y in range(sa.height):
        for x in range(sa.width):
            if obs_grid[y, x] == 0:
                continue
            terrain = int(sa.grid[y, x])
            if is_static(terrain):
                continue
            arch = sa.get_archetype(y, x)
            if arch not in arch_cells:
                arch_cells[arch] = []
            arch_cells[arch].append(counts_grid[y, x].copy())

print(f"{'Archetype':<60} {'Cells':>5} {'EB_τ':>5} {'JSD_τ':>6} {'EB_ll':>10}")
print("-" * 90)

for arch, cells in sorted(arch_cells.items(), key=lambda x: -len(x[1])):
    if len(cells) < 5:
        continue
    terrain = arch.initial_terrain
    prior = _get_calibrated_prior(arch, terrain)
    if prior is None:
        prior = _initial_terrain_prior(terrain)

    best_tau = 25.0
    best_ll = -np.inf
    all_ll = {}
    for tau_c in _TAU_GRID:
        ll = _log_marginal_likelihood(cells, prior, tau_c)
        all_ll[tau_c] = ll
        if ll > best_ll:
            best_ll = ll
            best_tau = float(tau_c)

    obs_counts = store._archetype_counts[arch]
    empirical = obs_counts.astype(np.float64)
    emp_sum = empirical.sum()
    if emp_sum > 0:
        empirical = empirical / emp_sum
    jsd = _jsd(empirical, prior)
    hardness = min(1.0, jsd / 0.20)
    jsd_tau = TAU_MAX - hardness * (TAU_MAX - TAU_MIN)

    avg_n = np.mean([c.sum() for c in cells])
    print(
        f"{str(arch):<60} {len(cells):>5} {best_tau:>5.0f} {jsd_tau:>6.1f} {best_ll:>10.1f}  avg_n={avg_n:.1f}"
    )
