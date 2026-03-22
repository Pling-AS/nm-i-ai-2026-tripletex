#!/usr/bin/env python3
"""Compute empirical correction for Astar Island predictions."""

import json
from collections import defaultdict
import numpy as np
from client import AstarClient
from features import SeedAnalysis

# Load calibration
with open('calibration.json') as f:
    cal = json.load(f)

priors = cal['archetype_priors']
counts = cal['archetype_counts']

def parse_archetype_key(key):
    # Remove 'CellArchetype(' and ')'
    inner = key[14:-1]
    parts = inner.split(', ')
    d = {}
    for p in parts:
        k, v = p.split('=')
        if v in ('True', 'False'):
            d[k] = v == 'True'
        else:
            d[k] = int(v)
    return d

# Group priors by (terrain, dist_bucket, coastal)
grouped_priors = defaultdict(lambda: np.zeros(6))
grouped_counts = defaultdict(int)

for key, prior in priors.items():
    arch = parse_archetype_key(key)
    terrain = arch['initial_terrain']
    dist_bucket = arch['dist_settlement_bucket']
    coastal = arch['is_coastal']
    n = counts[key]
    grouped_priors[(terrain, dist_bucket, coastal)] += np.array(prior) * n
    grouped_counts[(terrain, dist_bucket, coastal)] += n

for k in grouped_priors:
    grouped_priors[k] /= grouped_counts[k]

# Now compute empirical
client = AstarClient()
my_rounds = client.get_my_rounds()
completed = [r for r in my_rounds if r['status'] == 'completed']

sum_gt = np.zeros((12, 7, 2, 6))
cell_count = np.zeros((12, 7, 2))

for round_info in completed:
    round_id = round_info['id']
    detail = client.get_round_detail(round_id)
    for seed_idx in range(detail.seeds_count):
        data = client.get_analysis(round_id, seed_idx)
        gt = np.array(data['ground_truth'])
        initial_grid = data.get('initial_grid', detail.initial_states[seed_idx]['grid'])
        settlements = detail.initial_states[seed_idx]['settlements']
        seed_analysis = SeedAnalysis(initial_grid, settlements)
        h, w = gt.shape[:2]
        terrain = seed_analysis.grid
        dist = seed_analysis.dist_to_settlement
        bucket = np.minimum(dist.astype(int), 6)
        coastal = seed_analysis.coastal_mask.astype(int)
        for y in range(h):
            for x in range(w):
                t = terrain[y, x]
                b = bucket[y, x]
                c = coastal[y, x]
                sum_gt[t, b, c] += gt[y, x]
                cell_count[t, b, c] += 1

# Now output for terrain 4 and 11
print("terrain  dist  coastal  P(empty)_diff  P(sett)_diff  P(port)_diff  P(ruin)_diff  P(forest)_diff  N_cells")
for terrain in [4, 11]:
    for bucket in range(7):
        for coastal in [0, 1]:
            if cell_count[terrain, bucket, coastal] > 0:
                avg_gt = sum_gt[terrain, bucket, coastal] / cell_count[terrain, bucket, coastal]
                prior = grouped_priors.get((terrain, bucket, bool(coastal)), np.zeros(6))
                diff = avg_gt - prior
                print(f"{terrain}  {bucket}  {bool(coastal)}  {diff[0]:.6f}  {diff[1]:.6f}  {diff[2]:.6f}  {diff[3]:.6f}  {diff[4]:.6f}  {int(cell_count[terrain, bucket, coastal])}")
