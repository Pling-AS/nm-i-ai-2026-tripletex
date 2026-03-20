#!/usr/bin/env python3
"""Backtest empirical Bayes τ + neighbor counts + terrain-aware smooth."""

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from client import AstarClient
from features import SeedAnalysis
from observation_store import ObservationStore
import predictor as pred_module
from predictor import predict_full_grid_vectorized, compute_round_tau


def kl_divergence(p, q):
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(p > 0, p * np.log(p / np.maximum(q, 1e-10)), 0.0)
    return kl.sum(axis=-1)


def entropy(p):
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(p > 0, -p * np.log(p), 0.0)
    return h.sum(axis=-1)


def score_from_kl(gt, pred):
    ent = entropy(gt)
    kl = kl_divergence(gt, pred)
    dynamic = ent > 1e-8
    if not dynamic.any():
        return 100.0
    weighted_kl = (ent[dynamic] * kl[dynamic]).sum() / ent[dynamic].sum()
    return max(0, min(100, 100 * np.exp(-3 * weighted_kl)))


c = AstarClient()

rounds = {
    "R4": "8e839974-b13b-407b-a5e7-fc749d877195",
    "R6": "ae78003a-4efe-425a-881a-d16a39bca0ad",
    "R7": "36e581f1-73f8-453f-ab98-cbe3052b701b",
}

pred_module.NEIGHBOR_LAMBDA = 0.5
pred_module.NEIGHBOR_MAX_TOTAL = 2.0

for rname, rid in rounds.items():
    detail = c.get_round_detail(rid)
    obs_path = f"data/obs_{rid.split('-')[0]}"
    store = ObservationStore.load(obs_path)
    all_sa = [SeedAnalysis(s["grid"], s["settlements"]) for s in detail.initial_states]
    compute_round_tau(all_sa, store)

    gt_all = [np.array(c.get_analysis(rid, si)["ground_truth"]) for si in range(5)]

    seed_scores = []
    for si in range(5):
        sa = all_sa[si]
        pred = predict_full_grid_vectorized(si, store, sa)
        seed_scores.append(score_from_kl(gt_all[si], pred))

    avg = np.mean(seed_scores)
    print(f"{rname}: avg={avg:.2f}")
    for si in range(5):
        print(f"  seed {si}: {seed_scores[si]:.2f}")
