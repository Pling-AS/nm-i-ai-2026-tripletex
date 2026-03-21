#!/usr/bin/env python3
"""Competition simulator: model final standings and optimal time allocation.

Simulates remaining rounds of Astar Island and estimates total competition
score across all 3 tasks under different improvement scenarios.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class TaskScore:
    name: str
    current_score: float
    max_possible: float
    improvement_per_hour: float
    hours_invested: float


COMPETITION_END_HOUR = 12.5  # hours remaining

# Astar Island scoring: MAX(round_score × 1.05^round_number)
# Current best weighted: 144.9 (R10: 88.96 × 1.6289)
# Rounds remaining: ~4-5 more (R11-R15), each ~3h apart

ASTAR_SCORES = {
    "R3": (88.24, 1.1576),
    "R4": (90.53, 1.2155),
    "R6": (86.90, 1.3401),
    "R7": (66.38, 1.4071),
    "R8": (93.90, 1.4775),
    "R9": (92.90, 1.5514),
    "R10": (88.96, 1.6289),
}


def simulate_astar_rounds(
    base_easy_score: float = 93.0,
    base_moderate_score: float = 89.0,
    base_hard_score: float = 67.0,
    improvement: float = 0.0,
    n_remaining: int = 5,
    start_round: int = 11,
):
    """Simulate remaining Astar rounds with stochastic map difficulty."""
    np.random.seed(42)

    best_weighted = max(s * w for s, w in ASTAR_SCORES.values())

    map_types = ["easy", "moderate", "hard"]
    map_probs = [0.45, 0.35, 0.20]
    base_scores = {
        "easy": base_easy_score + improvement,
        "moderate": base_moderate_score + improvement,
        "hard": base_hard_score + improvement,
    }
    score_noise = {"easy": 1.5, "moderate": 2.0, "hard": 4.0}

    n_simulations = 1000
    final_scores = []

    for _ in range(n_simulations):
        sim_best = best_weighted
        for r in range(n_remaining):
            round_num = start_round + r
            weight = 1.05**round_num
            map_type = np.random.choice(map_types, p=map_probs)
            raw = base_scores[map_type] + np.random.normal(0, score_noise[map_type])
            raw = np.clip(raw, 0, 100)
            weighted = raw * weight
            sim_best = max(sim_best, weighted)
        final_scores.append(sim_best)

    return np.array(final_scores)


def estimate_task_scores():
    """Estimate current and potential scores for all 3 tasks."""
    print("=" * 70)
    print("COMPETITION SIMULATION — NM i AI 2026")
    print(f"Time remaining: ~{COMPETITION_END_HOUR:.1f} hours")
    print("=" * 70)

    print("\n## TASK 1 — NorgesGruppen (Object Detection)")
    print("Status: SUBMITTED (models trained, fusion pipeline ready)")
    print("Current estimated score: UNKNOWN (need to check submission portal)")
    print("Improvement potential: Re-train with augmentation, tune thresholds")
    print("Time needed: 2-4h for meaningful improvement")

    print("\n## TASK 2 — Tripletex (AI Accounting Agent)")
    print("Status: DEPLOYED (endpoint running, some tasks passing)")
    print("Current estimated score: UNKNOWN (need to check submission portal)")
    print("Improvement potential: Fix failing task types, add language support")
    print("Time needed: 2-6h depending on failure modes")

    print("\n## TASK 3 — Astar Island")
    print("Status: AUTORUN (proven config, rank #11)")

    scenarios = [
        ("No improvement (current)", 0.0),
        ("+1 point improvement", 1.0),
        ("+2 point improvement", 2.0),
        ("+3 point improvement", 3.0),
    ]

    print("\n### Astar Remaining Rounds Simulation (1000 trials each)")
    print(f"{'Scenario':<30} {'Median':>8} {'P25':>8} {'P75':>8} {'P95':>8}")
    print("-" * 65)

    for name, improvement in scenarios:
        scores = simulate_astar_rounds(improvement=improvement)
        p25, p50, p75, p95 = np.percentile(scores, [25, 50, 75, 95])
        print(f"{name:<30} {p50:8.1f} {p25:8.1f} {p75:8.1f} {p95:8.1f}")

    print("\n### What does this mean?")
    current_best = max(s * w for s, w in ASTAR_SCORES.values())
    print(f"Current best weighted: {current_best:.1f}")

    for round_num in [11, 12, 13, 14, 15]:
        weight = 1.05**round_num
        needed_raw = current_best / weight
        print(
            f"  R{round_num}: need raw ≥{needed_raw:.1f} to beat current best "
            f"(weight={weight:.4f})"
        )

    print("\n### Time Allocation Analysis")
    print(
        "Each task is 33% of total score. Marginal returns depend on current position:"
    )
    print()
    print("  ASTAR (Task 3):")
    print("    - Currently rank #11, near architecture ceiling")
    print("    - Autorun generates free points with zero effort")
    print(
        "    - Additional effort: diminishing returns (6 approaches tried, all failed)"
    )
    print("    - ROI: LOW (already optimized)")
    print()
    print("  NORGESGRUPPEN (Task 1):")
    print("    - Has trained models, submission ready")
    print("    - Unknown current score — CHECK PORTAL FIRST")
    print("    - If score is low: re-training could yield large gains")
    print("    - ROI: POTENTIALLY HIGH (if current score is poor)")
    print()
    print("  TRIPLETEX (Task 2):")
    print("    - Has endpoint, some tasks working")
    print("    - Unknown current score — CHECK PORTAL FIRST")
    print("    - Fixing failing tasks = direct score improvement")
    print("    - ROI: POTENTIALLY HIGH (each fixed task = +points)")
    print()
    print("### RECOMMENDATION")
    print("1. CHECK SCORES on app.ainm.no for Task 1 and Task 2 IMMEDIATELY")
    print(
        "2. Identify which task has the lowest score (biggest improvement opportunity)"
    )
    print("3. Focus remaining hours on that task")
    print("4. Let Astar autorun handle itself (it's free)")
    print()
    print(
        "The competition is 33/33/33 split. If we're rank #11 in Astar but rank #50+ "
        "in the other tasks, fixing those tasks has 10× more impact on overall placement."
    )


if __name__ == "__main__":
    estimate_task_scores()
