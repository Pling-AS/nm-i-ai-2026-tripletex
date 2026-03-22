import numpy as np
from features import SeedAnalysis
from query_strategy import _compute_prior_entropy_map, QueryPlan

def plan_multi_sample_queries(
    seeds_count: int,
    map_w: int,
    map_h: int,
    max_budget: int,
    seed_analyses: list[SeedAnalysis],
) -> list[QueryPlan]:
    vp = 15
    queries = []
    
    planned_n = np.zeros((seeds_count, map_h, map_w), dtype=np.float64)
    prior_entropy = np.zeros((seeds_count, map_h, map_w), dtype=np.float64)
    
    for s in range(seeds_count):
        prior_entropy[s] = _compute_prior_entropy_map(seed_analyses[s])
        prior_entropy[s] *= seed_analyses[s].priority_mask
        
    for _ in range(max_budget):
        best_score = -1.0
        best_query = None
        
        # Steeper penalty to encourage N=2-3 but not N=10
        # e.g., value / (planned_n + 1)^2
        marginal_value = prior_entropy / ((planned_n + 1.0) ** 2)
        
        for s in range(seeds_count):
            sat = np.zeros((map_h + 1, map_w + 1), dtype=np.float64)
            sat[1:, 1:] = np.cumsum(np.cumsum(marginal_value[s], axis=0), axis=1)
            
            max_x = max(0, map_w - vp)
            max_y = max(0, map_h - vp)
            
            for vy in range(max_y + 1):
                for vx in range(max_x + 1):
                    ey = min(vy + vp, map_h)
                    ex = min(vx + vp, map_w)
                    window_val = sat[ey, ex] - sat[vy, ex] - sat[ey, vx] + sat[vy, vx]
                    
                    if window_val > best_score:
                        best_score = window_val
                        best_query = (s, vx, vy, ex - vx, ey - vy)
                        
        if best_query is None:
            break
            
        s, vx, vy, vw, vh = best_query
        queries.append(QueryPlan(s, vx, vy, vw, vh))
        planned_n[s, vy:vy+vh, vx:vx+vw] += 1.0
        
    return queries
