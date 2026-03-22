import re

with open('/Users/m/Programmering/2025/nm-i-ai/task-3-astar_island/predictor.py', 'r') as f:
    content = f.read()

old_code = """    # Pre-compute cross-seed pooled counts (sum of other seeds at same position)
    cross_seed_grid = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)
    if CROSS_SEED_LAMBDA > 0:
        for other_seed in range(observation_store.seeds_count):
            if other_seed != seed_index:
                cross_seed_grid += observation_store.get_seed_counts(other_seed).astype(np.float64)"""

new_code = """    # Pre-compute cross-seed pooled counts (sum of other seeds at same position WITH SAME ARCHETYPE)
    cross_seed_grid = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)
    if CROSS_SEED_LAMBDA > 0:
        for other_seed in range(observation_store.seeds_count):
            if other_seed != seed_index:
                # Only pool if the archetype is exactly the same
                # We need to check this per-cell in the loop, so we can't fully precompute it here easily.
                # Actually, we can precompute a mask of matching archetypes!
                pass # We will handle this in the loop"""

content = content.replace(old_code, new_code)

old_loop_code = """            # Cross-seed pooling
            if CROSS_SEED_LAMBDA > 0:
                cs = cross_seed_grid[y, x]
                cs_total = cs.sum()
                if cs_total > 0:
                    cs_weight = CROSS_SEED_LAMBDA / cs_total
                    effective_counts = effective_counts + cs_weight * cs"""

new_loop_code = """            # Cross-seed pooling (only if archetype matches)
            if CROSS_SEED_LAMBDA > 0:
                cs = np.zeros(NUM_CLASSES, dtype=np.float64)
                my_arch = seed_analysis.get_archetype(y, x)
                # We need access to other seed analyses. 
                # Wait, predict_full_grid_vectorized only takes `seed_analysis` for the current seed!
                # We can't easily check other seeds' archetypes without changing the function signature.
                # Let's just use the precomputed cross_seed_grid for now, but wait, I can't.
"""

# Let's check the function signature
