with open('/Users/m/Programmering/2025/nm-i-ai/task-3-astar_island/predictor.py', 'r') as f:
    content = f.read()

content = content.replace(
    "def predict_full_grid_vectorized(\n    seed_index: int,\n    observation_store: ObservationStore,\n    seed_analysis: SeedAnalysis,\n) -> NDArray[np.floating]:",
    "def predict_full_grid_vectorized(\n    seed_index: int,\n    observation_store: ObservationStore,\n    seed_analyses: list[SeedAnalysis],\n) -> NDArray[np.floating]:"
)

content = content.replace(
    "h, w = seed_analysis.height, seed_analysis.width\n    grid = seed_analysis.grid\n    class_masks = seed_analysis.class_masks",
    "seed_analysis = seed_analyses[seed_index]\n    h, w = seed_analysis.height, seed_analysis.width\n    grid = seed_analysis.grid\n    class_masks = seed_analysis.class_masks"
)

old_cross_seed = """    # Pre-compute cross-seed pooled counts (sum of other seeds at same position)
    cross_seed_grid = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)
    if CROSS_SEED_LAMBDA > 0:
        for other_seed in range(observation_store.seeds_count):
            if other_seed != seed_index:
                cross_seed_grid += observation_store.get_seed_counts(other_seed).astype(np.float64)"""

new_cross_seed = """    # Pre-compute cross-seed pooled counts (sum of other seeds at same position WITH SAME ARCHETYPE)
    cross_seed_grid = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)
    if CROSS_SEED_LAMBDA > 0:
        my_archetypes = seed_analysis.archetypes
        for other_seed in range(observation_store.seeds_count):
            if other_seed != seed_index:
                other_archetypes = seed_analyses[other_seed].archetypes
                match_mask = (my_archetypes == other_archetypes)
                other_counts = observation_store.get_seed_counts(other_seed).astype(np.float64)
                cross_seed_grid += other_counts * match_mask[..., np.newaxis]"""

content = content.replace(old_cross_seed, new_cross_seed)

with open('/Users/m/Programmering/2025/nm-i-ai/task-3-astar_island/predictor.py', 'w') as f:
    f.write(content)
