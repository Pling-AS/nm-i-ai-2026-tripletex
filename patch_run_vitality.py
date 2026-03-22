import re

with open('task-3-astar_island/run.py', 'r') as f:
    content = f.read()

# In _query_phase, set observation_store.settlement_vitality
old_vitality_set = '''        vitality = {
            "alive_rate": alive_rate,
            "avg_population": avg_pop,
            "avg_food": avg_food,
            "avg_defense": avg_def,
            "n_observations": n_total,
        }
        log(
            f"  Settlement vitality: alive={alive_rate:.1%} "
            f"({n_alive}/{n_total}), avg_pop={avg_pop:.2f}, "
            f"avg_food={avg_food:.2f}, avg_def={avg_def:.2f}"
        )
        set_settlement_vitality(vitality)
    else:
        log("  No settlement data captured (predict-only or empty)")
        set_settlement_vitality(None)'''

new_vitality_set = '''        vitality = {
            "alive_rate": alive_rate,
            "avg_population": avg_pop,
            "avg_food": avg_food,
            "avg_defense": avg_def,
            "n_observations": n_total,
        }
        log(
            f"  Settlement vitality: alive={alive_rate:.1%} "
            f"({n_alive}/{n_total}), avg_pop={avg_pop:.2f}, "
            f"avg_food={avg_food:.2f}, avg_def={avg_def:.2f}"
        )
        observation_store.settlement_vitality = vitality
        set_settlement_vitality(vitality)
    else:
        log("  No settlement data captured (predict-only or empty)")
        observation_store.settlement_vitality = None
        set_settlement_vitality(None)'''

content = content.replace(old_vitality_set, new_vitality_set)

# In run(), set_settlement_vitality from observation_store if predict_only
old_run = '''    if predict_only:
        # Load saved observations
        obs_file = _obs_path(round_id)
        if not obs_file.with_suffix(".npz").exists():
            log(f"ERROR: No saved observations at {obs_file}.npz")
            log("  Run without --predict-only first to query and save observations.")
            sys.exit(1)
        log(f"  Loading saved observations from {obs_file}...")
        observation_store = ObservationStore.load(obs_file)
    else:
        # Full query pipeline
        observation_store = _query_phase(
            client, round_id, seed_analyses, width, height, seeds_count
        )'''

new_run = '''    if predict_only:
        # Load saved observations
        obs_file = _obs_path(round_id)
        if not obs_file.with_suffix(".npz").exists():
            log(f"ERROR: No saved observations at {obs_file}.npz")
            log("  Run without --predict-only first to query and save observations.")
            sys.exit(1)
        log(f"  Loading saved observations from {obs_file}...")
        observation_store = ObservationStore.load(obs_file)
        set_settlement_vitality(observation_store.settlement_vitality)
    else:
        # Full query pipeline
        observation_store = _query_phase(
            client, round_id, seed_analyses, width, height, seeds_count
        )'''

content = content.replace(old_run, new_run)

with open('task-3-astar_island/run.py', 'w') as f:
    f.write(content)
