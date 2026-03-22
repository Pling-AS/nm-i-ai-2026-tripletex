import re

with open('task-3-astar_island/run.py', 'r') as f:
    content = f.read()

old_loop = '''    for i, q in enumerate(coverage_queries):
        try:
            result = client.simulate(
                round_id=round_id,
                seed_index=q.seed_index,
                viewport_x=q.viewport_x,
                viewport_y=q.viewport_y,
                viewport_w=q.viewport_w,
                viewport_h=q.viewport_h,
            )
            observation_store.add_observation(
                seed_index=q.seed_index,
                viewport=result.viewport,
                grid=result.grid,
                archetypes=seed_analyses[q.seed_index].archetypes,
            )
            coverage_successes += 1
            # Capture settlement metadata from simulate response
            if result.settlements:
                for s in result.settlements:
                    all_settlement_obs.append(s)
            if (i + 1) % 10 == 0 or i == n_coverage - 1:
                log(
                    f"    [{i + 1}/{n_coverage}] seed={q.seed_index} "
                    f"vp=({q.viewport_x},{q.viewport_y}) "
                    f"budget={result.queries_used}/{result.queries_max}"
                )
        except BudgetExhaustedError as e:
            coverage_failures += 1
            repeat_budget = 0
            log(f"    [{i + 1}/{n_coverage}] BUDGET EXHAUSTED: {e}")
            break
        except APIError as e:
            coverage_failures += 1
            log(f"    [{i + 1}/{n_coverage}] API ERROR: {e}")
            if e.status_code == 429:
                repeat_budget = 0
                log("    Persistent rate limit after retries, stopping coverage.")
                break
        except Exception as e:
            coverage_failures += 1
            log(f"    [{i + 1}/{n_coverage}] ERROR: {e}")

    if n_coverage > 0:
        coverage_ratio = coverage_successes / n_coverage
        log(
            f"  Coverage health: {coverage_successes}/{n_coverage} successful "
            f"({coverage_ratio:.0%}), failures={coverage_failures}"
        )
        if coverage_ratio < 0.5:
            log(
                "  CRITICAL: Coverage query success fell below 50%. "
                "Predictions may be degraded."
            )
            if repeat_budget > 0:
                log("  Skipping repeat phase because coverage is severely degraded.")
                repeat_budget = 0'''

new_loop = '''    last_queries_used = queries_used
    last_queries_max = queries_max

    for i, q in enumerate(coverage_queries):
        try:
            result = client.simulate(
                round_id=round_id,
                seed_index=q.seed_index,
                viewport_x=q.viewport_x,
                viewport_y=q.viewport_y,
                viewport_w=q.viewport_w,
                viewport_h=q.viewport_h,
            )
            last_queries_used = result.queries_used
            last_queries_max = result.queries_max
            
            observation_store.add_observation(
                seed_index=q.seed_index,
                viewport=result.viewport,
                grid=result.grid,
                archetypes=seed_analyses[q.seed_index].archetypes,
            )
            coverage_successes += 1
            # Capture settlement metadata from simulate response
            if result.settlements:
                for s in result.settlements:
                    all_settlement_obs.append(s)
            if (i + 1) % 10 == 0 or i == n_coverage - 1:
                log(
                    f"    [{i + 1}/{n_coverage}] seed={q.seed_index} "
                    f"vp=({q.viewport_x},{q.viewport_y}) "
                    f"budget={result.queries_used}/{result.queries_max}"
                )
        except BudgetExhaustedError as e:
            coverage_failures += 1
            repeat_budget = 0
            log(f"    [{i + 1}/{n_coverage}] BUDGET EXHAUSTED: {e}")
            break
        except APIError as e:
            coverage_failures += 1
            log(f"    [{i + 1}/{n_coverage}] API ERROR: {e}")
            if e.status_code == 429:
                repeat_budget = 0
                log("    Persistent rate limit after retries, stopping coverage.")
                break
        except Exception as e:
            coverage_failures += 1
            log(f"    [{i + 1}/{n_coverage}] ERROR: {e}")

    if n_coverage > 0:
        coverage_ratio = coverage_successes / n_coverage
        log(
            f"  Coverage health: {coverage_successes}/{n_coverage} successful "
            f"({coverage_ratio:.0%}), failures={coverage_failures}"
        )
        if coverage_ratio < 0.5:
            log(
                "  CRITICAL: Coverage query success fell below 50%. "
                "Predictions may be degraded."
            )
            if repeat_budget > 0:
                log("  Skipping repeat phase because coverage is severely degraded.")
                repeat_budget = 0
        elif repeat_budget > 0:
            # Dynamically update repeat budget based on actual usage
            repeat_budget = max(0, last_queries_max - last_queries_used)
            log(f"  Updated repeat budget based on actual usage: {repeat_budget}")'''

content = content.replace(old_loop, new_loop)

with open('task-3-astar_island/run.py', 'w') as f:
    f.write(content)
