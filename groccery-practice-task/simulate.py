"""Offline simulation — run the bot against a local game engine.

Usage:
    python simulate.py                        # uses default round0_state.json + order_sequence.json
    python simulate.py recordings/hard/       # uses files from a recording directory

Records round0_state.json and order_sequence.json from live games (main.py),
then replays them here for unlimited fast iteration without tokens.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from bot.model import parse_state
from bot.planner import Planner
from bot.simulator import LocalGame


def simulate(data_dir: str | None = None, verbose: bool = True) -> int:
    if data_dir:
        base = Path(data_dir)
        round0_path = base / "round0_state.json"
        orders_path = base / "order_sequence.json"
    else:
        round0_path = Path("round0_state.json")
        orders_path = Path("order_sequence.json")

    if not round0_path.exists():
        print(f"Missing {round0_path}. Run a live game first to record it.")
        sys.exit(1)
    if not orders_path.exists():
        print(f"Missing {orders_path}. Run a live game first to record it.")
        sys.exit(1)

    game = LocalGame.from_files(round0_path, orders_path)
    planner = Planner()

    total_plan_ms = 0.0
    max_plan_ms = 0.0

    while True:
        msg = game.get_state()
        state = parse_state(msg)

        if state.round == 0 and verbose:
            print(f"  Grid: {state.width}x{state.height}")
            print(f"  Bots: {len(state.bots)}")
            print(f"  Items on map: {len(state.items)}")
            print(f"  Drop-off zones: {state.drop_off_zones}")
            active = state.active_order
            if active:
                print(f"  First order: {active.items_required}")

        t0 = time.monotonic()
        actions = planner.plan(state)
        elapsed_ms = (time.monotonic() - t0) * 1000
        total_plan_ms += elapsed_ms
        max_plan_ms = max(max_plan_ms, elapsed_ms)

        if verbose and (state.round % 50 == 0 or state.round < 5):
            active = state.active_order
            print(
                f"Round {state.round}/{state.max_rounds} | "
                f"Score: {state.score} | "
                f"Plan: {elapsed_ms:.0f}ms"
            )

        result = game.step(actions)
        if result:
            if verbose:
                print(
                    f"\nGame over! Score: {result['score']}, "
                    f"Items: {result['items_delivered']}, "
                    f"Rounds: {result['rounds_used']}"
                )
                print(
                    f"Planning: total={total_plan_ms:.0f}ms, "
                    f"avg={total_plan_ms / max(result['rounds_used'], 1):.1f}ms/round, "
                    f"max={max_plan_ms:.0f}ms"
                )
            return result["score"]

    return 0


def main() -> None:
    data_dir = None
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    print("=== Offline Simulation ===")
    score = simulate(data_dir)
    print(f"\nFinal score: {score}")


if __name__ == "__main__":
    main()
