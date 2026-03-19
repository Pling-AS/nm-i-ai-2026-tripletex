"""Grocery Bot — NM i AI 2026 competition entry."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from bot.model import parse_state
from bot.planner import Planner


def _load_dotenv() -> None:
    try:
        dotenv = __import__("dotenv")
    except ModuleNotFoundError:
        return
    dotenv.load_dotenv()


_load_dotenv()

DIFFICULTY_MAP = {
    "easy": "WS_ENDPOINT_EASY",
    "medium": "WS_ENDPOINT_MEDIUM",
    "hard": "WS_ENDPOINT_HARD",
    "expert": "WS_ENDPOINT_EXPERT",
    "nightmare": "WS_ENDPOINT_NIGHTMARE",
}


async def play(ws_url: str, difficulty: str = "unknown") -> None:
    websockets = __import__("websockets")
    planner = Planner()
    order_log: list[dict] = []
    seen_order_ids: set[str] = set()

    rec_dir = Path("recordings") / difficulty
    rec_dir.mkdir(parents=True, exist_ok=True)

    async with websockets.connect(ws_url) as ws:
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)

            if msg["type"] == "game_over":
                print(
                    f"Game over! Score: {msg['score']}, "
                    f"Items: {msg.get('items_delivered', '?')}, "
                    f"Rounds: {msg.get('rounds_used', '?')}"
                )
                if order_log:
                    for target in [
                        rec_dir / "order_sequence.json",
                        Path("order_sequence.json"),
                    ]:
                        with open(target, "w") as f:
                            json.dump(order_log, f, indent=2)
                    print(
                        f"  Order sequence saved: {len(order_log)} orders -> {rec_dir}/"
                    )
                break

            if msg.get("round", -1) == 0:
                for target in [
                    rec_dir / "round0_state.json",
                    Path("round0_state.json"),
                ]:
                    with open(target, "w") as f:
                        json.dump(msg, f, indent=2)

            state = parse_state(msg)

            for order in state.orders:
                if order.id not in seen_order_ids:
                    seen_order_ids.add(order.id)
                    order_log.append(
                        {
                            "round": state.round,
                            "order_id": order.id,
                            "items": order.items_required,
                            "status": order.status,
                        }
                    )

            if state.round == 0:
                print(f"  Grid: {state.width}x{state.height}")
                print(f"  Walls: {len(state.walls)} cells")
                print(f"  Drop-off zones: {state.drop_off_zones}")
                print(f"  Bot positions: {[b.position for b in state.bots]}")
                print(f"  Item types on map: {set(i.type for i in state.items)}")
                active = state.active_order
                if active:
                    print(f"  Active order needs: {active.items_required}")
                    print(f"  Active remaining: {active.remaining}")

                item_pos = {i.position: i.type[0] for i in state.items}
                drop_set = set(state.drop_off_zones)
                bot_pos = {b.position for b in state.bots}
                for y in range(state.height):
                    row = ""
                    for x in range(state.width):
                        p = (x, y)
                        if p in bot_pos:
                            row += "B"
                        elif p in drop_set:
                            row += "D"
                        elif p in item_pos:
                            row += item_pos[p]
                        elif p in state.walls:
                            row += "#"
                        else:
                            row += "."
                    print(f"  {y:2d} {row}")

            t0 = time.monotonic()
            actions = planner.plan(state)
            elapsed = time.monotonic() - t0

            if state.round % 50 == 0 or state.round < 30:
                active = state.active_order
                preview = state.preview_order
                print(
                    f"Round {state.round}/{state.max_rounds} | "
                    f"Score: {state.score} | "
                    f"Bots: {len(state.bots)} | "
                    f"Items on map: {len(state.items)} | "
                    f"Plan: {elapsed * 1000:.0f}ms"
                )
                if len(state.bots) == 1:
                    # BatchPlanner path
                    b = state.bots[0]
                    trip = planner._batch._trip
                    if trip:
                        remaining = [
                            f"{s.item_type}@{s.shelf_pos}"
                            for s in trip.pickups[trip.pickup_index :]
                        ]
                        print(
                            f"  Bot {b.id} @ {b.position} inv={b.inventory} "
                            f"| Trip: {trip.pickup_index}/{len(trip.pickups)} done "
                            f"| Next: {remaining}"
                        )
                    else:
                        print(
                            f"  Bot {b.id} @ {b.position} inv={b.inventory} "
                            f"| No trip planned"
                        )
                else:
                    for b in state.bots:
                        trip = planner.get_bot_trip(b.id)
                        if trip:
                            remaining = [
                                f"{s.item_type}@{s.shelf_pos}"
                                for s in trip.pickups[trip.pickup_index :]
                            ]
                            print(
                                f"  Bot {b.id} @ {b.position} inv={b.inventory} "
                                f"| Trip: {trip.pickup_index}/{len(trip.pickups)} done "
                                f"| Next: {remaining}"
                            )
                        else:
                            print(
                                f"  Bot {b.id} @ {b.position} inv={b.inventory} "
                                f"| No trip planned"
                            )
                print(f"  Active order: {active.remaining if active else 'NONE'}")
                print(f"  Actions: {actions}")

            await ws.send(json.dumps({"actions": actions}))


def main() -> None:
    difficulty = "easy"
    if len(sys.argv) > 1:
        difficulty = sys.argv[1].lower()

    if difficulty not in DIFFICULTY_MAP:
        print(f"Unknown difficulty: {difficulty}")
        print(f"Available: {', '.join(DIFFICULTY_MAP.keys())}")
        sys.exit(1)

    env_key = DIFFICULTY_MAP[difficulty]
    ws_url = os.environ.get(env_key)

    if not ws_url:
        print(f"Missing {env_key} in .env")
        sys.exit(1)

    print(f"Starting bot on {difficulty} difficulty...")
    asyncio.run(play(ws_url, difficulty))


if __name__ == "__main__":
    main()
