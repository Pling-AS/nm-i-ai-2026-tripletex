"""Central Coordinator — assigns goals to bots, runs PIBT for collision-free movement.

Architecture:
  BatchPlanner (per bot) decides WHAT to pick (trip planning + scoring).
  PIBT decides HOW to move (collision-free single-step pathfinding).
  Coordinator wires them together and handles pick_up / drop_off actions.

Per-round flow:
  1. Sync per-bot BatchPlanners; compute reserved types centrally.
  2. Call update_trip() on each bot -> get phase ("deliver" / "pick" / "idle").
  3. Identify stationary actions (pick_up if adjacent to shelf, drop_off if on DZ).
  4. Compute goal positions for moving bots from their phase + trip state.
  5. Run pibt_step() for moving bots -> collision-free next positions.
  6. Convert positions to game action dicts.
"""

from __future__ import annotations

import random
from collections import Counter

from bot.batch_planner import BatchPlanner
from bot.map_cache import MapCache
from bot.model import BotState, GameState, Item, Pos
from bot.pibt import pibt_step, pos_to_action


class Coordinator:
    __slots__ = (
        "_planners",
        "_priorities",
        "_rng",
    )

    def __init__(self) -> None:
        self._planners: dict[int, BatchPlanner] = {}
        self._priorities: dict[int, float] = {}
        self._rng = random.Random(42)

    def plan(self, state: GameState, mc: MapCache) -> list[dict]:
        bots = sorted(state.bots, key=lambda b: b.id)
        active = state.active_order

        if active is None:
            return [{"bot": b.id, "action": "wait"} for b in bots]

        self._sync_planners(state)

        # Phase 1: Update trips for all bots (centralized reserved-type tracking)
        phases: dict[int, str] = {}
        for bot in bots:
            planner = self._planners[bot.id]
            reserved = self._collect_reserved_types(bot.id, state)
            planner.set_reserved_types(reserved)
            reserved_shelves = self._collect_reserved_shelves(bot.id)

            phase, _ = planner.update_trip(
                state,
                mc,
                bot_id=bot.id,
                reserved_shelves=reserved_shelves,
                multi_bot=True,
            )
            phases[bot.id] = phase

        # Phase 2: Determine stationary actions (pick_up / drop_off)
        stationary: dict[int, dict] = {}
        goals: dict[int, Pos] = {}

        for bot in bots:
            phase = phases[bot.id]
            planner = self._planners[bot.id]
            trip = planner._trip

            if phase == "deliver":
                dz = mc.nearest_drop_zone(bot.position)
                if bot.position in set(mc.drop_zones):
                    if self._has_deliverable(bot, state):
                        stationary[bot.id] = {"bot": bot.id, "action": "drop_off"}
                        continue
                goals[bot.id] = dz

            elif phase == "pick":
                if trip and trip.pickup_index < len(trip.pickups):
                    step = trip.pickups[trip.pickup_index]
                    dist = abs(bot.position[0] - step.shelf_pos[0]) + abs(
                        bot.position[1] - step.shelf_pos[1]
                    )
                    if dist == 1 and not bot.is_full:
                        item_id = self._find_item_id(
                            state.items, step.shelf_pos, step.item_type
                        )
                        if item_id:
                            stationary[bot.id] = {
                                "bot": bot.id,
                                "action": "pick_up",
                                "item_id": item_id,
                            }
                            continue

                    pickup_cell = mc.best_pickup_cell(bot.position, step.shelf_pos)
                    if pickup_cell:
                        goals[bot.id] = pickup_cell
                    else:
                        goals[bot.id] = bot.position
                else:
                    goals[bot.id] = bot.position

            else:
                goals[bot.id] = self._retreat_target(bot, mc)

        # Phase 3: Run PIBT for moving bots
        moving_bots = [b for b in bots if b.id not in stationary]
        locked = {b.position for b in bots if b.id in stationary}

        if moving_bots:
            moving_positions = [b.position for b in moving_bots]
            moving_goals = [goals.get(b.id, b.position) for b in moving_bots]
            moving_priorities = [self._priorities.get(b.id, 0.0) for b in moving_bots]

            next_positions = pibt_step(
                moving_positions,
                moving_goals,
                mc,
                locked=locked,
                priorities=moving_priorities,
                rng=self._rng,
            )

            moving_actions: dict[int, dict] = {}
            for idx, bot in enumerate(moving_bots):
                moving_actions[bot.id] = pos_to_action(
                    bot.id, bot.position, next_positions[idx]
                )
        else:
            moving_actions = {}

        # Phase 4: Update priorities
        for bot in bots:
            goal = goals.get(bot.id, bot.position)
            if bot.id not in self._priorities:
                d = mc.distance(bot.position, goal)
                self._priorities[bot.id] = d / max(mc.width * mc.height, 1)

            if bot.id in stationary:
                self._priorities[bot.id] -= int(self._priorities[bot.id])
            elif bot.position != goal:
                self._priorities[bot.id] += 1
            else:
                self._priorities[bot.id] -= int(self._priorities[bot.id])

        # Phase 5: Assemble actions in bot order
        result: list[dict] = []
        for bot in bots:
            if bot.id in stationary:
                result.append(stationary[bot.id])
            elif bot.id in moving_actions:
                result.append(moving_actions[bot.id])
            else:
                result.append({"bot": bot.id, "action": "wait"})

        return result

    # --- Bot planner management ---

    def _sync_planners(self, state: GameState) -> None:
        live_ids = {b.id for b in state.bots}
        stale = [bid for bid in self._planners if bid not in live_ids]
        for bid in stale:
            del self._planners[bid]
            self._priorities.pop(bid, None)
        for bot in state.bots:
            if bot.id not in self._planners:
                self._planners[bot.id] = BatchPlanner()

    def _collect_reserved_types(
        self, exclude_bot_id: int, state: GameState
    ) -> Counter[str]:
        out: Counter[str] = Counter()
        active = state.active_order
        active_remaining = Counter(active.remaining) if active else Counter()

        for bot_id, planner in self._planners.items():
            if bot_id == exclude_bot_id:
                continue
            trip = planner._trip
            if trip is not None:
                for step in trip.pickups[trip.pickup_index :]:
                    out[step.item_type] += 1

            if active_remaining:
                bot = next((b for b in state.bots if b.id == bot_id), None)
                if bot:
                    for t in bot.inventory:
                        if active_remaining[t] > out[t]:
                            out[t] += 1
        return out

    def _collect_reserved_shelves(self, exclude_bot_id: int) -> set[Pos]:
        out: set[Pos] = set()
        for bot_id, planner in self._planners.items():
            if bot_id == exclude_bot_id:
                continue
            trip = planner._trip
            if trip is None:
                continue
            for step in trip.pickups[trip.pickup_index :]:
                out.add(step.shelf_pos)
        return out

    # --- Helpers ---

    def _has_deliverable(self, bot: BotState, state: GameState) -> bool:
        active = state.active_order
        if not active:
            return False
        remaining = list(active.remaining)
        for t in bot.inventory:
            if t in remaining:
                return True
        return False

    @staticmethod
    def _find_item_id(items: list[Item], shelf_pos: Pos, item_type: str) -> str | None:
        for item in items:
            if item.position == shelf_pos and item.type == item_type:
                return item.id
        return None

    def _retreat_target(self, bot: BotState, mc: MapCache) -> Pos:
        best: Pos | None = None
        best_score: tuple[int, int] | None = None
        for cell in mc.walkable:
            if cell in mc.narrow_cells or cell in set(mc.drop_zones):
                continue
            min_dz = min(mc.distance(cell, dz) for dz in mc.drop_zones)
            reach = mc.distance(bot.position, cell)
            if reach >= 9999:
                continue
            score = (-min_dz, reach)
            if best_score is None or score < best_score:
                best_score = score
                best = cell
        return best if best is not None else bot.position
