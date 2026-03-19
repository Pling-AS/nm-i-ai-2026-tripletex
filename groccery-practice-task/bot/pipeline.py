"""Pipeline Planner — global task assignment with pipelined picking and delivery.

Radical redesign: instead of per-bot BatchPlanner deciding independently,
a central planner assigns items to bots globally, maximizing throughput.

Key principles:
  1. Global demand tracking: know what's needed, what's in transit, what's assigned.
  2. Centralized assignment: assign items to nearest available bots.
  3. Fill inventory before delivering: pick up to 3 items, then deliver.
  4. Pipeline: while deliverers head to DZ, pickers grab next order's items.
  5. Zero idle time: every bot always has a goal.
  6. PIBT for collision-free movement.
  7. Deliver immediately if it would COMPLETE the active order (+5 bonus).

Per-round flow:
  1. Sync progress (detect pickups/deliveries that happened).
  2. Invalidate plans if order changed.
  3. Assign tasks to idle bots.
  4. Execute: stationary actions (pick_up/drop_off) or PIBT movement.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

from bot.map_cache import MapCache
from bot.model import BotState, GameState, Item, Order, Pos
from bot.pibt import pibt_step, pos_to_action


@dataclass(slots=True)
class PickTask:
    item_type: str
    shelf_pos: Pos
    pickup_cell: Pos  # walkable cell adjacent to shelf


@dataclass(slots=True)
class BotPlan:
    pickups: list[PickTask] = field(default_factory=list)
    pickup_index: int = 0
    phase: str = "idle"  # "pick", "deliver", "idle"
    for_order_id: str = ""  # which order these pickups are for
    _prev_inv_count: int = 0  # inventory count last round (for sync)


class PipelinePlanner:
    __slots__ = (
        "_plans",
        "_priorities",
        "_rng",
    )

    def __init__(self) -> None:
        self._plans: dict[int, BotPlan] = {}
        self._priorities: dict[int, float] = {}
        self._rng = random.Random(42)

    def plan(self, state: GameState, mc: MapCache) -> list[dict]:
        bots = sorted(state.bots, key=lambda b: b.id)
        active = state.active_order

        if active is None:
            return [{"bot": b.id, "action": "wait"} for b in bots]

        self._sync_plans(bots)
        self._sync_all_progress(bots, state)
        self._invalidate_stale_plans(bots, state)
        self._force_endgame_delivery(bots, state, mc)
        self._assign_tasks(bots, state, mc)

        # Build stationary actions and goals for PIBT
        stationary: dict[int, dict] = {}
        goals: dict[int, Pos] = {}

        for bot in bots:
            plan = self._plans[bot.id]

            if plan.phase == "deliver":
                dz = mc.nearest_drop_zone(bot.position)
                if bot.position in set(mc.drop_zones):
                    if self._has_deliverable(bot, active):
                        stationary[bot.id] = {"bot": bot.id, "action": "drop_off"}
                        continue
                    else:
                        # On DZ but nothing to deliver — shouldn't be in deliver phase
                        plan.phase = "idle"
                        goals[bot.id] = self._idle_target(bot, mc, bots)
                        continue
                goals[bot.id] = dz

            elif plan.phase == "pick":
                if plan.pickup_index < len(plan.pickups):
                    task = plan.pickups[plan.pickup_index]
                    dist = abs(bot.position[0] - task.shelf_pos[0]) + abs(
                        bot.position[1] - task.shelf_pos[1]
                    )
                    if dist == 1 and not bot.is_full:
                        item_id = _find_item_id(
                            state.items, task.shelf_pos, task.item_type
                        )
                        if item_id:
                            stationary[bot.id] = {
                                "bot": bot.id,
                                "action": "pick_up",
                                "item_id": item_id,
                            }
                            continue
                    goals[bot.id] = task.pickup_cell
                else:
                    # All pickups done → deliver
                    plan.phase = "deliver"
                    goals[bot.id] = mc.nearest_drop_zone(bot.position)

            else:  # idle
                goals[bot.id] = self._idle_target(bot, mc, bots)

        # Run PIBT for moving bots
        moving_bots = [b for b in bots if b.id not in stationary]
        locked = {b.position for b in bots if b.id in stationary}

        moving_actions: dict[int, dict] = {}
        if moving_bots:
            positions = [b.position for b in moving_bots]
            g = [goals.get(b.id, b.position) for b in moving_bots]
            pris = [self._priorities.get(b.id, 0.0) for b in moving_bots]

            next_pos = pibt_step(
                positions, g, mc, locked=locked, priorities=pris, rng=self._rng
            )

            for idx, bot in enumerate(moving_bots):
                moving_actions[bot.id] = pos_to_action(
                    bot.id, bot.position, next_pos[idx]
                )

        # Update priorities
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

        # Assemble
        result: list[dict] = []
        for bot in bots:
            if bot.id in stationary:
                result.append(stationary[bot.id])
            elif bot.id in moving_actions:
                result.append(moving_actions[bot.id])
            else:
                result.append({"bot": bot.id, "action": "wait"})
        return result

    # ──────────────────────── ASSIGNMENT ────────────────────────

    def _assign_tasks(
        self,
        bots: list[BotState],
        state: GameState,
        mc: MapCache,
    ) -> None:
        active = state.active_order
        preview = state.preview_order
        if active is None:
            return

        dispensers = _build_dispensers(state.items)

        # 1. Compute what the active order still needs (globally)
        active_remaining = list(active.remaining)

        # 2. Subtract items in transit (carried by bots heading to deliver)
        for bot in bots:
            plan = self._plans[bot.id]
            if plan.phase == "deliver":
                for t in bot.inventory:
                    if t in active_remaining:
                        active_remaining.remove(t)

        # 3. Subtract items assigned as pending pickups for active order
        for bot in bots:
            plan = self._plans[bot.id]
            if plan.phase == "pick" and plan.for_order_id == active.id:
                for task in plan.pickups[plan.pickup_index :]:
                    if task.item_type in active_remaining:
                        active_remaining.remove(task.item_type)
                # Also subtract items already picked up by this bot that match active
                for t in bot.inventory:
                    if t in active_remaining:
                        active_remaining.remove(t)

        # 4. Figure out which bots are idle and need tasks
        idle_bots: list[BotState] = []
        for bot in bots:
            plan = self._plans[bot.id]
            if plan.phase == "idle":
                idle_bots.append(bot)

        if not idle_bots:
            return

        preview_remaining: list[str] = []
        if preview:
            preview_remaining = list(preview.items_required)
            for b2 in bots:
                p2 = self._plans[b2.id]
                if p2.phase == "pick" and p2.for_order_id == preview.id:
                    for task in p2.pickups[p2.pickup_index :]:
                        if task.item_type in preview_remaining:
                            preview_remaining.remove(task.item_type)

        idle_bots.sort(
            key=lambda b: mc.distance(b.position, mc.nearest_drop_zone(b.position))
        )

        for bot in idle_bots:
            if bot.inventory and self._has_deliverable(bot, active):
                remaining_check = list(active_remaining)
                for t in bot.inventory:
                    if t in remaining_check:
                        remaining_check.remove(t)

                if not remaining_check or bot.is_full:
                    self._plans[bot.id] = BotPlan(
                        phase="deliver", for_order_id=active.id
                    )
                    continue

            slots = bot.free_slots
            if slots <= 0:
                if self._has_deliverable(bot, active):
                    self._plans[bot.id] = BotPlan(
                        phase="deliver", for_order_id=active.id
                    )
                continue

            pickups = self._pick_items(
                bot, active_remaining, dispensers, mc, slots, active.id
            )
            for task in pickups:
                if task.item_type in active_remaining:
                    active_remaining.remove(task.item_type)

            # If we have leftover slots and no more active demand, try preview
            remaining_slots = slots - len(pickups)
            if (
                remaining_slots > 0
                and preview
                and not active_remaining
                and state.rounds_left > 60
            ):
                preview_pickups = self._pick_items(
                    bot,
                    preview_remaining,
                    dispensers,
                    mc,
                    remaining_slots,
                    preview.id,
                    start_from=pickups[-1].pickup_cell if pickups else None,
                )
                for task in preview_pickups:
                    if task.item_type in preview_remaining:
                        preview_remaining.remove(task.item_type)
                pickups.extend(preview_pickups)

            if pickups:
                ordered = self._optimize_order(bot.position, pickups, mc)
                self._plans[bot.id] = BotPlan(
                    pickups=ordered,
                    pickup_index=0,
                    phase="pick",
                    for_order_id=active.id,
                    _prev_inv_count=len(bot.inventory),
                )
            elif bot.inventory and self._has_deliverable(bot, active):
                self._plans[bot.id] = BotPlan(phase="deliver", for_order_id=active.id)

    def _pick_items(
        self,
        bot: BotState,
        demand: list[str],
        dispensers: dict[str, list[Pos]],
        mc: MapCache,
        max_picks: int,
        order_id: str,
        start_from: Pos | None = None,
    ) -> list[PickTask]:
        """Greedily assign closest items from demand to this bot."""
        pickups: list[PickTask] = []
        available = list(demand)  # work on a copy
        ref_pos = start_from or bot.position

        for _ in range(max_picks):
            if not available:
                break

            best_cost = 9999
            best_type_idx = -1
            best_shelf: Pos = (0, 0)
            best_cell: Pos = (0, 0)

            for i, item_type in enumerate(available):
                shelves = dispensers.get(item_type, [])
                for shelf in shelves:
                    cell = mc.best_pickup_cell(ref_pos, shelf)
                    if cell is None:
                        continue
                    cost = mc.distance(ref_pos, cell)
                    if cost < best_cost:
                        best_cost = cost
                        best_type_idx = i
                        best_shelf = shelf
                        best_cell = cell

            if best_type_idx < 0:
                break

            pickups.append(
                PickTask(
                    item_type=available[best_type_idx],
                    shelf_pos=best_shelf,
                    pickup_cell=best_cell,
                )
            )
            available.pop(best_type_idx)
            ref_pos = best_cell

        return pickups

    def _optimize_order(
        self, start: Pos, pickups: list[PickTask], mc: MapCache
    ) -> list[PickTask]:
        """Nearest-neighbor TSP to minimize travel."""
        if len(pickups) <= 1:
            return list(pickups)
        remaining = list(range(len(pickups)))
        ordered: list[PickTask] = []
        current = start
        while remaining:
            best_idx = min(
                remaining, key=lambda i: mc.distance(current, pickups[i].pickup_cell)
            )
            remaining.remove(best_idx)
            ordered.append(pickups[best_idx])
            current = pickups[best_idx].pickup_cell
        return ordered

    # ──────────────────────── PROGRESS SYNC ────────────────────────

    def _sync_all_progress(self, bots: list[BotState], state: GameState) -> None:
        """Detect completed pickups by comparing inventory counts."""
        active = state.active_order
        for bot in bots:
            plan = self._plans[bot.id]

            if plan.phase == "pick" and plan.pickup_index < len(plan.pickups):
                # Check if inventory grew → pickup happened
                if len(bot.inventory) > plan._prev_inv_count:
                    plan.pickup_index += 1
                    plan._prev_inv_count = len(bot.inventory)

                # All pickups done → deliver
                if plan.pickup_index >= len(plan.pickups):
                    if active and self._has_deliverable(bot, active):
                        plan.phase = "deliver"
                    else:
                        plan.phase = "idle"

            elif plan.phase == "deliver":
                if not bot.inventory:
                    plan.phase = "idle"
                    plan.pickups = []
                    plan.pickup_index = 0
                elif active and not self._has_deliverable(bot, active):
                    # Carrying items but none match active order
                    plan.phase = "idle"
                    plan.pickups = []
                    plan.pickup_index = 0

            plan._prev_inv_count = len(bot.inventory)

    def _invalidate_stale_plans(self, bots: list[BotState], state: GameState) -> None:
        """Reset plans when orders change."""
        active = state.active_order
        if active is None:
            for bot in bots:
                self._plans[bot.id] = BotPlan()
            return

        for bot in bots:
            plan = self._plans[bot.id]
            if (
                plan.phase == "pick"
                and plan.for_order_id
                and plan.for_order_id != active.id
            ):
                # Order changed — check if carried items are still useful
                if bot.inventory and self._has_deliverable(bot, active):
                    plan.phase = "deliver"
                    plan.pickups = []
                    plan.pickup_index = 0
                    plan.for_order_id = active.id
                else:
                    plan.phase = "idle"
                    plan.pickups = []
                    plan.pickup_index = 0
                    plan.for_order_id = ""

            # End-game force delivery is handled in _force_endgame_delivery()

    def _sync_plans(self, bots: list[BotState]) -> None:
        live_ids = {b.id for b in bots}
        stale = [bid for bid in self._plans if bid not in live_ids]
        for bid in stale:
            del self._plans[bid]
            self._priorities.pop(bid, None)
        for bot in bots:
            if bot.id not in self._plans:
                self._plans[bot.id] = BotPlan(_prev_inv_count=len(bot.inventory))

    # ──────────────────────── END-GAME ────────────────────────

    def _force_endgame_delivery(
        self, bots: list[BotState], state: GameState, mc: MapCache
    ) -> None:
        """Force all bots with deliverable items to deliver when time is short."""
        active = state.active_order
        if active is None:
            return
        for bot in bots:
            plan = self._plans[bot.id]
            if (
                plan.phase == "pick"
                and bot.inventory
                and self._has_deliverable(bot, active)
            ):
                dz = mc.nearest_drop_zone(bot.position)
                dist = mc.distance(bot.position, dz)
                # Need to get to DZ + 1 round to drop off
                remaining_picks = len(plan.pickups) - plan.pickup_index
                # Estimate rounds to finish picking
                if remaining_picks > 0:
                    next_task = plan.pickups[plan.pickup_index]
                    pick_dist = mc.distance(bot.position, next_task.pickup_cell)
                    total_pick_cost = pick_dist + remaining_picks  # rough
                    deliver_after_pick = (
                        total_pick_cost + mc.distance(next_task.pickup_cell, dz) + 1
                    )
                else:
                    deliver_after_pick = dist + 1

                if state.rounds_left <= deliver_after_pick + 2:
                    plan.phase = "deliver"
                    plan.pickups = []
                    plan.pickup_index = 0

    # ──────────────────────── HELPERS ────────────────────────

    @staticmethod
    def _has_deliverable(bot: BotState, active: Order | None) -> bool:
        if not active:
            return False
        remaining = list(active.remaining)
        for t in bot.inventory:
            if t in remaining:
                return True
        return False

    def _idle_target(self, bot: BotState, mc: MapCache, bots: list[BotState]) -> Pos:
        """Park away from DZ and narrow corridors."""
        dz_set = set(mc.drop_zones)
        best: Pos | None = None
        best_score: tuple[int, int] | None = None
        for cell in mc.walkable:
            if cell in mc.narrow_cells or cell in dz_set:
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


def _build_dispensers(items: list[Item]) -> dict[str, list[Pos]]:
    by_type: dict[str, set[Pos]] = {}
    for item in items:
        by_type.setdefault(item.type, set()).add(item.position)
    return {k: sorted(v) for k, v in by_type.items()}


def _find_item_id(items: list[Item], shelf_pos: Pos, item_type: str) -> str | None:
    for item in items:
        if item.position == shelf_pos and item.type == item_type:
            return item.id
    return None
