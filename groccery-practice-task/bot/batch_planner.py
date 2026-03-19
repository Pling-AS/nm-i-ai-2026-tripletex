from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import permutations

from bot.map_cache import MapCache
from bot.model import BotState, GameState, Item, Order, Pos
from bot.pathfinding import next_step_toward

INF = 9999

ACTION_MAP: dict[tuple[int, int], str] = {
    (0, -1): "move_up",
    (0, 1): "move_down",
    (-1, 0): "move_left",
    (1, 0): "move_right",
}


@dataclass(slots=True)
class PickupStep:
    item_type: str
    shelf_pos: Pos


@dataclass(slots=True)
class TripPlan:
    active_id: str | None
    preview_id: str | None
    pickups: list[PickupStep]
    pickup_index: int
    start_inventory: Counter[str]
    acquired: Counter[str]


class BatchPlanner:
    __slots__ = ("_trip", "_reserved_types", "_retreat_target")

    def __init__(self) -> None:
        self._trip: TripPlan | None = None
        self._reserved_types: Counter[str] = Counter()
        self._retreat_target: Pos | None = None

    def set_reserved_types(self, reserved_types: Counter[str] | None) -> None:
        self._reserved_types = Counter(reserved_types or {})

    def update_trip(
        self,
        state: GameState,
        mc: MapCache,
        bot_id: int = 0,
        reserved_shelves: set[Pos] | None = None,
        multi_bot: bool = False,
    ) -> tuple[str, "TripPlan | None"]:
        """Update trip state without computing movement actions.

        Returns:
            (phase, trip) where phase is one of:
            - "deliver": bot should head to DZ (has deliverable items, done picking or forced)
            - "pick": bot should head to next pickup cell
            - "idle": bot has nothing to do
        """
        bot = next((b for b in state.bots if b.id == bot_id), None)
        if bot is None:
            return "idle", None

        active = state.active_order
        if multi_bot:
            if active and state.rounds_left > 80:
                active_need = self._compute_active_need(
                    active, bot.inventory, self._reserved_types
                )
                has_active_items = self._inventory_has_active_item(bot, active)
                preview = (
                    state.preview_order
                    if (not active_need and not has_active_items)
                    else None
                )
            else:
                preview = None
        else:
            preview = state.preview_order

        if active is None:
            self._trip = None
            return "idle", None

        self._sync_trip_progress(bot)

        max_pickups = 2 if multi_bot else 3
        if multi_bot and preview is not None:
            max_pickups = 1

        if not self._is_trip_valid(state, bot):
            self._trip = self._build_best_trip(
                state,
                bot,
                active,
                preview,
                mc,
                reserved_shelves=reserved_shelves,
                max_pickups=max_pickups,
            )
            if self._trip is not None:
                self._retreat_target = None

        if self._should_force_drop_now(state, bot, active, mc):
            return "deliver", self._trip

        if self._trip is None:
            if self._inventory_has_active_item(bot, active):
                return "deliver", None
            return "idle", None

        self._sync_trip_progress(bot)

        if self._trip.pickup_index >= len(self._trip.pickups):
            return "deliver", self._trip

        return "pick", self._trip

    def plan(
        self,
        state: GameState,
        mc: MapCache,
        bot_id: int = 0,
        blocked: set[Pos] | None = None,
        reserved_shelves: set[Pos] | None = None,
        dz_locked: bool = False,
        multi_bot: bool = False,
    ) -> dict:
        bot = next((b for b in state.bots if b.id == bot_id), None)
        if bot is None:
            bot = (
                state.bots[0]
                if state.bots
                else BotState(id=bot_id, position=(0, 0), inventory=[])
            )
        active = state.active_order
        if multi_bot:
            if active and state.rounds_left > 80:
                active_need = self._compute_active_need(
                    active, bot.inventory, self._reserved_types
                )
                has_active_items = self._inventory_has_active_item(bot, active)
                preview = (
                    state.preview_order
                    if (not active_need and not has_active_items)
                    else None
                )
            else:
                preview = None
        else:
            preview = state.preview_order

        if active is None:
            self._trip = None
            return {"bot": bot.id, "action": "wait"}

        self._sync_trip_progress(bot)

        max_pickups = 2 if multi_bot else 3
        if multi_bot and preview is not None:
            max_pickups = 1

        if not self._is_trip_valid(state, bot):
            self._trip = self._build_best_trip(
                state,
                bot,
                active,
                preview,
                mc,
                reserved_shelves=reserved_shelves,
                max_pickups=max_pickups,
            )
            if self._trip is not None:
                self._retreat_target = None

        if self._should_force_drop_now(state, bot, active, mc):
            return self._go_drop(
                bot, mc, state=state, blocked=blocked, dz_locked=dz_locked
            )

        if self._trip is None:
            if self._inventory_has_active_item(bot, active):
                return self._go_drop(
                    bot, mc, state=state, blocked=blocked, dz_locked=dz_locked
                )
            return self._retreat_from_dz_route(bot, mc, blocked=blocked)

        self._sync_trip_progress(bot)
        return self._execute_trip(state, bot, mc, blocked=blocked, dz_locked=dz_locked)

    def _is_trip_valid(self, state: GameState, bot: BotState) -> bool:
        trip = self._trip
        if trip is None:
            return False

        active = state.active_order
        preview = state.preview_order
        if active is None:
            return False

        if trip.active_id != active.id:
            return False
        if trip.preview_id != (preview.id if preview else None):
            return False

        inv = Counter(bot.inventory)
        for k, v in trip.start_inventory.items():
            if inv[k] < v:
                return False

        if trip.pickup_index > len(trip.pickups):
            return False

        if trip.pickup_index < len(trip.pickups):
            if bot.free_slots <= 0:
                return False

        if trip.pickup_index >= len(trip.pickups):
            if not bot.inventory:
                return False
            remaining = list(active.remaining)
            has_useful = False
            for t in bot.inventory:
                if t in remaining:
                    has_useful = True
                    remaining.remove(t)
            if not has_useful:
                return False

        return True

    def _sync_trip_progress(self, bot: BotState) -> None:
        trip = self._trip
        if trip is None:
            return

        inv = Counter(bot.inventory)
        while trip.pickup_index < len(trip.pickups):
            t = trip.pickups[trip.pickup_index].item_type
            baseline = trip.start_inventory[t] + trip.acquired[t]
            if inv[t] > baseline:
                trip.acquired[t] += 1
                trip.pickup_index += 1
                continue
            break

    def _build_best_trip(
        self,
        state: GameState,
        bot: BotState,
        active: Order,
        preview: Order | None,
        mc: MapCache,
        reserved_shelves: set[Pos] | None = None,
        max_pickups: int = 3,
    ) -> TripPlan | None:
        dispensers = self._build_dispensers(state.items)
        if not dispensers:
            return None

        active_need = self._compute_active_need(
            active,
            bot.inventory,
            self._reserved_types,
        )
        preview_need = self._compute_preview_need(preview, bot.inventory, active)

        if bot.free_slots <= 0:
            return TripPlan(
                active_id=active.id,
                preview_id=preview.id if preview else None,
                pickups=[],
                pickup_index=0,
                start_inventory=Counter(bot.inventory),
                acquired=Counter(),
            )

        candidate_loads = self._generate_candidate_loads(
            active_need, preview_need, bot.free_slots, max_pickups=max_pickups
        )
        if not candidate_loads:
            return None

        best_key: tuple[int, int, int, int, int, int, int] | None = None
        best_pickups: list[PickupStep] | None = None

        for load in candidate_loads:
            positions_per_slot = []
            feasible = True
            for t in load:
                opts = dispensers.get(t, [])
                if not opts:
                    feasible = False
                    break
                positions_per_slot.append(opts)
            if not feasible:
                continue

            for chosen_shelves in self._enumerate_shelf_choices(
                positions_per_slot,
                reserved_shelves,
            ):
                n = len(load)
                perm_index_iter = permutations(range(n))
                seen_perms: set[tuple[int, ...]] = set()
                for perm in perm_index_iter:
                    if perm in seen_perms:
                        continue
                    seen_perms.add(perm)

                    ordered = [PickupStep(load[i], chosen_shelves[i]) for i in perm]
                    route_cost = self._trip_route_cost(bot.position, ordered, mc)
                    if route_cost >= INF:
                        continue

                    if route_cost > state.rounds_left:
                        continue

                    active_complete, total_delivered, spillover, next_remaining = (
                        self._score_delivery(
                            bot.inventory,
                            ordered,
                            active,
                            preview,
                        )
                    )
                    score_points = total_delivered + (5 if active_complete else 0)
                    efficiency = score_points * 1000 // max(route_cost, 1)
                    reserved_hits = 0
                    if reserved_shelves:
                        for shelf in chosen_shelves:
                            if shelf in reserved_shelves:
                                reserved_hits += 1
                    key = (
                        1 if active_complete else 0,
                        efficiency,
                        -reserved_hits,
                        spillover,
                        -next_remaining,
                        total_delivered,
                        -route_cost,
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best_pickups = ordered

        if best_pickups is None:
            return None

        return TripPlan(
            active_id=active.id,
            preview_id=preview.id if preview else None,
            pickups=best_pickups,
            pickup_index=0,
            start_inventory=Counter(bot.inventory),
            acquired=Counter(),
        )

    def _build_dispensers(self, items: list[Item]) -> dict[str, list[Pos]]:
        by_type: dict[str, set[Pos]] = {}
        for item in items:
            by_type.setdefault(item.type, set()).add(item.position)
        return {k: sorted(v) for k, v in by_type.items()}

    def _compute_active_need(
        self,
        active: Order,
        inventory: list[str],
        reserved_types: Counter[str] | None = None,
    ) -> list[str]:
        need = list(active.remaining)
        for t in inventory:
            if t in need:
                need.remove(t)
        reserved = Counter(reserved_types or {})
        if reserved:
            filtered: list[str] = []
            for t in need:
                if reserved[t] > 0:
                    reserved[t] -= 1
                    continue
                filtered.append(t)
            need = filtered
        return need

    def _compute_preview_need(
        self, preview: Order | None, inventory: list[str], active: Order
    ) -> list[str]:
        if preview is None:
            return []
        leftover_inv = list(inventory)
        for t in active.remaining:
            if t in leftover_inv:
                leftover_inv.remove(t)
        need = list(preview.items_required)
        for t in leftover_inv:
            if t in need:
                need.remove(t)
        return need

    def _generate_candidate_loads(
        self,
        active_need: list[str],
        preview_need: list[str],
        free_slots: int,
        max_pickups: int = 3,
    ) -> list[tuple[str, ...]]:
        cap = min(max_pickups, free_slots)
        if cap <= 0:
            return []

        active_counts = Counter(active_need)
        preview_counts = Counter(preview_need)

        active_total = sum(active_counts.values())
        preview_total = sum(preview_counts.values())

        out: set[tuple[str, ...]] = set()

        for n in range(1, cap + 1):
            if active_total + preview_total < n:
                continue

            min_active = 1 if active_total > 0 else 0
            max_active = min(n, active_total)
            for a_take in range(min_active, max_active + 1):
                p_take = n - a_take
                if p_take > preview_total:
                    continue

                active_multisets = self._enumerate_type_multisets(active_counts, a_take)
                preview_multisets = self._enumerate_type_multisets(
                    preview_counts, p_take
                )
                if not active_multisets or not preview_multisets:
                    continue

                for am in active_multisets:
                    for pm in preview_multisets:
                        out.add(tuple(am + pm))

        return sorted(out)

    def _enumerate_type_multisets(
        self,
        counts: Counter[str],
        k: int,
    ) -> list[list[str]]:
        if k == 0:
            return [[]]
        if not counts:
            return []

        keys = sorted(counts.keys())
        out: list[list[str]] = []

        def rec(i: int, left: int, acc: list[str]) -> None:
            if left == 0:
                out.append(list(acc))
                return
            if i >= len(keys):
                return

            t = keys[i]
            max_take = min(left, counts[t])
            for c in range(max_take, -1, -1):
                if c > 0:
                    acc.extend([t] * c)
                rec(i + 1, left - c, acc)
                if c > 0:
                    del acc[-c:]

        rec(0, k, [])
        return out

    def _enumerate_shelf_choices(
        self,
        options: list[list[Pos]],
        reserved_shelves: set[Pos] | None = None,
    ) -> list[list[Pos]]:
        out: list[list[Pos]] = []
        reserved = reserved_shelves or set()

        # Filter out reserved shelves (hard exclusion), keep fallback if all reserved
        filtered_options = []
        for opts in options:
            unreserved = [p for p in opts if p not in reserved]
            filtered_options.append(
                sorted(unreserved, key=lambda p: (p[0], p[1]))
                if unreserved
                else sorted(opts, key=lambda p: (p[0], p[1]))
            )

        def rec(i: int, acc: list[Pos]) -> None:
            if i >= len(filtered_options):
                out.append(list(acc))
                return
            for pos in filtered_options[i]:
                acc.append(pos)
                rec(i + 1, acc)
                acc.pop()

        rec(0, [])
        return out

    def _trip_route_cost(
        self, start: Pos, pickups: list[PickupStep], mc: MapCache
    ) -> int:
        if not pickups:
            return 1

        frontier: dict[Pos, int] = {start: 0}

        for step in pickups:
            adj = mc.shelf_adj.get(step.shelf_pos, [])
            if not adj:
                return INF

            next_frontier: dict[Pos, int] = {}
            for cell in adj:
                best = INF
                for prev_cell, prev_cost in frontier.items():
                    d = mc.distance(prev_cell, cell)
                    if d >= INF:
                        continue
                    cand = prev_cost + d
                    if cand < best:
                        best = cand
                if best < INF:
                    next_frontier[cell] = best

            if not next_frontier:
                return INF
            frontier = next_frontier

        best_total = INF
        for pos, move_cost in frontier.items():
            drop_cost = min(mc.distance(pos, dz) for dz in mc.drop_zones)
            if drop_cost >= INF:
                continue
            total = move_cost + len(pickups) + drop_cost + 1
            if total < best_total:
                best_total = total

        return best_total

    def _score_delivery(
        self,
        current_inventory: list[str],
        pickups: list[PickupStep],
        active: Order,
        preview: Order | None,
    ) -> tuple[bool, int, int, int]:
        carried = list(current_inventory) + [p.item_type for p in pickups]

        active_remaining = list(active.remaining)
        delivered_active = 0
        remaining_inventory = list(carried)

        for t in carried:
            if t in active_remaining:
                active_remaining.remove(t)
                delivered_active += 1
                remaining_inventory.remove(t)

        active_complete = len(active_remaining) == 0

        spillover = 0
        next_remaining = 0
        if active_complete and preview is not None:
            preview_remaining = list(preview.items_required)
            for t in remaining_inventory:
                if t in preview_remaining:
                    preview_remaining.remove(t)
                    spillover += 1
            next_remaining = len(preview_remaining)

        return active_complete, delivered_active + spillover, spillover, next_remaining

    def _execute_trip(
        self,
        state: GameState,
        bot: BotState,
        mc: MapCache,
        blocked: set[Pos] | None = None,
        dz_locked: bool = False,
    ) -> dict:
        trip = self._trip
        if trip is None:
            return {"bot": bot.id, "action": "wait"}

        if trip.pickup_index >= len(trip.pickups):
            return self._go_drop(
                bot, mc, state=state, blocked=blocked, dz_locked=dz_locked
            )

        step = trip.pickups[trip.pickup_index]
        if (
            abs(bot.position[0] - step.shelf_pos[0])
            + abs(bot.position[1] - step.shelf_pos[1])
            == 1
        ):
            item_id = self._find_item_id(state.items, step.shelf_pos, step.item_type)
            if item_id is None:
                self._trip = None
                return {"bot": bot.id, "action": "wait"}
            return {"bot": bot.id, "action": "pick_up", "item_id": item_id}

        pickup_cell = self._best_reachable_pickup(
            bot.position, step.shelf_pos, mc, blocked
        )
        if pickup_cell is None:
            self._trip = None
            return {"bot": bot.id, "action": "wait"}

        if bot.position == pickup_cell:
            item_id = self._find_item_id(state.items, step.shelf_pos, step.item_type)
            if item_id is None:
                self._trip = None
                return {"bot": bot.id, "action": "wait"}
            return {"bot": bot.id, "action": "pick_up", "item_id": item_id}

        return self._move_toward(bot, pickup_cell, mc, blocked=blocked)

    def _best_reachable_pickup(
        self,
        bot_pos: Pos,
        shelf_pos: Pos,
        mc: MapCache,
        blocked: set[Pos] | None,
    ) -> Pos | None:
        adj_cells = mc.shelf_adj.get(shelf_pos, [])
        if not adj_cells:
            return None
        blocked_set = blocked or set()
        candidates = sorted(
            adj_cells,
            key=lambda c: (
                c in blocked_set,
                c in mc.narrow_cells,
                mc.distance(bot_pos, c),
            ),
        )
        for cell in candidates:
            if cell not in blocked_set:
                return cell
        return candidates[0] if candidates else None

    def _find_item_id(
        self, items: list[Item], shelf_pos: Pos, item_type: str
    ) -> str | None:
        for item in items:
            if item.position == shelf_pos and item.type == item_type:
                return item.id
        return None

    def _should_force_drop_now(
        self,
        state: GameState,
        bot: BotState,
        active: Order,
        mc: MapCache,
    ) -> bool:
        if not bot.inventory:
            return False
        if not self._inventory_has_active_item(bot, active):
            return False

        dist_drop = mc.distance(bot.position, mc.nearest_drop_zone(bot.position))
        if state.rounds_left <= dist_drop + 1:
            return True

        trip = self._trip
        if trip is None:
            return False

        remaining_pickups = max(0, len(trip.pickups) - trip.pickup_index)
        if remaining_pickups == 0:
            return False

        rough_needed = remaining_pickups + dist_drop + 1
        return state.rounds_left <= rough_needed

    def _inventory_has_active_item(self, bot: BotState, active: Order) -> bool:
        remaining = list(active.remaining)
        for t in bot.inventory:
            if t in remaining:
                return True
        return False

    def _go_drop(
        self,
        bot: BotState,
        mc: MapCache,
        state: GameState | None = None,
        blocked: set[Pos] | None = None,
        dz_locked: bool = False,
    ) -> dict:
        if state and state.active_order:
            if not self._inventory_has_active_item(bot, state.active_order):
                return self._retreat_from_dz_route(bot, mc, blocked=blocked)
        dz = mc.nearest_drop_zone(bot.position)
        if dz_locked:
            bot_count = len(state.bots) if state else 1
            if bot_count >= 12:
                return self._retreat_from_dz_route(bot, mc, blocked=blocked)
            elif bot_count >= 4:
                max_d = min(5 + bot_count, 20)
                return self._hold_near_drop(
                    bot, dz, mc, blocked=blocked, min_dist=5, max_dist=max_d
                )
            else:
                return self._hold_near_drop(bot, dz, mc, blocked=blocked)
        if bot.position == dz:
            return {"bot": bot.id, "action": "drop_off"}
        return self._move_toward(bot, dz, mc, blocked=blocked)

    def _hold_near_drop(
        self,
        bot: BotState,
        dz: Pos,
        mc: MapCache,
        blocked: set[Pos] | None = None,
        min_dist: int = 1,
        max_dist: int = 3,
    ) -> dict:
        dist_now = mc.distance(bot.position, dz)
        if min_dist <= dist_now <= max_dist:
            return {"bot": bot.id, "action": "wait"}

        best_cell: Pos | None = None
        best_score: tuple[int, int] | None = None
        blocked_set = blocked or set()
        for cell in mc.walkable:
            if cell == dz or cell in blocked_set:
                continue
            dd = mc.distance(cell, dz)
            if dd < min_dist or dd > max_dist:
                continue
            db = mc.distance(bot.position, cell)
            if db >= INF:
                continue
            score = (db, dd)
            if best_score is None or score < best_score:
                best_score = score
                best_cell = cell

        if best_cell is None:
            return {"bot": bot.id, "action": "wait"}
        return self._move_toward(bot, best_cell, mc, blocked=blocked)

    def _retreat_from_dz_route(
        self,
        bot: BotState,
        mc: MapCache,
        blocked: set[Pos] | None = None,
    ) -> dict:
        if self._retreat_target is not None:
            if bot.position == self._retreat_target:
                return {"bot": bot.id, "action": "wait"}
            d = mc.distance(bot.position, self._retreat_target)
            if d < 9999:
                return self._move_toward(bot, self._retreat_target, mc, blocked=blocked)

        blocked_set = blocked or set()

        best_cell: Pos | None = None
        best_score: tuple[int, int, int] | None = None
        for cell in mc.walkable:
            if cell in blocked_set or cell in mc.narrow_cells:
                continue
            if cell in mc.drop_zones:
                continue
            min_dz = min(mc.distance(cell, dz) for dz in mc.drop_zones)
            reach = mc.distance(bot.position, cell)
            if reach >= 9999:
                continue
            score = (-min_dz, reach, cell[0])
            if best_score is None or score < best_score:
                best_score = score
                best_cell = cell

        if best_cell is None or best_cell == bot.position:
            return {"bot": bot.id, "action": "wait"}

        self._retreat_target = best_cell
        return self._move_toward(bot, best_cell, mc, blocked=blocked)

    def _move_toward(
        self,
        bot: BotState,
        dst: Pos,
        mc: MapCache,
        blocked: set[Pos] | None = None,
    ) -> dict:
        if blocked:
            step = next_step_toward(mc, bot.position, dst, blocked)
        else:
            step = mc.get_next_hop(bot.position, dst)
        if step is None:
            return {"bot": bot.id, "action": "wait"}
        dx = step[0] - bot.position[0]
        dy = step[1] - bot.position[1]
        return {"bot": bot.id, "action": ACTION_MAP.get((dx, dy), "wait")}
