"""Local game simulator — replay games offline without tokens.

Uses round0_state.json for map layout and order_sequence.json for order queue.
Implements the same game rules as the server: movement, collision, pickup, delivery.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

Pos = tuple[int, int]

DELTAS: dict[str, tuple[int, int]] = {
    "move_up": (0, -1),
    "move_down": (0, 1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
}


class LocalGame:
    __slots__ = (
        "width",
        "height",
        "walls",
        "shelves",
        "items",
        "drop_off",
        "drop_off_zones",
        "bots",
        "max_rounds",
        "round",
        "score",
        "items_delivered",
        "_order_queue",
        "_active",
        "_preview",
        "_order_counter",
    )

    def __init__(self, round0: dict[str, Any], orders: list[dict[str, Any]]) -> None:
        grid = round0["grid"]
        self.width: int = grid["width"]
        self.height: int = grid["height"]
        self.walls: set[Pos] = {(w[0], w[1]) for w in grid["walls"]}
        self.shelves: set[Pos] = {
            (it["position"][0], it["position"][1]) for it in round0["items"]
        }
        self.items: list[dict[str, Any]] = [
            {
                "id": it["id"],
                "type": it["type"],
                "position": (it["position"][0], it["position"][1]),
            }
            for it in round0["items"]
        ]
        self.drop_off: Pos = (round0["drop_off"][0], round0["drop_off"][1])
        self.drop_off_zones: list[Pos] = [
            (z[0], z[1]) for z in round0.get("drop_off_zones", [round0["drop_off"]])
        ]
        self.bots: list[dict[str, Any]] = [
            {
                "id": b["id"],
                "position": (b["position"][0], b["position"][1]),
                "inventory": list(b["inventory"]),
            }
            for b in round0["bots"]
        ]
        self.max_rounds: int = round0["max_rounds"]
        self.round: int = 0
        self.score: int = 0
        self.items_delivered: int = 0

        self._order_queue: list[dict[str, Any]] = []
        self._active: dict[str, Any] | None = None
        self._preview: dict[str, Any] | None = None
        self._order_counter: int = 0

        self._init_orders(round0, orders)

    def _init_orders(
        self, round0: dict[str, Any], orders: list[dict[str, Any]]
    ) -> None:
        seen_ids: set[str] = set()
        for o in round0.get("orders", []):
            entry = {"order_id": o["id"], "items": list(o["items_required"])}
            self._order_queue.append(entry)
            seen_ids.add(o["id"])

        for o in orders:
            if o["order_id"] not in seen_ids:
                self._order_queue.append(
                    {"order_id": o["order_id"], "items": list(o["items"])}
                )
                seen_ids.add(o["order_id"])

        if self._order_queue:
            self._active = self._make_order(self._order_queue.pop(0))
        if self._order_queue:
            self._preview = self._make_order(self._order_queue.pop(0))

    def _make_order(self, entry: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": entry["order_id"],
            "items_required": list(entry["items"]),
            "items_delivered": [],
            "complete": False,
            "status": "active",
        }

    def get_state(self) -> dict[str, Any]:
        orders = []
        if self._active and not self._active["complete"]:
            orders.append({**self._active, "status": "active"})
        if self._preview:
            orders.append({**self._preview, "status": "preview"})

        return {
            "type": "game_state",
            "round": self.round,
            "max_rounds": self.max_rounds,
            "grid": {
                "width": self.width,
                "height": self.height,
                "walls": [list(w) for w in sorted(self.walls)],
            },
            "bots": [
                {
                    "id": b["id"],
                    "position": list(b["position"]),
                    "inventory": list(b["inventory"]),
                }
                for b in self.bots
            ],
            "items": [
                {"id": it["id"], "type": it["type"], "position": list(it["position"])}
                for it in self.items
            ],
            "orders": orders,
            "drop_off": list(self.drop_off),
            "drop_off_zones": [list(z) for z in self.drop_off_zones],
            "score": self.score,
        }

    def step(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        action_map = {a["bot"]: a for a in actions}
        dz_set = set(self.drop_off_zones)

        # 1. Resolve intended positions (movement + stationary)
        intended: dict[int, Pos] = {}
        pickups: dict[int, str] = {}
        dropoffs: set[int] = set()

        for bot in self.bots:
            act = action_map.get(bot["id"], {"action": "wait"})
            action = act.get("action", "wait")

            if action in DELTAS:
                dx, dy = DELTAS[action]
                nx, ny = bot["position"][0] + dx, bot["position"][1] + dy
                target = (nx, ny)
                if (
                    0 <= nx < self.width
                    and 0 <= ny < self.height
                    and target not in self.walls
                    and target not in self.shelves
                ):
                    intended[bot["id"]] = target
                else:
                    intended[bot["id"]] = bot["position"]
            elif action == "pick_up":
                pickups[bot["id"]] = act.get("item_id", "")
                intended[bot["id"]] = bot["position"]
            elif action == "drop_off":
                dropoffs.add(bot["id"])
                intended[bot["id"]] = bot["position"]
            else:
                intended[bot["id"]] = bot["position"]

        # 2. Resolve collisions
        resolved = self._resolve_collisions(intended)

        # 3. Apply positions
        bot_map = {b["id"]: b for b in self.bots}
        for bid, pos in resolved.items():
            bot_map[bid]["position"] = pos

        # 4. Process pickups
        for bid, item_id in pickups.items():
            self._do_pickup(bot_map[bid], item_id)

        # 5. Process dropoffs (with spillover chain)
        for bid in dropoffs:
            bot = bot_map[bid]
            if bot["position"] in dz_set:
                self._do_dropoff_chain(bot)

        # 6. Advance round
        self.round += 1
        if self.round >= self.max_rounds:
            return {
                "type": "game_over",
                "score": self.score,
                "items_delivered": self.items_delivered,
                "rounds_used": self.round,
            }
        return None

    def _resolve_collisions(self, intended: dict[int, Pos]) -> dict[int, Pos]:
        bot_map = {b["id"]: b for b in self.bots}
        resolved = dict(intended)

        # Iteratively resolve until stable
        changed = True
        while changed:
            changed = False

            # Vertex collisions: two+ bots targeting same cell
            target_bots: dict[Pos, list[int]] = {}
            for bid, target in resolved.items():
                target_bots.setdefault(target, []).append(bid)

            for target, bids in target_bots.items():
                if len(bids) < 2:
                    continue
                # Bot already at the cell keeps it; others revert
                staying = [bid for bid in bids if bot_map[bid]["position"] == target]
                if staying:
                    for bid in bids:
                        if bid != staying[0]:
                            old = bot_map[bid]["position"]
                            if resolved[bid] != old:
                                resolved[bid] = old
                                changed = True
                else:
                    for bid in bids:
                        old = bot_map[bid]["position"]
                        if resolved[bid] != old:
                            resolved[bid] = old
                            changed = True

            # Edge collisions: swap detection
            ids = sorted(resolved.keys())
            for i, a_id in enumerate(ids):
                for b_id in ids[i + 1 :]:
                    a_cur = bot_map[a_id]["position"]
                    b_cur = bot_map[b_id]["position"]
                    if resolved[a_id] == b_cur and resolved[b_id] == a_cur:
                        if resolved[a_id] != a_cur:
                            resolved[a_id] = a_cur
                            changed = True
                        if resolved[b_id] != b_cur:
                            resolved[b_id] = b_cur
                            changed = True

        return resolved

    def _do_pickup(self, bot: dict[str, Any], item_id: str) -> None:
        if len(bot["inventory"]) >= 3:
            return
        item = next((it for it in self.items if it["id"] == item_id), None)
        if item is None:
            return
        bx, by = bot["position"]
        ix, iy = item["position"]
        if abs(bx - ix) + abs(by - iy) != 1:
            return
        bot["inventory"].append(item["type"])

    def _do_dropoff_chain(self, bot: dict[str, Any]) -> None:
        while True:
            if not self._active or self._active["complete"]:
                break
            delivered_any = self._deliver_matching(bot)
            if not delivered_any:
                break
            # Check order completion
            remaining = self._order_remaining(self._active)
            if not remaining:
                self._active["complete"] = True
                self.score += 5
                self._advance_orders()

    def _deliver_matching(self, bot: dict[str, Any]) -> bool:
        if not self._active:
            return False
        remaining = self._order_remaining(self._active)
        delivered = False
        new_inv: list[str] = []
        for t in bot["inventory"]:
            if t in remaining:
                remaining.remove(t)
                self._active["items_delivered"].append(t)
                self.score += 1
                self.items_delivered += 1
                delivered = True
            else:
                new_inv.append(t)
        bot["inventory"] = new_inv
        return delivered

    @staticmethod
    def _order_remaining(order: dict[str, Any]) -> list[str]:
        needed = list(order["items_required"])
        for d in order["items_delivered"]:
            if d in needed:
                needed.remove(d)
        return needed

    def _advance_orders(self) -> None:
        self._active = self._preview
        if self._active:
            self._active["status"] = "active"
            self._active["items_delivered"] = []
            self._active["complete"] = False
        self._preview = None
        if self._order_queue:
            self._preview = self._make_order(self._order_queue.pop(0))

    @classmethod
    def from_files(
        cls,
        round0_path: str | Path = "round0_state.json",
        orders_path: str | Path = "order_sequence.json",
    ) -> "LocalGame":
        with open(round0_path) as f:
            round0 = json.load(f)
        with open(orders_path) as f:
            orders = json.load(f)
        return cls(round0, orders)
