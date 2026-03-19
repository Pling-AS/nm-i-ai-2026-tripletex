"""Data models for game state parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Pos = tuple[int, int]  # (x, y)


@dataclass(slots=True)
class Item:
    id: str
    type: str
    position: Pos


@dataclass(slots=True)
class Order:
    id: str
    items_required: list[str]
    items_delivered: list[str]
    complete: bool
    status: str  # "active" | "preview"

    @property
    def remaining(self) -> list[str]:
        """Items still needed (accounts for delivered)."""
        needed = list(self.items_required)
        for d in self.items_delivered:
            if d in needed:
                needed.remove(d)
        return needed


@dataclass(slots=True)
class BotState:
    id: int
    position: Pos
    inventory: list[str]

    @property
    def free_slots(self) -> int:
        return 3 - len(self.inventory)

    @property
    def is_full(self) -> bool:
        return len(self.inventory) >= 3


@dataclass(slots=True)
class GameState:
    round: int
    max_rounds: int
    width: int
    height: int
    walls: set[Pos]
    bots: list[BotState]
    items: list[Item]
    orders: list[Order]
    drop_off: Pos
    drop_off_zones: list[Pos]
    score: int

    @property
    def active_order(self) -> Order | None:
        for o in self.orders:
            if o.status == "active":
                return o
        return None

    @property
    def preview_order(self) -> Order | None:
        for o in self.orders:
            if o.status == "preview":
                return o
        return None

    @property
    def rounds_left(self) -> int:
        return self.max_rounds - self.round


def parse_state(msg: dict[str, Any]) -> GameState:
    """Parse raw JSON game_state message into GameState."""
    grid = msg["grid"]
    walls = {(w[0], w[1]) for w in grid["walls"]}

    bots = [
        BotState(
            id=b["id"],
            position=(b["position"][0], b["position"][1]),
            inventory=list(b["inventory"]),
        )
        for b in msg["bots"]
    ]

    items = [
        Item(
            id=it["id"],
            type=it["type"],
            position=(it["position"][0], it["position"][1]),
        )
        for it in msg["items"]
    ]

    orders = [
        Order(
            id=o["id"],
            items_required=list(o["items_required"]),
            items_delivered=list(o["items_delivered"]),
            complete=o["complete"],
            status=o["status"],
        )
        for o in msg["orders"]
    ]

    drop_off = (msg["drop_off"][0], msg["drop_off"][1])
    drop_off_zones = [(z[0], z[1]) for z in msg.get("drop_off_zones", [drop_off])]

    return GameState(
        round=msg["round"],
        max_rounds=msg["max_rounds"],
        width=grid["width"],
        height=grid["height"],
        walls=walls,
        bots=bots,
        items=items,
        orders=orders,
        drop_off=drop_off,
        drop_off_zones=drop_off_zones,
        score=msg["score"],
    )
