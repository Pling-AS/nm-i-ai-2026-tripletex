"""Job generation and Hungarian assignment for bot-to-task matching."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from scipy.optimize import linear_sum_assignment

from bot.map_cache import MapCache
from bot.model import BotState, GameState, Item, Order, Pos


class JobType(Enum):
    DELIVER = auto()
    PICKUP_ACTIVE = auto()
    PICKUP_PREVIEW = auto()
    IDLE = auto()


@dataclass(slots=True)
class Job:
    type: JobType
    target_pos: Pos  # where the bot needs to go
    item_id: str | None  # for pickup jobs
    item_type: str | None  # type of item to pick/deliver
    drop_zone: Pos | None  # for deliver jobs


def generate_jobs(
    state: GameState,
    mc: MapCache,
    reserved_items: set[str],
) -> list[Job]:
    """Build candidate jobs from current game state.

    Job priority: deliver > pickup_active > pickup_preview.
    Items already reserved by another bot are excluded.
    """
    jobs: list[Job] = []
    active = state.active_order
    preview = state.preview_order

    active_remaining = _remaining_types(active, state, reserved_items) if active else {}
    preview_remaining = (
        _remaining_types(preview, state, reserved_items) if preview else {}
    )

    items_by_type: dict[str, list[Item]] = {}
    for item in state.items:
        if item.id not in reserved_items:
            items_by_type.setdefault(item.type, []).append(item)

    for itype in items_by_type:
        items_by_type[itype].sort(
            key=lambda it: (
                mc.distance(
                    mc.best_pickup_cell(mc.drop_zones[0], it.position)
                    or mc.drop_zones[0],
                    mc.drop_zones[0],
                )
            )
        )

    # Deliver jobs — one per drop zone (assigned via cost to bots carrying active items)
    for dz in mc.drop_zones:
        jobs.append(
            Job(
                type=JobType.DELIVER,
                target_pos=dz,
                item_id=None,
                item_type=None,
                drop_zone=dz,
            )
        )

    # Active pickup jobs — include multiple shelf locations for needed types
    for item_type, count in active_remaining.items():
        available = items_by_type.get(item_type, [])
        for item in available[: max(count, 3)]:
            adj_cells = mc.shelf_adj.get(item.position, [])
            if adj_cells:
                jobs.append(
                    Job(
                        type=JobType.PICKUP_ACTIVE,
                        target_pos=adj_cells[0],
                        item_id=item.id,
                        item_type=item.type,
                        drop_zone=None,
                    )
                )

    # Preview pickup jobs — include multiple shelf locations for needed types
    for item_type, count in preview_remaining.items():
        available = items_by_type.get(item_type, [])
        for item in available[: max(count, 3)]:
            adj_cells = mc.shelf_adj.get(item.position, [])
            if adj_cells:
                jobs.append(
                    Job(
                        type=JobType.PICKUP_PREVIEW,
                        target_pos=adj_cells[0],
                        item_id=item.id,
                        item_type=item.type,
                        drop_zone=None,
                    )
                )

    # Always have an idle job as fallback
    jobs.append(
        Job(
            type=JobType.IDLE,
            target_pos=(0, 0),
            item_id=None,
            item_type=None,
            drop_zone=None,
        )
    )

    return jobs


def assign_jobs(
    bots: list[BotState],
    jobs: list[Job],
    mc: MapCache,
    state: GameState,
) -> dict[int, Job]:
    """Solve bot-to-job assignment via Hungarian algorithm.

    Returns mapping of bot_id -> assigned Job.
    """
    n_bots = len(bots)
    n_jobs = len(jobs)

    if n_bots == 0 or n_jobs == 0:
        return {}

    # Cost matrix: bots x jobs
    cost = np.full((n_bots, n_jobs), 10000.0)

    for i, bot in enumerate(bots):
        for j, job in enumerate(jobs):
            cost[i, j] = _job_cost(bot, job, mc, state)

    # Pad if more bots than jobs
    if n_bots > n_jobs:
        pad = np.full((n_bots, n_bots - n_jobs), 10000.0)
        cost = np.hstack([cost, pad])

    row_ind, col_ind = linear_sum_assignment(cost)

    assignment: dict[int, Job] = {}
    for r, c in zip(row_ind, col_ind):
        if c < n_jobs:
            assignment[bots[r].id] = jobs[c]
        else:
            assignment[bots[r].id] = Job(
                type=JobType.IDLE,
                target_pos=bots[r].position,
                item_id=None,
                item_type=None,
                drop_zone=None,
            )

    return assignment


def _job_cost(bot: BotState, job: Job, mc: MapCache, state: GameState) -> float:
    if job.type == JobType.IDLE:
        return 5000.0

    if job.type == JobType.DELIVER:
        if not bot.inventory:
            return 9000.0
        active = state.active_order
        if not active:
            return 9000.0
        active_rem = list(active.remaining)
        active_count = 0
        for inv_item in bot.inventory:
            if inv_item in active_rem:
                active_count += 1
                active_rem.remove(inv_item)
        if active_count == 0:
            return 9000.0
        dist_to_drop = mc.distance(bot.position, job.target_pos)
        reward = active_count * 3.0
        remaining_after = len(active_rem)
        if remaining_after == 0:
            reward += 15.0
        if bot.is_full:
            reward += 5.0
        elif bot.free_slots > 0 and remaining_after > 0:
            penalty = bot.free_slots * 10.0
            return dist_to_drop * 1.0 + penalty
        return max(0.0, dist_to_drop * 1.0 - reward)

    if job.type == JobType.PICKUP_ACTIVE:
        if bot.is_full:
            return 8000.0
        pickup_cell = mc.best_pickup_cell(bot.position, _item_pos_from_job(job, state))
        if pickup_cell is None:
            return 8500.0
        dist_to_pickup = mc.distance(bot.position, pickup_cell)
        dz = mc.nearest_drop_zone(pickup_cell)
        dist_pickup_to_drop = mc.distance(pickup_cell, dz)
        drop_weight = 0.3 + len(bot.inventory) * 0.2
        cost = dist_to_pickup * 1.0 + dist_pickup_to_drop * drop_weight
        return max(0.0, cost)

    if job.type == JobType.PICKUP_PREVIEW:
        if bot.is_full:
            return 8000.0
        if bot.free_slots <= 1:
            return 7000.0
        if job.item_type and bot.inventory.count(job.item_type) >= 1:
            return 7500.0
        pickup_cell = mc.best_pickup_cell(bot.position, _item_pos_from_job(job, state))
        if pickup_cell is None:
            return 8500.0
        dist_to_pickup = mc.distance(bot.position, pickup_cell)
        dist_pickup_to_drop = mc.distance(
            pickup_cell, mc.nearest_drop_zone(pickup_cell)
        )
        active = state.active_order
        if active:
            active_need = list(active.remaining)
            for b in state.bots:
                for inv in b.inventory:
                    if inv in active_need:
                        active_need.remove(inv)
            unmet_active = len(active_need)
            if unmet_active == 0:
                penalty = 2.0
            elif bot.free_slots > unmet_active:
                penalty = 8.0
            else:
                penalty = 40.0
        else:
            penalty = 2.0
        return dist_to_pickup * 1.2 + dist_pickup_to_drop * 0.3 + penalty

    return 10000.0


def _remaining_types(
    order: Order | None,
    state: GameState,
    reserved_items: set[str],
) -> dict[str, int]:
    """Count remaining needed item types for an order, excluding reserved + carried."""
    if order is None:
        return {}

    needed = list(order.remaining)

    # Subtract items already being carried by any bot
    for bot in state.bots:
        for inv_item in bot.inventory:
            if inv_item in needed:
                needed.remove(inv_item)

    # Subtract reserved items by type
    reserved_types: list[str] = []
    for item in state.items:
        if item.id in reserved_items:
            reserved_types.append(item.type)

    for rt in reserved_types:
        if rt in needed:
            needed.remove(rt)

    counts: dict[str, int] = {}
    for t in needed:
        counts[t] = counts.get(t, 0) + 1
    return counts


def _item_pos_from_job(job: Job, state: GameState) -> Pos:
    """Get shelf position of the item referenced by a pickup job."""
    if job.item_id:
        for item in state.items:
            if item.id == job.item_id:
                return item.position
    return (0, 0)
