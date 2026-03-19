"""Pathfinding: BFS-table lookup with bounded A* fallback for dynamic obstacles."""

from __future__ import annotations

import heapq

from bot.map_cache import DIRS, MapCache
from bot.model import Pos


def next_step_toward(
    mc: MapCache,
    src: Pos,
    dst: Pos,
    blocked: set[Pos] | None = None,
) -> Pos | None:
    if src == dst:
        return None

    if blocked is None or not blocked:
        return mc.get_next_hop(src, dst)

    return _astar_bounded(mc, src, dst, blocked, max_expansions=200)


def _astar_bounded(
    mc: MapCache,
    start: Pos,
    goal: Pos,
    blocked: set[Pos],
    max_expansions: int = 150,
) -> Pos | None:
    """A* with Manhattan heuristic, respecting dynamic blocked cells.

    Returns the first step on the shortest path, or None if unreachable
    within the expansion budget.
    """
    if start == goal:
        return None

    def h(p: Pos) -> int:
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    # (f, g, pos, first_step)
    open_set: list[tuple[int, int, Pos, Pos | None]] = [(h(start), 0, start, None)]
    visited: set[Pos] = set()
    expansions = 0

    while open_set and expansions < max_expansions:
        f, g, cur, first = heapq.heappop(open_set)

        if cur in visited:
            continue
        visited.add(cur)
        expansions += 1

        if cur == goal:
            return first

        for nb in mc.neighbors.get(cur, []):
            if nb in visited or nb in blocked:
                continue
            step = first if first is not None else nb
            ng = g + 1
            heapq.heappush(open_set, (ng + h(nb), ng, nb, step))

    return None


def path_distance_dynamic(
    mc: MapCache,
    start: Pos,
    goal: Pos,
    blocked: set[Pos],
    max_expansions: int = 200,
) -> int:
    """BFS distance from start to goal avoiding blocked cells."""
    if start == goal:
        return 0

    if not blocked:
        return mc.distance(start, goal)

    # Check if static path is unblocked
    static_d = mc.distance(start, goal)
    if static_d == 9999:
        return 9999

    from collections import deque

    q: deque[tuple[Pos, int]] = deque([(start, 0)])
    visited: set[Pos] = {start}
    expansions = 0

    while q and expansions < max_expansions:
        cur, d = q.popleft()
        expansions += 1
        for nb in mc.neighbors.get(cur, []):
            if nb in visited or nb in blocked:
                continue
            if nb == goal:
                return d + 1
            visited.add(nb)
            q.append((nb, d + 1))

    return 9999
