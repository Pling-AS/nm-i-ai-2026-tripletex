"""PIBT (Priority Inheritance with Backtracking) — collision-free multi-agent movement.

Adapted from Okumura et al. (AIJ 2022) and Kei18/pypibt.
Plans ONE timestep: given current positions + goals, returns collision-free next positions.

Algorithm:
  1. Sort agents by priority (highest first).
  2. For each unplanned agent, call funcPIBT recursively:
     a. Build candidates: [stay] + neighbors, sorted by distance-to-goal.
     b. For each candidate v:
        - Skip if v reserved by another agent (vertex collision).
        - Skip if agent at v is swapping into our position (edge collision).
        - Reserve v. If agent j currently at v hasn't been planned, recursively
          call funcPIBT(j) — this is priority inheritance.
        - If j fails to move, backtrack and try next candidate.
     c. If all candidates fail, agent stays at current position.
"""

from __future__ import annotations

import random

from bot.map_cache import MapCache
from bot.model import Pos

DELTA_TO_MOVE: dict[tuple[int, int], str] = {
    (0, -1): "move_up",
    (0, 1): "move_down",
    (-1, 0): "move_left",
    (1, 0): "move_right",
}


def pibt_step(
    positions: list[Pos],
    goals: list[Pos],
    mc: MapCache,
    locked: set[Pos] | None = None,
    priorities: list[float] | None = None,
    rng: random.Random | None = None,
) -> list[Pos]:
    """Compute collision-free next positions for all agents.

    Args:
        positions: Current position of each agent (indexed 0..N-1).
        goals: Goal position of each agent (indexed 0..N-1).
        mc: MapCache for neighbor lookups and distance queries.
        locked: Cells occupied by stationary agents outside PIBT (impassable).
        priorities: Per-agent priority; higher = planned first.
        rng: Random generator for candidate tie-breaking.

    Returns:
        List of next positions, one per agent. Guaranteed:
        - No two agents at the same cell (vertex collision free).
        - No two agents swapping positions (edge collision free).
        - Locked cells are never assigned.
    """
    n = len(positions)
    if n == 0:
        return []

    if rng is None:
        rng = random.Random(42)
    if priorities is None:
        priorities = [0.0] * n

    locked_set = locked or set()

    # Occupancy tables (dict-based, only populated cells stored)
    occupied_now: dict[Pos, int] = {}  # current: pos -> agent index
    occupied_nxt: dict[Pos, int] = {}  # next:    pos -> agent index

    # Pre-reserve locked cells (stationary external agents)
    for pos in locked_set:
        occupied_nxt[pos] = -1  # sentinel: external lock

    # Mark current positions
    for i, pos in enumerate(positions):
        occupied_now[pos] = i

    # Result array: None = not yet assigned
    q_to: list[Pos | None] = [None] * n

    def func_pibt(i: int) -> bool:
        """Assign collision-free next position for agent i (recursive)."""
        # Candidates: stay in place + walkable neighbors
        candidates = list(mc.neighbors.get(positions[i], []))
        candidates.append(positions[i])  # stay is always an option
        rng.shuffle(candidates)
        candidates.sort(key=lambda v: mc.distance(v, goals[i]))

        for v in candidates:
            # Vertex collision: skip if cell already reserved
            if v in occupied_nxt:
                continue

            # Who currently occupies v?
            j = occupied_now.get(v, -1)

            # Edge collision: skip if j is moving to our current position (swap)
            if j >= 0 and q_to[j] == positions[i]:
                continue

            # Reserve v for agent i
            q_to[i] = v
            occupied_nxt[v] = i

            # Priority inheritance: if j exists at v and hasn't been planned,
            # recursively plan j. If j fails, backtrack (try next candidate).
            # Note: j's failure sets occupied_nxt[v] = j, naturally un-reserving
            # our claim — no explicit cleanup needed.
            if j >= 0 and q_to[j] is None and not func_pibt(j):
                continue

            return True

        # All candidates failed: stay at current position
        q_to[i] = positions[i]
        occupied_nxt[positions[i]] = i
        return False

    # Plan agents in priority order (highest first)
    order = sorted(range(n), key=lambda i: priorities[i], reverse=True)
    for i in order:
        if q_to[i] is None:
            func_pibt(i)

    # Build result (defensive: shouldn't have None, but fallback to current pos)
    result: list[Pos] = []
    for i in range(n):
        p = q_to[i]
        result.append(p if p is not None else positions[i])
    return result


def pos_to_action(bot_id: int, current: Pos, next_pos: Pos) -> dict:
    """Convert a position change to a game action dict."""
    if current == next_pos:
        return {"bot": bot_id, "action": "wait"}
    dx = next_pos[0] - current[0]
    dy = next_pos[1] - current[1]
    move = DELTA_TO_MOVE.get((dx, dy), "wait")
    return {"bot": bot_id, "action": move}
