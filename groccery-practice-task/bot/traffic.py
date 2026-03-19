"""ID-aware collision resolution — plan actions in priority order with swap detection."""

from __future__ import annotations

from bot.assignment import Job, JobType
from bot.map_cache import MapCache
from bot.model import BotState, GameState, Pos
from bot.pathfinding import next_step_toward


ACTION_MAP: dict[tuple[int, int], str] = {
    (0, -1): "move_up",
    (0, 1): "move_down",
    (-1, 0): "move_left",
    (1, 0): "move_right",
}


def resolve_actions(
    state: GameState,
    mc: MapCache,
    assignments: dict[int, Job],
) -> list[dict]:
    bots_by_id = sorted(state.bots, key=lambda b: b.id)

    goal_map: dict[int, Pos | None] = {}
    for bot in bots_by_id:
        job = assignments.get(bot.id)
        goal_map[bot.id] = _job_goal(bot, job, state, mc) if job else None

    def _priority_key(b: BotState) -> tuple[int, int, int]:
        job = assignments.get(b.id)
        if job is None or job.type == JobType.IDLE:
            tier = 3
        elif job.type == JobType.DELIVER:
            tier = 0
        elif job.type == JobType.PICKUP_ACTIVE:
            tier = 1
        else:
            tier = 2
        g = goal_map[b.id]
        dist = mc.distance(b.position, g) if g else 9999
        return (tier, dist, b.id)

    priority_order = sorted(bots_by_id, key=_priority_key)

    corridor_owner: dict[int, int] = _compute_corridor_owners(state.bots, mc)

    reserved_next: set[Pos] = set()
    reserved_by: dict[Pos, int] = {}
    unmoved_positions: set[Pos] = {b.position for b in bots_by_id}
    action_map: dict[int, dict] = {}

    dz_set = set(mc.drop_zones)

    for bot in priority_order:
        unmoved_positions.discard(bot.position)
        job = assignments.get(bot.id)

        must_yield = bot.position in reserved_next

        if job is None or job.type == JobType.IDLE:
            near_dz = bot.position in dz_set or any(
                nb in dz_set for nb in mc.neighbors.get(bot.position, [])
            )
            should_move = must_yield or near_dz
            if should_move:
                side = _find_side_step_away_from_dz(
                    bot, mc, reserved_next, unmoved_positions, dz_set
                )
                if side:
                    action = _make_move(bot, side)
                    reserved_next.add(side)
                    reserved_by[side] = bot.id
                    action_map[bot.id] = action
                    continue
                side = _find_side_step(bot, mc, reserved_next, unmoved_positions)
                if side:
                    action = _make_move(bot, side)
                    reserved_next.add(side)
                    reserved_by[side] = bot.id
                    action_map[bot.id] = action
                    continue
            action = _make_wait(bot)
            reserved_next.add(bot.position)
            reserved_by[bot.position] = bot.id
            action_map[bot.id] = action
            continue

        if must_yield:
            side = _find_side_step(bot, mc, reserved_next, unmoved_positions)
            if side:
                action = _make_move(bot, side)
                reserved_next.add(side)
                reserved_by[side] = bot.id
                action_map[bot.id] = action
                continue
            action = _make_wait(bot)
            reserved_next.add(bot.position)
            reserved_by[bot.position] = bot.id
            action_map[bot.id] = action
            continue

        corridor_blocked = _corridor_blocked_cells(bot, mc, corridor_owner)
        action = _plan_bot_action(
            bot, job, state, mc, reserved_next, unmoved_positions, corridor_blocked
        )
        target = _action_target(bot, action)

        reserved_next.add(target)
        reserved_by[target] = bot.id
        action_map[bot.id] = action

    _resolve_swaps(bots_by_id, action_map, reserved_next, reserved_by, _priority_key)

    return [action_map[b.id] for b in bots_by_id]


def _compute_corridor_owners(bots: list[BotState], mc: MapCache) -> dict[int, int]:
    owners: dict[int, int] = {}
    for bot in bots:
        seg = mc.corridor_id.get(bot.position)
        if seg is not None:
            owners[seg] = bot.id
    return owners


def _corridor_blocked_cells(
    bot: BotState, mc: MapCache, corridor_owner: dict[int, int]
) -> set[Pos]:
    blocked: set[Pos] = set()
    for seg_id, owner_id in corridor_owner.items():
        if owner_id != bot.id:
            blocked |= mc.corridor_segments[seg_id]
    return blocked


def _resolve_swaps(
    bots: list[BotState],
    action_map: dict[int, dict],
    reserved_next: set[Pos],
    reserved_by: dict[Pos, int],
    priority_fn,
) -> None:
    bot_target: dict[int, Pos] = {}
    for bot in bots:
        bot_target[bot.id] = _action_target_from_pos(bot.position, action_map[bot.id])

    for i, a in enumerate(bots):
        t_a = bot_target[a.id]
        if t_a == a.position:
            continue
        for b in bots[i + 1 :]:
            t_b = bot_target[b.id]
            if t_b == b.position:
                continue
            if t_a == b.position and t_b == a.position:
                pa = priority_fn(a)
                pb = priority_fn(b)
                loser = b if pa <= pb else a
                action_map[loser.id] = {"bot": loser.id, "action": "wait"}
                old_target = bot_target[loser.id]
                reserved_next.discard(old_target)
                reserved_next.add(loser.position)
                reserved_by[loser.position] = loser.id
                bot_target[loser.id] = loser.position


def _action_target_from_pos(pos: Pos, action: dict) -> Pos:
    a = action.get("action", "wait")
    x, y = pos
    if a == "move_up":
        return (x, y - 1)
    if a == "move_down":
        return (x, y + 1)
    if a == "move_left":
        return (x - 1, y)
    if a == "move_right":
        return (x + 1, y)
    return (x, y)


def _job_goal(bot: BotState, job: Job, state: GameState, mc: MapCache) -> Pos | None:
    if job.type == JobType.DELIVER:
        return job.drop_zone or mc.nearest_drop_zone(bot.position)
    if job.type in (JobType.PICKUP_ACTIVE, JobType.PICKUP_PREVIEW) and job.item_id:
        item = _find_item(state, job.item_id)
        if item:
            return mc.best_pickup_cell(bot.position, item.position)
    return None


def _find_side_step_away_from_dz(
    bot: BotState,
    mc: MapCache,
    reserved: set[Pos],
    unmoved: set[Pos],
    dz_set: set[Pos],
) -> Pos | None:
    bx, by = bot.position
    best: Pos | None = None
    best_dist = -1
    for nb in mc.neighbors.get(bot.position, []):
        if nb not in reserved and nb not in unmoved and nb not in dz_set:
            min_dz_dist = min(abs(nb[0] - dz[0]) + abs(nb[1] - dz[1]) for dz in dz_set)
            if min_dz_dist > best_dist:
                best_dist = min_dz_dist
                best = nb
    return best


def _find_side_step(
    bot: BotState,
    mc: MapCache,
    reserved: set[Pos],
    unmoved: set[Pos],
) -> Pos | None:
    for nb in mc.neighbors.get(bot.position, []):
        if nb not in reserved and nb not in unmoved:
            return nb
    return None


def _plan_bot_action(
    bot: BotState,
    job: Job,
    state: GameState,
    mc: MapCache,
    reserved: set[Pos],
    unmoved: set[Pos],
    corridor_blocked: set[Pos],
) -> dict:
    x, y = bot.position

    if job.type == JobType.DELIVER:
        if job.drop_zone and (x, y) == job.drop_zone:
            return {"bot": bot.id, "action": "drop_off"}
        goal = job.drop_zone or mc.nearest_drop_zone(bot.position)
        return _move_toward_goal(bot, goal, mc, reserved, unmoved, corridor_blocked)

    if job.type in (JobType.PICKUP_ACTIVE, JobType.PICKUP_PREVIEW):
        if job.item_id:
            item = _find_item(state, job.item_id)
            if item:
                ix, iy = item.position
                if abs(ix - x) + abs(iy - y) == 1:
                    return {"bot": bot.id, "action": "pick_up", "item_id": job.item_id}

                pickup_cell = mc.best_pickup_cell(bot.position, item.position)
                if pickup_cell:
                    if (x, y) == pickup_cell:
                        return {
                            "bot": bot.id,
                            "action": "pick_up",
                            "item_id": job.item_id,
                        }
                    return _move_toward_goal(
                        bot, pickup_cell, mc, reserved, unmoved, corridor_blocked
                    )

    return _make_wait(bot)


def _move_toward_goal(
    bot: BotState,
    goal: Pos,
    mc: MapCache,
    reserved: set[Pos],
    unmoved: set[Pos],
    corridor_blocked: set[Pos],
) -> dict:
    blocked = reserved | unmoved | corridor_blocked
    step = next_step_toward(mc, bot.position, goal, blocked)

    if step is not None and step not in blocked:
        return _make_move(bot, step)

    blocked_no_unmoved = reserved | corridor_blocked
    step = next_step_toward(mc, bot.position, goal, blocked_no_unmoved)
    if step is not None and step not in blocked_no_unmoved:
        return _make_move(bot, step)

    for nb in mc.neighbors.get(bot.position, []):
        if nb not in blocked:
            return _make_move(bot, nb)

    return _make_wait(bot)


def _make_move(bot: BotState, target: Pos) -> dict:
    dx = target[0] - bot.position[0]
    dy = target[1] - bot.position[1]
    action_name = ACTION_MAP.get((dx, dy), "wait")
    return {"bot": bot.id, "action": action_name}


def _make_wait(bot: BotState) -> dict:
    return {"bot": bot.id, "action": "wait"}


def _action_target(bot: BotState, action: dict) -> Pos:
    a = action.get("action", "wait")
    x, y = bot.position
    if a == "move_up":
        return (x, y - 1)
    if a == "move_down":
        return (x, y + 1)
    if a == "move_left":
        return (x - 1, y)
    if a == "move_right":
        return (x + 1, y)
    return (x, y)


def _find_item(state: GameState, item_id: str):
    for item in state.items:
        if item.id == item_id:
            return item
    return None
