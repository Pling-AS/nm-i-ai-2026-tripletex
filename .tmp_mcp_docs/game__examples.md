# Submission Guide

## How to Play

1. Sign in at [app.ainm.no](https://app.ainm.no) with Google
2. Create or join a team
3. Go to the [Challenge page](https://app.ainm.no/challenge)
4. Pick a map and click "Play" to get a WebSocket URL with token
5. Connect your bot to the WebSocket URL
6. Your bot receives game state each round and responds with actions
7. Best score per map is saved automatically

## Rate Limits

- 60 second cooldown between games
- Max 40 games per hour per team
- Max 300 games per day per team

## Example Bot (Python + websockets)

A minimal bot that connects via WebSocket:

```python
import asyncio
import json
import websockets

WS_URL = "wss://game.ainm.no/ws?token=YOUR_TOKEN_HERE"


async def play():
    async with websockets.connect(WS_URL) as ws:
        async for message in ws:
            data = json.loads(message)

            if data["type"] == "game_over":
                print(f"Game over! Score: {data['score']}, Rounds: {data['rounds_used']}")
                break

            if data["type"] == "game_state":
                actions = decide_actions(data)
                await ws.send(json.dumps({"actions": actions}))


def decide_actions(state):
    bots = state["bots"]
    items = state["items"]
    orders = state["orders"]
    drop_off = state["drop_off"]

    actions = []
    for bot in bots:
        action = decide_bot_action(bot, items, orders, drop_off)
        actions.append(action)
    return actions


def decide_bot_action(bot, items, orders, drop_off):
    bx, by = bot["position"]
    inventory = bot["inventory"]

    # Find the active order (status == "active")
    active = next((o for o in orders if o.get("status") == "active" and not o["complete"]), None)
    if not active:
        return {"bot": bot["id"], "action": "wait"}

    # What does the active order still need?
    needed = {}
    for item in active["items_required"]:
        needed[item] = needed.get(item, 0) + 1
    for item in active["items_delivered"]:
        needed[item] = needed.get(item, 0) - 1
    needed = {k: v for k, v in needed.items() if v > 0}

    # If we have useful items and we're at dropoff, deliver
    has_useful = any(needed.get(item, 0) > 0 for item in inventory)
    if has_useful and bx == drop_off[0] and by == drop_off[1]:
        return {"bot": bot["id"], "action": "drop_off"}

    # If inventory full or has useful items, go deliver
    if len(inventory) >= 3 or (has_useful and not needed):
        return navigate_to(bot["id"], bx, by, drop_off[0], drop_off[1])

    # Find nearest needed item
    best_item = None
    best_dist = float("inf")
    for item in items:
        if needed.get(item["type"], 0) > 0:
            ix, iy = item["position"]
            dist = abs(bx - ix) + abs(by - iy)
            if dist < best_dist:
                best_dist = dist
                best_item = item

    if best_item:
        ix, iy = best_item["position"]
        if abs(bx - ix) + abs(by - iy) == 1:
            return {"bot": bot["id"], "action": "pick_up", "item_id": best_item["id"]}
        return navigate_to(bot["id"], bx, by, ix, iy)

    if has_useful:
        return navigate_to(bot["id"], bx, by, drop_off[0], drop_off[1])

    return {"bot": bot["id"], "action": "wait"}


def navigate_to(bot_id, x, y, tx, ty):
    dx = tx - x
    dy = ty - y
    if abs(dx) > abs(dy):
        return {"bot": bot_id, "action": "move_right" if dx > 0 else "move_left"}
    if dy != 0:
        return {"bot": bot_id, "action": "move_down" if dy > 0 else "move_up"}
    if dx != 0:
        return {"bot": bot_id, "action": "move_right" if dx > 0 else "move_left"}
    return {"bot": bot_id, "action": "wait"}


asyncio.run(play())
```

This simple bot treats each bot identically â they all greedily go for the nearest needed item. To improve:

- **Assign roles** â use `bot["id"]` to split bots into different map regions
- **Add pathfinding** â BFS/A* around walls and shelves
- **Coordinate pickups** â track what each bot is targeting to avoid duplication
- **Order prioritization** â focus on nearly-complete orders first

## Deploying Your Bot

Your bot runs **locally** â it connects out to the game server via WebSocket. No hosting required!

If you want to run it on a server:
- Any machine with Python and internet access works
- No HTTPS or public endpoint needed
- The bot is the WebSocket **client**, not server

## Using This MCP Server with Claude

Add this MCP server to Claude Code:

```bash
claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp
```

Then Claude can read the challenge docs and help you build your bot.

## Debugging Tips

- Print the full `state` on the first round to understand the structure
- Track `score` changes between rounds to verify deliveries
- Check if bots are moving (compare positions between rounds)
- Common issues:
  - Moving into walls (check `grid.walls` before moving)
  - Trying to pick up from non-adjacent shelves (must be Manhattan distance 1)
  - Trying to drop off when not on the drop-off cell
  - Not handling `game_over` messages (causes connection errors)
