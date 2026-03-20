# WebSocket Protocol Specification

## Connection

Connect via WebSocket to the URL provided when you request a game token:

```
wss://game.ainm.no/ws?token=<jwt_token>
```

Get a token by clicking "Play" on a map at [app.ainm.no/challenge](https://app.ainm.no/challenge), or by calling the `request_game(map_id)` MCP tool.

## Message Flow

```
Server â Client: {"type": "game_state", ...}     (round 0)
Client â Server: {"actions": [...]}
Server â Client: {"type": "game_state", ...}     (round 1)
Client â Server: {"actions": [...]}
...
Server â Client: {"type": "game_over", ...}       (final)
```

## Game State Message

```json
{
  "type": "game_state",
  "round": 42,
  "max_rounds": 300,
  "action_status": "ok",
  "grid": {
    "width": 14,
    "height": 10,
    "walls": [[1,1], [1,2], [3,1]]
  },
  "bots": [
    {"id": 0, "position": [3, 7], "inventory": ["milk"]},
    {"id": 1, "position": [5, 3], "inventory": []},
    {"id": 2, "position": [10, 7], "inventory": ["bread", "eggs"]}
  ],
  "items": [
    {"id": "item_0", "type": "milk", "position": [2, 1]},
    {"id": "item_1", "type": "bread", "position": [4, 1]}
  ],
  "orders": [
    {
      "id": "order_0",
      "items_required": ["milk", "bread", "eggs"],
      "items_delivered": ["milk"],
      "complete": false,
      "status": "active"
    },
    {
      "id": "order_1",
      "items_required": ["cheese", "butter", "pasta"],
      "items_delivered": [],
      "complete": false,
      "status": "preview"
    }
  ],
  "drop_off": [6, 9],
  "score": 12,
  "active_order_index": 0,
  "total_orders": 8
}
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `round` | int | Current round number (0-indexed) |
| `max_rounds` | int | Maximum rounds (300, or 500 for Nightmare) |
| `action_status` | string | Result of your last action: `"ok"`, `"timeout"`, or `"error"` |
| `grid.width` | int | Grid width in cells |
| `grid.height` | int | Grid height in cells |
| `grid.walls` | int[][] | List of [x, y] wall positions |
| `bots` | object[] | All bots (1-10 depending on difficulty) with id, position [x,y], and inventory |
| `items` | object[] | All items on shelves with id, type, and position [x,y] |
| `orders` | object[] | Only active + preview orders (max 2). Each has `status`: `"active"` or `"preview"` |
| `drop_off` | int[] | [x, y] position of the drop-off zone (EasyâExpert) |
| `drop_off_zones` | int[][] | Array of [x, y] positions (Nightmare only, 3 zones) |
| `score` | int | Current score |
| `active_order_index` | int | Index of the current active order |
| `total_orders` | int | Total number of orders in the game |

### `action_status`

Every `game_state` message includes `action_status`, which tells you whether the server received and processed your last response:

| Value | Meaning |
|-------|---------|
| `"ok"` | Your actions were received and applied normally |
| `"timeout"` | Your bot didn't respond within the 2-second window â all actions were set to `null` (bots waited) |
| `"error"` | Your message was received but couldn't be parsed (invalid JSON, wrong format, etc.) â all actions were set to `null` |

On round 0, `action_status` is always `"ok"` since there was no previous action.

Use this field to detect and debug connectivity or parsing issues. If you see repeated `"timeout"` values, your bot is too slow. If you see `"error"`, check your JSON format.

## Bot Response

Send within **2 seconds** of receiving the game state:

```json
{
  "actions": [
    {"bot": 0, "action": "move_up"},
    {"bot": 1, "action": "pick_up", "item_id": "item_3"},
    {"bot": 2, "action": "drop_off"}
  ]
}
```

### Optional `round` Field

You can include an optional `round` field in your action message to guard against desync:

```json
{
  "round": 42,
  "actions": [
    {"bot": 0, "action": "move_up"},
    {"bot": 1, "action": "pick_up", "item_id": "item_3"},
    {"bot": 2, "action": "drop_off"}
  ]
}
```

If `round` is included and doesn't match the server's current round number, your actions are rejected (treated as if you sent nothing â all bots wait). This is useful for detecting when your bot has fallen out of sync with the server, for example due to network latency causing you to respond to a stale game state.

If you omit the `round` field, the server accepts your actions unconditionally. Including it is recommended but not required.

### Actions

| Action | Extra Fields | Description |
|--------|-------------|-------------|
| `move_up` | â | Move one cell up (y-1) |
| `move_down` | â | Move one cell down (y+1) |
| `move_left` | â | Move one cell left (x-1) |
| `move_right` | â | Move one cell right (x+1) |
| `pick_up` | `item_id` | Pick up item from adjacent shelf |
| `drop_off` | â | Deliver matching items to active order at drop-off zone |
| `wait` | â | Do nothing |

### Move Rules

- Moves to walls, shelves, or out-of-bounds cells fail silently (treated as `wait`)
- Moves to a cell occupied by another bot fail silently (`blocked_by_bot`)
- Actions resolve in **bot ID order** â bot 0 moves first, then bot 1, etc.
- The spawn tile (bottom-right) is exempt from collision â bots can share it at game start

### Pickup Rules

- Bot must be **adjacent** (Manhattan distance 1) to the shelf containing the item
- Bot inventory must not be full (max 3 items)
- `item_id` must match an item on the map

### Dropoff Rules

- Bot must be standing **on** the drop-off cell
- Bot must have items in inventory
- Only items matching the **active order** are delivered â non-matching items **stay in inventory**
- Each delivered item = **+1 point**
- Completed order = **+5 bonus points**
- When the active order completes, the next order activates immediately and remaining items are re-checked

## Game Over Message

When the game ends, the server sends a `game_over` message instead of another `game_state`. This is the final message â the WebSocket closes after this.

```json
{
  "type": "game_over",
  "score": 47,
  "rounds_used": 200,
  "items_delivered": 22,
  "orders_completed": 5
}
```

| Field | Type | Description |
|-------|------|-------------|
| `score` | int | Final score (`items_delivered + orders_completed * 5`) |
| `rounds_used` | int | Number of rounds played before the game ended |
| `items_delivered` | int | Total items delivered across all orders |
| `orders_completed` | int | Number of fully completed orders |

The game ends when any of these conditions is met:
- **Max rounds reached** â 300 rounds (500 for nightmare difficulty)
- **Wall-clock time limit** â 120 seconds (300 for nightmare)
- **Client disconnect** â your WebSocket connection drops

Your bot **must** handle `game_over` messages. Check `data["type"]` before processing â if it's `"game_over"`, print the results and exit cleanly. Failing to handle this will cause your bot to error when it tries to parse the message as a game state.

## Timeouts & Errors

- **2 second** timeout per round for your response
- Timeout â all bots wait (no action), next `action_status` will be `"timeout"`
- Unparseable message â all bots wait, next `action_status` will be `"error"`
- Invalid individual actions â treated as `wait` (but `action_status` is still `"ok"`)
- Disconnect â game ends immediately, score is saved
- **120 second** wall-clock limit per game (300 seconds for nightmare difficulty)

### Coordinate System

- Origin `(0, 0)` is the **top-left** corner
- X increases to the right
- Y increases downward
