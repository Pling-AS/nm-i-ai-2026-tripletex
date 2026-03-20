# Grocery Bot Game Mechanics

## Concept

You control bots navigating a grocery store to fulfill orders sequentially. Pick up items from shelves, deliver them to the drop-off zone, complete orders one at a time for bonus points. Bot count scales by difficulty.

## Store Layout

The store is a rectangular grid with border walls:

- **Floor** (`.`) â walkable cells
- **Walls** (`#`) â impassable barriers (borders + aisle walls)
- **Shelves** â contain items, not walkable. Pick up by standing adjacent.
- **Drop-off** (`D`) â where you deliver items, also walkable

Stores have parallel vertical aisles (shelf-walkway-shelf, 3 cells wide), connected by horizontal corridors at top, bottom, and mid-height.

## 5 Difficulty Levels

| Level | Grid | Bots | Aisles | Item Types | Maps | Rounds | Time Limit |
|-------|------|------|--------|------------|------|--------|------------|
| Easy | 12Ã10 | 1 | 2 | 4 | 5 | 300 | 2 min |
| Medium | 16Ã12 | 3 | 3 | 8 | 5 | 300 | 2 min |
| Hard | 22Ã14 | 5 | 4 | 12 | 5 | 300 | 2 min |
| Expert | 28Ã18 | 10 | 5 | 16 | 5 | 300 | 2 min |
| Nightmare | 30Ã18 | 20 | 6 | 21 | 1 | 500 | 5 min |

21 maps total. Nightmare features 3 drop-off zones instead of 1. Grid structure is fixed per map. **Item placement and orders change daily** (seeded from map_seed + day_of_competition). Same day = same game (deterministic).

## Game Flow

1. All bots start at bottom-right of the store (inside border)
2. Each round, your bot receives the full game state via WebSocket
3. You respond with actions for each bot
4. The game runs for **300 rounds** maximum
5. Wall-clock limit: **120 seconds** per game

## Bots

- **Bot count varies** by difficulty (1, 3, 5, 10, or 20)
- **Inventory capacity**: 3 items per bot
- **Collision**: bots block each other â no two bots can occupy the same tile. Actions resolve in bot ID order (lower IDs move first). Spawn tile is exempt so bots can start stacked.
- **Full visibility**: all items on all shelves are always visible

## Sequential Orders (Infinite)

Orders are revealed **one at a time** and keep generating indefinitely:

- **Active order**: the current order you must complete. Full details visible. You can deliver items for this order.
- **Preview order**: the next order. Full details visible. You CANNOT deliver items for it yet, but you can pre-pick items.
- **Hidden orders**: all remaining orders are not shown.
- **Infinite**: when you complete an order, a new one appears. Orders never run out. Rounds are the only limit.

When the active order is completed:
- The preview order becomes active
- A new order becomes the preview
- Any items in bot inventories that match the new active order are auto-delivered

### Pickup rules
- Bots can pick up **any item** from any shelf, regardless of which order needs it
- Bad picks waste inventory slots â choose wisely

### Dropoff rules
- Only the **active order** can be delivered to
- Items matching the active order are consumed; non-matching items **stay in inventory**
- When the active order completes, the next order activates immediately and remaining items are re-checked

### Order sizes

| Level | Items per Order |
|-------|----------------|
| Easy | 3-4 |
| Medium | 3-5 |
| Hard | 3-5 |
| Expert | 4-6 |
| Nightmare | 4-7 |

## Actions

Each bot can perform one action per round:

| Action | Description |
|--------|-------------|
| `move_up` | Move one cell up |
| `move_down` | Move one cell down |
| `move_left` | Move one cell left |
| `move_right` | Move one cell right |
| `pick_up` | Pick up an item from adjacent shelf (requires `item_id`) |
| `drop_off` | Deliver matching inventory at the drop-off zone |
| `wait` | Do nothing |

Invalid actions are treated as `wait` â no penalty, no error.

## Key Constraints

- **300 rounds** â plan carefully, every round counts
- **3 items per bot** inventory capacity
- **Sequential orders** â complete one before the next activates
- **Infinite orders** â rounds are the only limit
- **No fog of war** â full map visible every round
- **Deterministic per day** â same game every run within a day
- **60s cooldown** between games, max **40/hour** and **300/day** per team
- **Disconnect = game over** â score what you have, no reconnect
