# Grocery Bot Scoring

## Score Formula

Per game:
```
score = items_delivered Ã 1 + orders_completed Ã 5
```

- **+1 point** for each item delivered to the drop-off
- **+5 bonus** for completing an entire order (all required items delivered)

## Leaderboard

Your **leaderboard score** is the **sum of your best scores across all 21 maps**.

- Play each map as many times as you want (60s cooldown, 40/hour, 300/day)
- Only your highest score per map is saved
- Deterministic within a day â same algorithm = same score
- To maximize your rank: get good scores on ALL 21 maps

## Daily Rotation

Item placement on shelves and order contents change daily at midnight UTC. The grid structure (walls, shelf positions) stays the same. This prevents hardcoding solutions while keeping games deterministic within a single day.

## Infinite Orders

Orders never run out. When you complete the active order, the next one activates and a new preview appears. The only limit is the **300 round** cap. Score as much as you can before time runs out.

## Score Examples

| Scenario | Items | Orders | Score |
|----------|-------|--------|-------|
| Delivered 3 items, no complete orders | 3 | 0 | 3 |
| Delivered 4 items, completed 1 order | 4 | 1 | 9 |
| Delivered 15 items, completed 3 orders | 15 | 3 | 30 |
| Delivered 50 items, completed 10 orders | 50 | 10 | 100 |

## Game End Conditions

| Condition | Description |
|-----------|-------------|
| Max rounds | 300 rounds (500 for Nightmare) |
| Wall-clock timeout | 120 seconds (300 seconds for Nightmare) |
| Disconnect | Client disconnected |

