# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-18
**Branch:** master (no commits yet)

## OVERVIEW

NM i AI 2026 competition bot — controls warehouse worker swarm in a grocery store via WebSocket. Python 3.13, uv-managed, asyncio + websockets stack.

## STRUCTURE

```
./
├── main.py           # Bot entry point — asyncio loop, WS connection, difficulty routing
├── get_token.py      # Playwright token fetcher — automates app.ainm.no/challenge
├── pyproject.toml    # uv project config — websockets, python-dotenv, scipy, playwright
├── opencode.json     # MCP server config (competition docs)
├── .env              # WS_ENDPOINT_{EASY,MEDIUM,HARD,EXPERT,NIGHTMARE} JWT tokens
├── .game-tracker.json # Rate limit tracking (auto-generated, gitignored)
├── .browser-data/    # Playwright persistent browser state (gitignored)
├── .python-version   # Pins Python 3.13
├── bot/              # Core bot package
│   ├── model.py      # Dataclasses: Item, Order, BotState, GameState + parse_state()
│   ├── planner.py    # Top-level round orchestration
│   ├── assignment.py # Job types + Hungarian solver (scipy)
│   ├── map_cache.py  # All-pairs BFS, neighbor graph, shelf adjacency
│   ├── pathfinding.py # BFS table + A* fallback
│   └── traffic.py    # Collision resolution, reservation sets
└── docs/
    ├── challenge.md  # Full game API spec + example bot (377 lines)
    └── getting-started.md  # Competition onboarding
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Game protocol spec | `docs/challenge.md` | WebSocket JSON schema, actions, scoring, constraints |
| Example bot code | `docs/challenge.md:311-377` | Naive greedy bot — no pathfinding, no collision avoidance |
| Bot entry point | `main.py` | Reads WS_ENDPOINT from .env, runs asyncio game loop |
| Token fetcher | `get_token.py` | Playwright automation — fetches fresh JWT tokens from app.ainm.no |
| Rate limit tracking | `.game-tracker.json` | Auto-generated JSON — tracks game requests for rate limiting |
| WS connection URL | `.env` → `WS_ENDPOINT_EASY` | Full `wss://game.ainm.no/ws?token=<jwt>` URL |
| Competition docs MCP | `opencode.json` | `groccery-bot` MCP at `https://mcp-docs.ainm.no/mcp` |

## GAME MECHANICS (CRITICAL)

### Difficulty Levels

| Level | Grid | Bots | Item Types | Rounds | Drop Zones |
|-------|------|------|-----------|--------|------------|
| Easy | 12x10 | 1 | 4 | 300 | 1 |
| Medium | 16x12 | 3 | 8 | 300 | 1 |
| Hard | 22x14 | 5 | 12 | 300 | 1 |
| Expert | 28x18 | 10 | 16 | 300 | 1 |
| Nightmare | 30x18 | 20 | 21 | 500 | 3 |

### Coordinate System

Origin `(0,0)` = top-left. X right, Y down.

### Actions Per Round

`move_up` (y-1), `move_down` (y+1), `move_left` (x-1), `move_right` (x+1), `pick_up` (+ item_id), `drop_off`, `wait`

### Scoring

+1 per item delivered, +5 bonus per completed order. Leaderboard = sum of best score on each map.

## ANTI-PATTERNS (THIS PROJECT)

- **DO NOT** hardcode WS tokens — read from `.env`
- **DO NOT** commit `.env` — contains live JWT
- **DO NOT** ignore 2-second response timeout — exceeding it = all bots `wait`
- **DO NOT** reconnect after disconnect — game is over, no reconnect allowed
- **DO NOT** spam games — 60s cooldown, max 40/hour, 300/day per team
- **DO NOT** attempt `pick_up` from distance > 1 (Manhattan) — silently fails
- **DO NOT** attempt `drop_off` when not standing on drop-off cell
- **DO NOT** ignore bot collisions — no two bots on same tile (except spawn)
- **DO NOT** deliver to preview orders — only active order accepts deliveries
- **DO NOT** overfill inventory — max 3 items per bot, `pick_up` fails silently when full

## CONVENTIONS

- Package manager: `uv` (no pip, no poetry)
- Entry pattern: `asyncio.run(play())` with `websockets.connect()`
- No linter/formatter/type-checker configured — blank slate
- No tests configured
- No CI/CD

## COMMANDS

```bash
# Run bot
python main.py easy         # or: uv run main.py easy

# Get fresh game token (automated)
python get_token.py easy    # Fetch token for one difficulty
python get_token.py --all   # Fetch tokens for all difficulties
python get_token.py --status # Show rate limit status
python get_token.py --login  # Open browser for first-time auth

# Setup (if fresh clone)
uv sync                     # Install all dependencies
uv run playwright install chromium  # Install browser
```

## NOTES

- Maps are deterministic per day — same day = same item placement and orders
- Nightmare uses `drop_off_zones` array (3 zones) — `drop_off` field gives only the primary
- Non-matching items at drop-off stay in inventory silently (not an error)
- When order completes, next activates immediately and remaining inventory is re-checked
- Preview order items can be pre-picked before they become active
- Example bot in docs uses naive greedy movement (no A*, no wall avoidance) — will get stuck
