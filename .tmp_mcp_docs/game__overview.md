# Grocery Bot Challenge

## What is this?

The Grocery Bot was the pre-competition warm-up challenge. It is not part of the main competition scoring.

- **Task type**: Real-time game (WebSocket)
- **Platform**: [app.ainm.no](https://app.ainm.no)

## How It Works

1. **Pick a map** from the 21 available maps on the Challenge page
2. **Get a WebSocket URL** â click Play to get a game token
3. **Connect your bot** to the WebSocket URL
4. **Receive game state** each round as JSON
5. **Respond with actions** â one per bot (move, pickup, dropoff, or wait)
6. **Best score per map** is saved automatically. Leaderboard = sum of all 21 best scores.

## Difficulty Levels

Bot count and grid size increase with difficulty:

| Level | Bots | Grid | Maps | Rounds | Time Limit |
|-------|------|------|------|--------|------------|
| Easy | 1 | 12Ã10 | 5 | 300 | 2 min |
| Medium | 3 | 16Ã12 | 5 | 300 | 2 min |
| Hard | 5 | 22Ã14 | 5 | 300 | 2 min |
| Expert | 10 | 28Ã18 | 5 | 300 | 2 min |
| Nightmare | 20 | 30Ã18 | 1 | 500 | 5 min |

Nightmare features 3 drop-off zones instead of 1.

## Quick Start

1. Sign in at [app.ainm.no](https://app.ainm.no) with Google
2. Create or join a team
3. Go to the Challenge page, pick a difficulty, click Play
4. Copy the WebSocket URL and connect your bot
5. Play all 21 maps to maximize your leaderboard score

## Key Features

- **WebSocket** â you connect to the game server, not the other way around
- **No fog of war** â full map visible from round 1
- **Bot collision** â bots block each other (no two on same tile, except spawn)
- **Infinite orders** â orders keep generating, rounds are the only limit
- **Daily rotation** â item placement and orders change daily to prevent hardcoding
- **Deterministic within a day** â same map + same day = same game every time
