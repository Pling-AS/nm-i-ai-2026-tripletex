# Tripletex Accounting Agent

This project is a competition-grade AI accounting agent for the Norwegian Championship in AI. It exposes a single `POST /solve` endpoint, accepts the Tripletex proxy credentials from each request, and uses OpenRouter plus bounded Tripletex tools to complete accounting tasks inside a fresh Tripletex environment.

## Architecture

- FastAPI service with `POST /solve` and `GET /health`
- OpenRouter-powered planner and tool-calling executor
- Local search over the provided `task_docs/api_spec.json` for just-in-time Tripletex endpoint discovery
- Bounded Tripletex API tool layer with clean error handling and compact tool results
- Attachment preprocessing for PDFs, text files, and images
- Optional bearer token protection for your public submission endpoint

## Project layout

```text
src/tripletex_agent/
  main.py
  agent.py
  config.py
  files.py
  openrouter.py
  prompts.py
  schemas.py
  spec_index.py
  trace.py
  tripletex.py
tests/
references/dexter/
```

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -e .
```

3. Copy the env template and fill in your values:

```bash
cp .env.example .env
```

At minimum, set `OPENROUTER_API_KEY`.

Useful optional values:

- `APP_API_KEY` to protect your public submission endpoint
- `OPENROUTER_MODEL` to switch models without code changes
- `AGENT_MAX_STEPS` to control loop depth (recommended: at least `28` for the guarded generic agent)

## Run locally

```bash
uvicorn tripletex_agent.main:app --reload --host 0.0.0.0 --port 8000
```

## Validate locally

```bash
python3 -m compileall src tests
```

If you install dev dependencies, you can also run `pytest`.

## Dashboard

The agent includes a real-time monitoring dashboard accessible at `/dashboard`.

Features:
- **Active Runs**: Watch agent steps, tool calls, and reasoning in real-time.
- **Run History**: View all past runs with detailed traces and error logs.
- **Competition Integration**: If `AINM_JWT_TOKEN` is set, it pulls submission scores and matches them to local runs.
- **Batch Runner**: Trigger multiple competition submissions automatically.

## Endpoint contract

### `POST /solve`


Request body:

```json
{
  "prompt": "Opprett en ansatt med navn Ola Nordmann, ola@example.org. Han skal være kontoadministrator.",
  "files": [],
  "tripletex_credentials": {
    "base_url": "https://tx-proxy.ainm.no/v2",
    "session_token": "abc123"
  }
}
```

Response body:

```json
{
  "status": "completed"
}
```

If `APP_API_KEY` is set, requests must include:

```text
Authorization: Bearer <APP_API_KEY>
```

## Debug traces

Each `/solve` run writes a JSONL trace file to `runs/`.

The trace captures:

- The incoming prompt and attachment summary
- Planner output
- Tool start and tool result events
- Final execution stats or error output

This is useful for inspecting why the agent made certain Tripletex calls while keeping the public API response minimal.

## External references

The `virattt/dexter` repository has been cloned into `references/dexter/` for inspiration and comparison.
That folder is gitignored so it will not pollute your submission repository.

## Deployment

This repository includes a `Dockerfile`, which makes it straightforward to deploy to Render, Railway, Fly.io, or another container-based host. For the competition submission, you need:

- A public HTTPS URL pointing to this app
- The optional bearer token value if you enable endpoint protection

Recommended submission process:

- Deploy the app with your production `OPENROUTER_API_KEY`
- Verify `POST /health` and `POST /solve` on the deployed URL
- Decide whether to set `APP_API_KEY`
- Submit the HTTPS endpoint URL and the optional bearer token

## Current design goal

The implementation is optimized for correctness first, then low-error and low-call execution. The examples in the task folder are treated as loose guidance, not as the target behavior.
