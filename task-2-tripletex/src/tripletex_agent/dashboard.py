"""Monitoring dashboard for Tripletex agent runs."""

import asyncio
import json
import logging
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

from tripletex_agent.config import get_settings
from tripletex_agent.prompts import EXECUTOR_SYSTEM_PROMPT
from tripletex_agent.trace import RunTrace, RUNS_DIR

logger = logging.getLogger(__name__)

# --- Cached competition submissions for run matching ---
_submissions_cache: list[dict[str, Any]] = []
_submissions_cache_ts: float = 0


async def _get_cached_submissions() -> list[dict[str, Any]]:
    """Fetch and merge competition submissions from both API endpoints.

    /tasks/{task_id}/submissions — has precise `started_at` (when our endpoint was called)
    /tripletex/my/submissions   — has `feedback` with detailed check results
    """
    global _submissions_cache, _submissions_cache_ts
    import time

    now = time.monotonic()
    active_count = len(RunTrace.get_active_runs())
    has_pending = any(
        s.get("status") in ("in_progress", "pending") for s in _submissions_cache
    )
    ttl = 15.0 if (active_count > 0 or has_pending) else 120.0

    if _submissions_cache and (now - _submissions_cache_ts) < ttl:
        return _submissions_cache

    settings = get_settings()
    token = settings.ainm_jwt_token
    if not token:
        return _submissions_cache

    try:
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(timeout=10) as client:
            # Fetch both endpoints in parallel
            task_id = settings.ainm_tripletex_task_id
            tasks_url = f"https://api.ainm.no/tasks/{task_id}/submissions"
            my_url = "https://api.ainm.no/tripletex/my/submissions"

            tasks_resp, my_resp = await asyncio.gather(
                client.get(tasks_url, headers=headers),
                client.get(my_url, headers=headers),
                return_exceptions=True,
            )

            # Parse /tasks/ endpoint (has started_at)
            tasks_data: dict[str, dict[str, Any]] = {}
            if isinstance(tasks_resp, httpx.Response) and tasks_resp.status_code == 200:
                for sub in tasks_resp.json():
                    tasks_data[sub["id"]] = sub

            # Parse /my/submissions endpoint (has feedback/checks)
            my_data: list[dict[str, Any]] = []
            if isinstance(my_resp, httpx.Response) and my_resp.status_code == 200:
                my_data = my_resp.json()

            # Merge: use /my/ as base (has feedback), enrich with /tasks/ (has started_at)
            merged: list[dict[str, Any]] = []
            for sub in my_data:
                sid = sub.get("id", "")
                task_sub = tasks_data.get(sid, {})
                sub["platform_started_at"] = task_sub.get("started_at")
                sub["endpoint_url"] = task_sub.get("endpoint_url")
                merged.append(sub)

            _submissions_cache = merged
            _submissions_cache_ts = now
    except Exception as exc:
        logger.debug("Submissions cache refresh failed: %s", exc)

    return _submissions_cache


def _match_submission_to_run(
    run_started_at: str,
    run_duration: float | None,
    submissions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Match a local run to a competition submission by timestamp.

    Uses platform's `started_at` (when our /solve was called) for tight matching.
    Falls back to `queued_at` if started_at unavailable.
    Window: 15 seconds (platform calls us within ms of queuing).
    """
    if not run_started_at or not submissions:
        return None
    try:
        run_ts = datetime.fromisoformat(run_started_at)
    except ValueError:
        return None

    best_match = None
    best_delta = 15.0  # tight 15s window

    for sub in submissions:
        # Prefer platform_started_at (exact moment our endpoint was called)
        ts_str = sub.get("platform_started_at") or sub.get("queued_at")
        if not ts_str:
            continue
        try:
            sub_ts = datetime.fromisoformat(ts_str)
        except ValueError:
            continue
        delta = abs((sub_ts - run_ts).total_seconds())
        if delta < best_delta:
            best_delta = delta
            best_match = sub

    return best_match


router = APIRouter()


def _parse_trace_file(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    except (json.JSONDecodeError, OSError):
        pass
    return events


def _summarize_run(path: Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    filename = path.name
    run_id = (
        filename.rsplit("_", 1)[-1].replace(".jsonl", "")
        if "_" in filename
        else filename
    )

    init_event = next((e for e in events if e["event_type"] == "init"), None)
    planner_event = next((e for e in events if e["event_type"] == "planner"), None)
    execution_brief_event = next(
        (e for e in events if e["event_type"] == "execution_brief"), None
    )
    done_event = next((e for e in events if e["event_type"] == "done"), None)
    error_event = next((e for e in events if e["event_type"] == "error"), None)
    final_event = next((e for e in events if e["event_type"] == "final_payload"), None)

    prompt = ""
    base_url = ""
    file_count = 0
    metadata: dict[str, Any] = {}
    if init_event:
        payload = init_event.get("payload", {})
        prompt = payload.get("prompt", "")
        base_url = payload.get("base_url", "")
        file_count = payload.get("file_count", 0)
        metadata = payload.get("metadata", {})

    # Merge any later metadata_update events (e.g., routing decisions after planner)
    for event in events:
        if event.get("event_type") == "metadata_update":
            metadata.update(event.get("payload", {}))

    task_type = ""
    goal = ""
    planner_payload: dict[str, Any] = {}
    if planner_event:
        payload = planner_event.get("payload", {})
        task_type = payload.get("task_type", "")
        goal = payload.get("goal", "")
        planner_payload = payload

    execution_brief: dict[str, Any] = {}
    if execution_brief_event:
        execution_brief = execution_brief_event.get("payload", {})

    # Determine status
    status = "unknown"
    if done_event:
        status = "completed"
    elif error_event:
        status = "error"
    elif run_id in RunTrace._active:
        status = "running"
    elif events:
        last_event = events[-1]["event_type"]
        status = "error" if last_event == "error" else "incomplete"

    # Stats
    call_count = 0
    error_count = 0
    call_log: list[dict[str, Any]] = []
    if done_event:
        payload = done_event.get("payload", {})
        call_count = payload.get("tripletex_call_count", 0)
        error_count = payload.get("tripletex_error_count", 0)
        call_log = payload.get("tripletex_call_log", [])

    # Duration
    duration_seconds: float | None = None
    if events:
        try:
            first_ts = datetime.fromisoformat(events[0]["timestamp"])
            actual_run_events = [
                e
                for e in events
                if e["event_type"] not in ("competition_scoring", "error_summary")
            ]
            if actual_run_events:
                last_ts = datetime.fromisoformat(actual_run_events[-1]["timestamp"])
                duration_seconds = round((last_ts - first_ts).total_seconds(), 1)
        except (KeyError, ValueError):
            pass

    # Source
    source = metadata.get("source", "")
    if not source:
        source = "competition" if "tx-proxy" in base_url else "simulation"

    # Tool call counts
    tool_starts = [e for e in events if e["event_type"] == "tool_start"]
    tool_errors = [
        e
        for e in events
        if e["event_type"] == "tool_result"
        and not e.get("payload", {}).get("result", {}).get("ok", True)
    ]

    # Preflight rejections (schema_validation_errors in tool_result)
    preflight_rejections: list[dict[str, Any]] = []
    for e in events:
        if e["event_type"] == "tool_result":
            result = e.get("payload", {}).get("result", {})
            if result.get("schema_validation_errors"):
                preflight_rejections.append(
                    {
                        "error": result.get("error", ""),
                        "fields": [
                            p.get("path", "")
                            for p in result["schema_validation_errors"]
                        ],
                        "tool_name": e.get("payload", {}).get("tool_name", ""),
                    }
                )

    # Auto-stripped fields (successful calls where fields were removed)
    auto_stripped: list[dict[str, Any]] = []
    for e in events:
        if e["event_type"] == "tool_result":
            result = e.get("payload", {}).get("result", {})
            if result.get("ok") and result.get("stripped_fields"):
                auto_stripped.append(
                    {
                        "fields": result["stripped_fields"],
                        "tool_name": e.get("payload", {}).get("tool_name", ""),
                    }
                )

    # Summary text from final_payload
    summary_text = ""
    if final_event:
        summary_text = final_event.get("payload", {}).get("summary", "")

    error_message = ""
    if error_event:
        error_message = error_event.get("payload", {}).get("message", "")

    enforcer_rejections = [
        e.get("payload", {}) for e in events if e["event_type"] == "enforcer_rejected"
    ]
    semantic_rejections = [
        e.get("payload", {})
        for e in events
        if e["event_type"] == "semantic_enforcer_rejected"
    ]
    enforcer_overrides = [
        e.get("payload", {}) for e in events if e["event_type"] == "enforcer_override"
    ]

    return {
        "run_id": run_id,
        "filename": filename,
        "started_at": events[0]["timestamp"] if events else "",
        "prompt": prompt,
        "task_type": task_type,
        "goal": goal,
        "status": status,
        "source": source,
        "duration_seconds": duration_seconds,
        "tripletex_call_count": call_count,
        "tripletex_error_count": error_count,
        "tripletex_call_log": call_log,
        "tool_call_count": len(tool_starts),
        "tool_error_count": len(tool_errors),
        "file_count": file_count,
        "summary": summary_text,
        "error_message": error_message,
        "metadata": metadata,
        "event_count": len(events),
        "preflight_rejections": preflight_rejections,
        "preflight_rejection_count": len(preflight_rejections),
        "auto_stripped": auto_stripped,
        "enforcer_rejections": enforcer_rejections,
        "semantic_rejections": semantic_rejections,
        "enforcer_overrides": enforcer_overrides,
        "planner_payload": planner_payload,
        "execution_brief": execution_brief,
        "executor_system_prompt": EXECUTOR_SYSTEM_PROMPT,
    }


@router.get("/api/runs")
async def list_runs() -> JSONResponse:
    runs: list[dict[str, Any]] = []
    if RUNS_DIR.exists():
        for path in sorted(RUNS_DIR.glob("*.jsonl"), reverse=True):
            if path.name == "raw_requests.jsonl":
                continue
            events = _parse_trace_file(path)
            if events:
                runs.append(_summarize_run(path, events))

    # Also add active runs not yet on disk
    for run_id, trace in RunTrace.get_active_runs().items():
        if not any(r["run_id"] == run_id for r in runs):
            events = _parse_trace_file(trace.path) if trace.path.exists() else []
            summary = (
                _summarize_run(trace.path, events)
                if events
                else {
                    "run_id": run_id,
                    "filename": trace.path.name,
                    "started_at": trace.started_at.isoformat(),
                    "prompt": trace.prompt,
                    "task_type": "",
                    "goal": "",
                    "status": "running",
                    "source": trace.metadata.get("source", "unknown"),
                    "duration_seconds": round(
                        (datetime.now(UTC) - trace.started_at).total_seconds(), 1
                    ),
                    "tripletex_call_count": 0,
                    "tripletex_error_count": 0,
                    "tripletex_call_log": [],
                    "tool_call_count": 0,
                    "tool_error_count": 0,
                    "file_count": 0,
                    "summary": "",
                    "error_message": "",
                    "metadata": trace.metadata,
                    "event_count": 0,
                }
            )
            summary["status"] = "running"
            runs.insert(0, summary)

    # Match competition runs with submission scores
    submissions = await _get_cached_submissions()
    used_sub_ids: set[str] = set()
    for run in runs:
        if run.get("source") != "competition":
            run["competition_score"] = None
            continue
        match = _match_submission_to_run(
            run["started_at"], run.get("duration_seconds"), submissions
        )
        if match and match.get("id") not in used_sub_ids:
            used_sub_ids.add(match["id"])
            feedback = match.get("feedback", {})
            checks = feedback.get("checks", [])
            passed = sum(1 for c in checks if "passed" in c.lower())
            run["competition_score"] = {
                "score_raw": match.get("score_raw", 0),
                "score_max": match.get("score_max", 0),
                "normalized_score": match.get("normalized_score", 0),
                "checks_passed": passed,
                "checks_total": len(checks),
                "comment": feedback.get("comment", ""),
                "checks": checks,
                "submission_id": match["id"],
                "status": match.get("status", "unknown"),
            }
        else:
            if "competition_score" not in run:
                run["competition_score"] = None

    active_count = len(RunTrace.get_active_runs())
    has_pending = any(
        s.get("status") in ("in_progress", "pending") for s in submissions
    )
    return JSONResponse(
        {
            "runs": runs,
            "active_count": active_count,
            "total_count": len(runs),
            "has_pending_submissions": has_pending,
        }
    )


@router.post("/api/runs/enrich")
async def enrich_runs() -> JSONResponse:
    """Enrich completed runs with competition scoring and error summaries."""
    submissions = await _get_cached_submissions()
    enriched_count = 0
    skipped_count = 0

    if RUNS_DIR.exists():
        for path in sorted(RUNS_DIR.glob("*.jsonl"), reverse=True):
            if path.name == "raw_requests.jsonl":
                continue
            events = _parse_trace_file(path)
            if not events:
                continue
            result = _enrich_run_file(path, events, submissions)
            if result is not None:
                enriched_count += 1
            else:
                skipped_count += 1

    overview = _generate_overview()

    return JSONResponse(
        {
            "enriched": enriched_count,
            "skipped": skipped_count,
            "overview_runs": overview.get("total_runs", 0),
        }
    )


@router.get("/api/runs/overview")
async def runs_overview() -> JSONResponse:
    """Return the overview JSON for all runs."""
    overview_path = RUNS_DIR / "overview.json"
    if overview_path.exists():
        try:
            with overview_path.open("r", encoding="utf-8") as f:
                return JSONResponse(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    # Generate fresh if missing
    overview = _generate_overview()
    return JSONResponse(overview)


@router.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> JSONResponse:
    if RUNS_DIR.exists():
        for path in RUNS_DIR.glob("*.jsonl"):
            if run_id in path.name:
                events = _parse_trace_file(path)
                summary = _summarize_run(path, events)
                # Attach competition score if available
                if summary.get("source") == "competition":
                    submissions = await _get_cached_submissions()
                    match = _match_submission_to_run(
                        summary["started_at"],
                        summary.get("duration_seconds"),
                        submissions,
                    )
                    if match:
                        feedback = match.get("feedback", {})
                        checks = feedback.get("checks", [])
                        passed = sum(1 for c in checks if "passed" in c.lower())
                        summary["competition_score"] = {
                            "score_raw": match.get("score_raw", 0),
                            "score_max": match.get("score_max", 0),
                            "normalized_score": match.get("normalized_score", 0),
                            "checks_passed": passed,
                            "checks_total": len(checks),
                            "comment": feedback.get("comment", ""),
                            "checks": checks,
                            "submission_id": match["id"],
                            "status": match.get("status", "unknown"),
                        }
                return JSONResponse(
                    {
                        "summary": summary,
                        "events": events,
                    }
                )
    return JSONResponse({"error": "Run not found"}, status_code=404)


@router.get("/api/settings")
async def get_settings_endpoint() -> JSONResponse:
    s = get_settings()
    return JSONResponse(
        {
            "model": s.openrouter_model,
            "planner_model": s.planner_model,
            "tier1_executor_model": s.tier1_executor_model,
            "tier2_executor_model": s.tier2_executor_model,
            "tier3_executor_model": s.tier3_executor_model,
            "max_steps": s.agent_max_steps,
            "temperature": s.agent_model_temperature,
            "http_timeout": s.http_timeout_seconds,
            "max_attachment_chars": s.max_attachment_text_chars,
            "log_level": s.log_level,
        }
    )


@router.get("/api/competition/submissions")
async def competition_submissions() -> JSONResponse:
    """Proxy to api.ainm.no to fetch our Tripletex submission scores."""
    settings = get_settings()
    token = settings.ainm_jwt_token
    if not token:
        return JSONResponse(
            {"error": "AINM_JWT_TOKEN not configured", "submissions": []}
        )

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://api.ainm.no/tripletex/my/submissions",
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
            submissions = resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch competition submissions: %s", exc)
        return JSONResponse({"error": str(exc), "submissions": []})

    # Compute summary stats
    completed = [s for s in submissions if s.get("status") == "completed"]
    best_normalized = max((s.get("normalized_score", 0) for s in completed), default=0)
    best_raw = max(
        (
            s.get("score_raw", 0)
            for s in completed
            if s.get("normalized_score", 0) == best_normalized
        ),
        default=0,
    )
    total_checks_passed = 0
    total_checks = 0
    perfect_runs = 0
    for s in completed:
        checks = s.get("feedback", {}).get("checks", [])
        passed = sum(1 for c in checks if "passed" in c.lower())
        total_checks_passed += passed
        total_checks += len(checks)
        if passed == len(checks) and len(checks) > 0:
            perfect_runs += 1

    return JSONResponse(
        {
            "submissions": submissions,
            "summary": {
                "total": len(submissions),
                "completed": len(completed),
                "best_normalized_score": best_normalized,
                "best_raw_score": best_raw,
                "perfect_runs": perfect_runs,
                "total_checks": total_checks,
                "total_checks_passed": total_checks_passed,
                "check_pass_rate": round(total_checks_passed / total_checks, 3)
                if total_checks > 0
                else 0,
            },
        }
    )


@router.post("/api/competition/submit")
async def competition_submit(request_body: dict | None = None) -> JSONResponse:
    """Trigger a new competition submission via the ainm.no API.

    Captures the submission_id for deterministic run matching.
    Body: {"endpoint_url": "https://...", "endpoint_api_key": null}
    """
    settings = get_settings()
    token = settings.ainm_jwt_token
    task_id = settings.ainm_tripletex_task_id
    if not token:
        return JSONResponse({"error": "AINM_JWT_TOKEN not configured"}, status_code=400)

    # Default endpoint URL from settings
    body = request_body or {}
    endpoint_url = body.get("endpoint_url", settings.local_solve_url)
    endpoint_api_key = body.get("endpoint_api_key", settings.app_api_key)

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"https://api.ainm.no/tasks/{task_id}/submissions",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "endpoint_url": endpoint_url,
                    "endpoint_api_key": endpoint_api_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        return JSONResponse(
            {"error": f"API returned {exc.response.status_code}: {exc.response.text}"},
            status_code=exc.response.status_code,
        )
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    # Invalidate cache so next poll picks up the new submission
    global _submissions_cache_ts
    _submissions_cache_ts = 0

    return JSONResponse(
        {
            "submission_id": data.get("id"),
            "status": data.get("status"),
            "daily_submissions_used": data.get("daily_submissions_used"),
            "daily_submissions_max": data.get("daily_submissions_max"),
        }
    )


_batch_runner_active = False
_batch_runner_results: list[dict[str, Any]] = []
_batch_runner_progress: dict[str, Any] = {
    "total": 0,
    "completed": 0,
    "current": None,
    "status": "idle",
}


@router.post("/api/competition/batch")
async def competition_batch(request_body: dict | None = None) -> JSONResponse:
    global _batch_runner_active, _batch_runner_results, _batch_runner_progress
    body = request_body or {}
    count = min(int(body.get("count", 5)), 50)
    delay = int(body.get("delay_seconds", 5))

    if _batch_runner_active:
        return JSONResponse(
            {
                "error": "Batch runner already active",
                "progress": _batch_runner_progress,
            },
            status_code=409,
        )

    import asyncio

    asyncio.create_task(_run_batch(count, delay))
    return JSONResponse({"started": True, "count": count, "delay_seconds": delay})


@router.get("/api/competition/batch/status")
async def competition_batch_status() -> JSONResponse:
    return JSONResponse(
        {
            "active": _batch_runner_active,
            "progress": _batch_runner_progress,
            "results": _batch_runner_results[-20:],
        }
    )


@router.post("/api/competition/batch/stop")
async def competition_batch_stop() -> JSONResponse:
    global _batch_runner_active
    _batch_runner_active = False
    return JSONResponse({"stopped": True, "progress": _batch_runner_progress})


async def _run_batch(count: int, delay: int) -> None:
    global _batch_runner_active, _batch_runner_results, _batch_runner_progress
    _batch_runner_active = True
    _batch_runner_results = []
    _batch_runner_progress = {
        "total": count,
        "completed": 0,
        "current": None,
        "status": "running",
    }

    settings = get_settings()
    token = settings.ainm_jwt_token
    task_id = settings.ainm_tripletex_task_id
    endpoint_api_key = settings.app_api_key

    endpoint_url = None
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            ngrok_resp = await c.get("http://localhost:4040/api/tunnels")
            tunnels = ngrok_resp.json().get("tunnels", [])
            for t in tunnels:
                if "ngrok" in t.get("public_url", ""):
                    endpoint_url = t["public_url"] + "/solve"
                    break
    except Exception:
        pass
    if not endpoint_url:
        endpoint_url = settings.local_solve_url

    for i in range(count):
        if not _batch_runner_active:
            _batch_runner_progress["status"] = "stopped"
            break

        _batch_runner_progress["current"] = i + 1
        _batch_runner_progress["status"] = f"submitting {i + 1}/{count}"

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"https://api.ainm.no/tasks/{task_id}/submissions",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "endpoint_url": endpoint_url,
                        "endpoint_api_key": endpoint_api_key,
                    },
                )
                resp.raise_for_status()
                sub_data = resp.json()
                sub_id = sub_data.get("id", "?")

            global _submissions_cache_ts
            _submissions_cache_ts = 0

            _batch_runner_progress["status"] = (
                f"waiting for run {i + 1}/{count} (submission {sub_id[:8]})"
            )

            run_completed = False
            for _ in range(120):
                await asyncio.sleep(5)
                if not _batch_runner_active:
                    break
                latest_files = sorted(
                    RUNS_DIR.glob("*.jsonl"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for f in latest_files[:3]:
                    if f.name == "raw_requests.jsonl":
                        continue
                    events = _parse_trace_file(f)
                    done_event = next(
                        (e for e in events if e["event_type"] == "done"), None
                    )
                    if done_event:
                        init_event = next(
                            (e for e in events if e["event_type"] == "init"), None
                        )
                        src = (
                            (init_event or {})
                            .get("payload", {})
                            .get("metadata", {})
                            .get("source", "")
                        )
                        if src == "competition":
                            dp = done_event["payload"]
                            _batch_runner_results.append(
                                {
                                    "run": i + 1,
                                    "file": f.name,
                                    "calls": dp.get("tripletex_call_count", 0),
                                    "errors": dp.get("tripletex_error_count", 0),
                                    "submission_id": sub_id[:12],
                                }
                            )
                            run_completed = True
                            break
                if run_completed:
                    break

            if not run_completed and _batch_runner_active:
                _batch_runner_results.append(
                    {
                        "run": i + 1,
                        "submission_id": sub_id[:12],
                        "status": "timeout",
                    }
                )

        except Exception as exc:
            _batch_runner_results.append(
                {
                    "run": i + 1,
                    "error": str(exc)[:200],
                }
            )

        _batch_runner_progress["completed"] = i + 1

        if i < count - 1 and _batch_runner_active:
            _batch_runner_progress["status"] = (
                f"waiting {delay}s before next submission"
            )
            await asyncio.sleep(delay)

    _batch_runner_progress["status"] = (
        "completed" if _batch_runner_active else "stopped"
    )
    _batch_runner_active = False


def _enrich_run_file(
    path: Path,
    events: list[dict[str, Any]],
    submissions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Append competition_scoring and error_summary events to a run JSONL if missing.

    Returns the enrichment data if newly added, None if already enriched or not applicable.
    """
    # Skip if already enriched
    existing_types = {e["event_type"] for e in events}
    if "competition_scoring" in existing_types:
        return None

    summary = _summarize_run(path, events)

    # Only enrich completed competition runs
    if summary["source"] != "competition" or summary["status"] not in (
        "completed",
        "error",
    ):
        return None

    # Match to submission
    match = _match_submission_to_run(
        summary["started_at"], summary.get("duration_seconds"), submissions
    )

    # Collect all errors from tool_result events
    errors: list[dict[str, Any]] = []
    for e in events:
        if e["event_type"] == "tool_result":
            result = e.get("payload", {}).get("result", {})
            if not result.get("ok", True):
                errors.append(
                    {
                        "tool_name": e.get("payload", {}).get("tool_name", ""),
                        "error": result.get("error", ""),
                        "status_code": result.get("status_code"),
                        "validation_summary": result.get("validation_summary"),
                    }
                )
        elif e["event_type"] == "error":
            errors.append(
                {
                    "tool_name": "agent",
                    "error": e.get("payload", {}).get("message", ""),
                }
            )

    scoring_data: dict[str, Any] = {}
    if match:
        feedback = match.get("feedback", {})
        checks = feedback.get("checks", [])
        passed = sum(1 for c in checks if "passed" in c.lower())
        scoring_data = {
            "submission_id": match.get("id"),
            "status": match.get("status"),
            "score_raw": match.get("score_raw", 0),
            "score_max": match.get("score_max", 0),
            "normalized_score": match.get("normalized_score", 0),
            "checks_passed": passed,
            "checks_total": len(checks),
            "checks": checks,
            "comment": feedback.get("comment", ""),
        }
    else:
        scoring_data = {"status": "no_match", "comment": "No matching submission found"}

    # Append events to JSONL
    from datetime import datetime, UTC

    now = datetime.now(UTC).isoformat()
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": now,
                    "event_type": "competition_scoring",
                    "payload": scoring_data,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        if errors:
            f.write(
                json.dumps(
                    {
                        "timestamp": now,
                        "event_type": "error_summary",
                        "payload": {
                            "total_errors": len(errors),
                            "errors": errors,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {"scoring": scoring_data, "error_count": len(errors)}


def _generate_overview() -> dict[str, Any]:
    """Generate overview.json with one entry per run."""
    from datetime import datetime, UTC

    runs_data: list[dict[str, Any]] = []
    if RUNS_DIR.exists():
        for path in sorted(RUNS_DIR.glob("*.jsonl"), reverse=True):
            if path.name == "raw_requests.jsonl":
                continue
            events = _parse_trace_file(path)
            if not events:
                continue

            summary = _summarize_run(path, events)

            # Extract scoring from competition_scoring event if present
            scoring_event = next(
                (e for e in events if e["event_type"] == "competition_scoring"), None
            )
            scoring = scoring_event.get("payload", {}) if scoring_event else {}

            # Extract error summary from error_summary event if present
            error_event = next(
                (e for e in events if e["event_type"] == "error_summary"), None
            )
            error_data = error_event.get("payload", {}) if error_event else {}

            runs_data.append(
                {
                    "filename": path.name,
                    "run_id": summary["run_id"],
                    "started_at": summary["started_at"],
                    "task_type": summary["task_type"],
                    "task_tier": summary.get("metadata", {}).get("task_tier"),
                    "executor_model": summary.get("metadata", {}).get(
                        "executor_model", ""
                    ),
                    "status": summary["status"],
                    "source": summary["source"],
                    "duration_seconds": summary["duration_seconds"],
                    "api_calls": summary["tripletex_call_count"],
                    "api_errors": summary["tripletex_error_count"],
                    "tool_calls": summary["tool_call_count"],
                    "preflight_rejections": summary.get("preflight_rejection_count", 0),
                    "score_raw": scoring.get("score_raw"),
                    "score_max": scoring.get("score_max"),
                    "normalized_score": scoring.get("normalized_score"),
                    "checks_passed": scoring.get("checks_passed"),
                    "checks_total": scoring.get("checks_total"),
                    "checks": scoring.get("checks", []),
                    "total_errors": error_data.get("total_errors", 0),
                    "errors": [
                        e.get("error", "") for e in error_data.get("errors", [])
                    ],
                    "prompt_preview": summary["prompt"][:120],
                }
            )

    overview = {
        "runs": runs_data,
        "total_runs": len(runs_data),
        "generated_at": datetime.now(UTC).isoformat(),
    }

    # Write to disk
    overview_path = RUNS_DIR / "overview.json"
    with overview_path.open("w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    return overview


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    return HTMLResponse(DASHBOARD_HTML)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tripletex Agent Monitor</title>
<style>
:root {
  --bg: #0f1117;
  --surface: #1a1d27;
  --surface2: #242736;
  --border: #2e3245;
  --text: #e1e4ed;
  --text2: #8b90a0;
  --green: #34d399;
  --red: #f87171;
  --yellow: #fbbf24;
  --blue: #60a5fa;
  --purple: #a78bfa;
  --orange: #fb923c;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
header {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 12px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
}
header h1 { font-size: 16px; font-weight: 600; }
header h1 span { color: var(--blue); }
.header-meta {
  display: flex;
  gap: 20px;
  font-size: 13px;
  color: var(--text2);
}
.header-meta .value { color: var(--text); font-weight: 500; }
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 11px;
  font-weight: 600;
}
.badge-green { background: rgba(52,211,153,.15); color: var(--green); }
.badge-red { background: rgba(248,113,113,.15); color: var(--red); }
.badge-yellow { background: rgba(251,191,36,.15); color: var(--yellow); }
.badge-blue { background: rgba(96,165,250,.15); color: var(--blue); }
.badge-gray { background: rgba(139,144,160,.15); color: var(--text2); }

main {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* Left panel - run list */
.run-list {
  width: 420px;
  min-width: 420px;
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.run-list-header {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 12px;
  flex-shrink: 0;
}
.run-list-header input {
  width: 100%;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 10px;
  color: var(--text);
  font-size: 13px;
  outline: none;
}
.run-list-header input:focus { border-color: var(--blue); }
.run-list-header-filters {
  display: flex;
  gap: 8px;
  align-items: center;
  width: 100%;
}
.filter-btn {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 6px 10px;
  color: var(--text2);
  font-size: 12px;
  cursor: pointer;
}
.filter-btn:hover, .filter-btn.active { border-color: var(--blue); color: var(--text); }
.run-list-body {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}
.run-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  margin-bottom: 6px;
  cursor: pointer;
  transition: border-color .15s;
}
.run-card:hover { border-color: var(--blue); }
.run-card.selected { border-color: var(--blue); background: var(--surface2); }
.run-card-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}
.run-card-type { font-size: 13px; font-weight: 600; }
.run-card-time { font-size: 11px; color: var(--text2); }
.run-card-prompt {
  font-size: 12px;
  color: var(--text2);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-bottom: 6px;
}
.run-card-stats {
  display: flex;
  gap: 12px;
  font-size: 11px;
  color: var(--text2);
}
.run-card-stats .stat-value { color: var(--text); font-weight: 500; }
.stat-err { color: var(--red) !important; }

/* Right panel - detail */
.run-detail {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}
.run-detail-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--text2);
  font-size: 14px;
}
.detail-header {
  margin-bottom: 20px;
}
.detail-header h2 {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 8px;
}
.detail-prompt {
  background: var(--surface2);
  border-radius: 8px;
  padding: 12px 16px;
  font-size: 13px;
  line-height: 1.5;
  margin-bottom: 12px;
  white-space: pre-wrap;
  word-break: break-word;
}
.detail-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 10px;
  margin-bottom: 20px;
}
.detail-stat {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 14px;
}
.detail-stat-label { font-size: 11px; color: var(--text2); margin-bottom: 2px; }
.detail-stat-value { font-size: 16px; font-weight: 600; }

/* Timeline */
.timeline { margin-top: 16px; }
.timeline h3 { font-size: 14px; margin-bottom: 12px; color: var(--text2); }
.tl-event {
  display: flex;
  gap: 12px;
  padding: 8px 0;
  border-bottom: 1px solid var(--border);
  font-size: 13px;
  cursor: pointer;
}
.tl-event:hover { background: var(--surface2); }
.tl-time { color: var(--text2); font-family: 'SF Mono', 'Menlo', monospace; font-size: 11px; min-width: 80px; padding-top: 2px; }
.tl-badge { min-width: 120px; }
.tl-detail {
  flex: 1;
  color: var(--text2);
  font-size: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.tl-payload {
  display: none;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px;
  margin: 6px 0 6px 92px;
  font-family: 'SF Mono', 'Menlo', monospace;
  font-size: 11px;
  line-height: 1.4;
  max-height: 400px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-all;
}
.tl-payload.open { display: block; }

/* Call log */
.call-log { margin-top: 20px; }
.call-log h3 { font-size: 14px; margin-bottom: 10px; color: var(--text2); }
.call-log-entry {
  display: flex;
  gap: 10px;
  padding: 5px 0;
  font-size: 12px;
  font-family: 'SF Mono', 'Menlo', monospace;
}
.call-method { font-weight: 600; min-width: 55px; }
.call-path { color: var(--text); flex: 1; }
.call-status { min-width: 40px; text-align: right; }
.call-2xx { color: var(--green); }
.call-4xx { color: var(--red); }
.call-5xx { color: var(--orange); }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text2); }

/* Refresh indicator */
.refresh-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--green);
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: .4; }
}

/* Header flash animations */
@keyframes flashStart { 0% { background: #2a1f0f; box-shadow: 0 2px 12px rgba(251,191,36,.25); } 100% { background: var(--surface); box-shadow: none; } }
@keyframes flashDone { 0% { background: #0f2a1a; box-shadow: 0 2px 12px rgba(52,211,153,.25); } 100% { background: var(--surface); box-shadow: none; } }
@keyframes flashError { 0% { background: #2a0f0f; box-shadow: 0 2px 12px rgba(248,113,113,.25); } 100% { background: var(--surface); box-shadow: none; } }
header.flash-start { animation: flashStart 1.5s ease-out; }
header.flash-done { animation: flashDone 1.5s ease-out; }
header.flash-error { animation: flashError 1.5s ease-out; }

/* Sound toggle */
#sound-toggle {
  cursor: pointer;
  font-size: 14px;
  user-select: none;
  padding: 2px 6px;
  border-radius: 4px;
  transition: background .15s;
}
#sound-toggle:hover { background: var(--surface2); }
.competition-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 12px; }
.competition-card-top { display: flex; justify-content: space-between; margin-bottom: 8px; }
.check-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.check-pill { font-size: 11px; padding: 2px 8px; border-radius: 10px; background: var(--surface2); color: var(--text2); border: 1px solid transparent; }
.check-passed { background: rgba(52,211,153,.15); color: var(--green); border-color: rgba(52,211,153,.2); }
.check-failed { background: rgba(248,113,113,.15); color: var(--red); border-color: rgba(248,113,113,.2); }
</style>
</head>
<body>

<header>
  <h1><span>&#9670;</span> Tripletex Agent Monitor</h1>
  <div class="header-meta">
    <div>Planner: <span class="value" id="hdr-planner">-</span></div>
    <div>T1: <span class="value" id="hdr-t1">-</span></div>
    <div>T2: <span class="value" id="hdr-t2">-</span></div>
    <div>T3: <span class="value" id="hdr-t3">-</span></div>
    <div id="hdr-best-score-container" style="display:none;color:var(--yellow);font-weight:600;">&#127942; Best: <span id="hdr-best-score">-</span></div>
    <div>Active: <span class="value" id="hdr-active">0</span></div>
    <div>Total: <span class="value" id="hdr-total">0</span></div>
    <div class="refresh-dot" title="Auto-refreshing"></div>
    <div id="sound-toggle" title="Toggle sounds" onclick="toggleSound()">&#x1F50A;</div>
  </div>
</header>

<main>
  <div class="run-list">
    <div class="run-list-header">
      <input type="text" id="search" placeholder="Search prompts...">
      <div class="run-list-header-filters">
        <button class="filter-btn active" data-filter="all">All</button>
        <button class="filter-btn" data-filter="completed">&#x2714;</button>
        <button class="filter-btn" data-filter="error">&#x2716;</button>
        <button class="filter-btn" data-filter="running">&#x25cf;</button>
        <button class="filter-btn" id="btn-scores" style="margin-left:auto;color:var(--yellow);">&#127942; Scores</button>
        <button class="filter-btn" data-filter="raw" id="btn-raw" style="color:var(--orange);">&#x26A0; Raw</button>
      </div>
    </div>
    <div class="run-list-body" id="run-list-body"></div>
  </div>
  <div class="run-detail" id="run-detail">
    <div class="run-detail-empty">Select a run to view details</div>
  </div>
</main>

<script>
let allRuns = [];
let selectedRunId = null;
let activeFilter = 'all';
let previousRunStatuses = {};
let soundEnabled = true;
let lastCompetitionRefresh = 0;
let competitionViewActive = false;
let lastSubmissionCount = 0;
let globalHasPendingSubmissions = false;
let previousScoreStatuses = {};

// --- Web Audio API Sound Engine ---
const AudioCtx = window.AudioContext || window.webkitAudioContext;
let audioCtx = null;
function getAudioCtx() {
  if (!audioCtx) audioCtx = new AudioCtx();
  return audioCtx;
}

function toggleSound() {
  soundEnabled = !soundEnabled;
  document.getElementById('sound-toggle').innerHTML = soundEnabled ? '&#x1F50A;' : '&#x1F507;';
}

function playStartSound() {
  if (!soundEnabled) return;
  try {
    const ctx = getAudioCtx();
    const now = ctx.currentTime;

    // Layer 1: Deep dramatic bass sweep (the "BWOOM")
    const bass = ctx.createOscillator();
    const bassGain = ctx.createGain();
    bass.type = 'sawtooth';
    bass.frequency.setValueAtTime(90, now);
    bass.frequency.exponentialRampToValueAtTime(35, now + 0.8);
    bassGain.gain.setValueAtTime(0, now);
    bassGain.gain.linearRampToValueAtTime(0.25, now + 0.04);
    bassGain.gain.exponentialRampToValueAtTime(0.001, now + 0.8);
    bass.connect(bassGain);
    bassGain.connect(ctx.destination);
    bass.start(now);
    bass.stop(now + 0.85);

    // Layer 2: Eerie minor third (Bb3 — unsettling)
    const mid = ctx.createOscillator();
    const midGain = ctx.createGain();
    mid.type = 'sine';
    mid.frequency.setValueAtTime(233, now + 0.03);
    midGain.gain.setValueAtTime(0, now);
    midGain.gain.linearRampToValueAtTime(0.12, now + 0.06);
    midGain.gain.exponentialRampToValueAtTime(0.001, now + 0.6);
    mid.connect(midGain);
    midGain.connect(ctx.destination);
    mid.start(now + 0.03);
    mid.stop(now + 0.65);

    // Layer 3: High descending whistle (eerie)
    const high = ctx.createOscillator();
    const highGain = ctx.createGain();
    high.type = 'sine';
    high.frequency.setValueAtTime(900, now + 0.05);
    high.frequency.exponentialRampToValueAtTime(400, now + 0.5);
    highGain.gain.setValueAtTime(0, now);
    highGain.gain.linearRampToValueAtTime(0.06, now + 0.08);
    highGain.gain.exponentialRampToValueAtTime(0.001, now + 0.5);
    high.connect(highGain);
    highGain.connect(ctx.destination);
    high.start(now + 0.05);
    high.stop(now + 0.55);
  } catch (e) { /* audio not available */ }
}

function playCompletionSound() {
  if (!soundEnabled) return;
  try {
    const ctx = getAudioCtx();
    const now = ctx.currentTime;

    // Note 1: A5 (880Hz) — bright base
    const n1 = ctx.createOscillator();
    const g1 = ctx.createGain();
    n1.type = 'sine';
    n1.frequency.setValueAtTime(880, now);
    g1.gain.setValueAtTime(0.3, now);
    g1.gain.exponentialRampToValueAtTime(0.001, now + 0.5);
    n1.connect(g1);
    g1.connect(ctx.destination);
    n1.start(now);
    n1.stop(now + 0.55);

    // Note 2: D6 (1175Hz) — ascending fourth, delayed
    const n2 = ctx.createOscillator();
    const g2 = ctx.createGain();
    n2.type = 'sine';
    n2.frequency.setValueAtTime(1175, now + 0.12);
    g2.gain.setValueAtTime(0, now);
    g2.gain.setValueAtTime(0.25, now + 0.12);
    g2.gain.exponentialRampToValueAtTime(0.001, now + 0.65);
    n2.connect(g2);
    g2.connect(ctx.destination);
    n2.start(now + 0.12);
    n2.stop(now + 0.7);

    // Shimmer: A6 (1760Hz) — harmonic sparkle
    const n3 = ctx.createOscillator();
    const g3 = ctx.createGain();
    n3.type = 'sine';
    n3.frequency.setValueAtTime(1760, now + 0.12);
    g3.gain.setValueAtTime(0, now);
    g3.gain.setValueAtTime(0.07, now + 0.12);
    g3.gain.exponentialRampToValueAtTime(0.001, now + 0.55);
    n3.connect(g3);
    g3.connect(ctx.destination);
    n3.start(now + 0.12);
    n3.stop(now + 0.6);
  } catch (e) { /* audio not available */ }
}

function playErrorSound() {
  if (!soundEnabled) return;
  try {
    const ctx = getAudioCtx();
    const now = ctx.currentTime;

    // Descending buzz
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'square';
    osc.frequency.setValueAtTime(220, now);
    osc.frequency.exponentialRampToValueAtTime(80, now + 0.35);
    gain.gain.setValueAtTime(0.15, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(now);
    osc.stop(now + 0.4);

    // Second shorter buzz
    const osc2 = ctx.createOscillator();
    const gain2 = ctx.createGain();
    osc2.type = 'square';
    osc2.frequency.setValueAtTime(180, now + 0.2);
    osc2.frequency.exponentialRampToValueAtTime(60, now + 0.45);
    gain2.gain.setValueAtTime(0, now);
    gain2.gain.setValueAtTime(0.12, now + 0.2);
    gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.45);
    osc2.connect(gain2);
    gain2.connect(ctx.destination);
    osc2.start(now + 0.2);
    osc2.stop(now + 0.5);
  } catch (e) { /* audio not available */ }
}

function flashHeader(type) {
  const h = document.querySelector('header');
  h.classList.remove('flash-start', 'flash-done', 'flash-error');
  void h.offsetWidth; // force reflow to restart animation
  h.classList.add('flash-' + type);
  setTimeout(() => h.classList.remove('flash-' + type), 1500);
}

const statusBadge = (status) => {
  const map = {
    completed: ['badge-green', '&#x2714; completed'],
    error: ['badge-red', '&#x2716; error'],
    running: ['badge-yellow', '&#x25cf; running'],
    incomplete: ['badge-gray', '? incomplete'],
    unknown: ['badge-gray', '? unknown'],
  };
  const [cls, text] = map[status] || map.unknown;
  return `<span class="badge ${cls}">${text}</span>`;
};

const sourceBadge = (source) => {
  if (source === 'competition') return `<span class="badge badge-purple" style="background:rgba(167,139,250,.15);color:#a78bfa;">COMP</span>`;
  return `<span class="badge badge-blue">SIM</span>`;
};

const eventBadge = (type) => {
  const map = {
    init: 'badge-blue',
    planner: 'badge-purple',
    tool_start: 'badge-gray',
    tool_result: 'badge-gray',
    final_payload: 'badge-green',
    done: 'badge-green',
    error: 'badge-red',
    blocked_warning: 'badge-yellow',
    recovery: 'badge-yellow',
    enforcer_rejected: 'badge-red',
    semantic_enforcer_rejected: 'badge-red',
    enforcer_override: 'badge-yellow',
    enforcer_passed: 'badge-green',
    thinking: 'badge-purple',
    assistant_reasoning: 'badge-blue',
    api_advisor_query: 'badge-blue',
    api_advisor_response: 'badge-green',
    execution_brief: 'badge-blue',
    competition_scoring: 'badge-green',
    error_summary: 'badge-red',
    attachments_prepared: 'badge-gray',
    metadata_update: 'badge-blue',
  };
  return `<span class="badge ${map[type] || 'badge-gray'}" style="${type === 'planner' ? 'background:rgba(167,139,250,.15);color:#a78bfa;' : ''}">${type}</span>`;
};

const fmtTime = (ts) => {
  if (!ts) return '-';
  try { return new Date(ts).toLocaleTimeString('nb-NO', {hour:'2-digit',minute:'2-digit',second:'2-digit'}); }
  catch { return ts.split('T')[1]?.slice(0,8) || ts; }
};

const fmtDuration = (s) => {
  if (s == null) return '-';
  if (s < 60) return s.toFixed(1) + 's';
  return Math.floor(s/60) + 'm ' + Math.round(s%60) + 's';
};

const truncate = (s, n) => s && s.length > n ? s.slice(0,n) + '...' : (s || '');

function renderRunList() {
  const search = document.getElementById('search').value.toLowerCase();
  const body = document.getElementById('run-list-body');
  const filtered = allRuns.filter(r => {
    if (activeFilter !== 'all' && r.status !== activeFilter) return false;
    if (search && !r.prompt.toLowerCase().includes(search) && !r.task_type.toLowerCase().includes(search)) return false;
    return true;
  });
  body.innerHTML = filtered.map(r => `
    <div class="run-card ${r.run_id === selectedRunId ? 'selected' : ''}" onclick="selectRun('${r.run_id}')">
      <div class="run-card-top">
        <span class="run-card-type">${r.task_type || '(pending)'}</span>
        <span class="run-card-time">${fmtTime(r.started_at)}</span>
      </div>
      <div style="display:flex;gap:6px;align-items:center;margin-bottom:4px;">
        ${statusBadge(r.status)}
        ${sourceBadge(r.source)}
      </div>
      <div class="run-card-prompt">${truncate(r.prompt, 80)}</div>
      ${r.competition_score ? (r.competition_score.status === 'in_progress' || r.competition_score.status === 'pending' || r.status === 'running' ? 
        `<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
           <span class="badge badge-yellow" style="font-size:11px;">Evaluating...</span>
         </div>` :
        `<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
          <span class="badge" title="Leaderboard Points" style="background:rgba(251,191,36,.15);color:var(--yellow);font-size:12px;font-weight:700;">&#127942; ${(r.competition_score.normalized_score || 0).toFixed(2)} pts</span>
          <span style="font-size:11px;color:var(--text2);">Checks: ${r.competition_score.score_raw}/${r.competition_score.score_max} pts (${r.competition_score.checks_passed}/${r.competition_score.checks_total})</span>
          ${r.competition_score.checks_total > 0 && r.competition_score.checks_passed === r.competition_score.checks_total ? '<span style="color:var(--green);font-size:11px;font-weight:600;">PERFECT</span>' : ''}
        </div>`) : ''}
      <div class="run-card-stats">
        <span>API: <span class="stat-value">${r.tripletex_call_count}</span></span>
        <span>Errors: <span class="stat-value ${r.tripletex_error_count > 0 ? 'stat-err' : ''}">${r.tripletex_error_count}</span></span>
        ${r.preflight_rejection_count > 0 ? `<span>Preflight: <span class="stat-value" style="color:var(--orange)">${r.preflight_rejection_count}</span></span>` : ''}
        ${(r.enforcer_rejections || []).length > 0 ? `<span>🛡️ <span class="stat-value" style="color:var(--red)">${r.enforcer_rejections.length}</span></span>` : ''}
        ${(r.semantic_rejections || []).length > 0 ? `<span>🧠 <span class="stat-value" style="color:var(--red)">${r.semantic_rejections.length}</span></span>` : ''}
        ${(r.enforcer_overrides || []).length > 0 ? `<span>⚡ <span class="stat-value" style="color:var(--yellow)">${r.enforcer_overrides.length}</span></span>` : ''}
        <span>Tools: <span class="stat-value">${r.tool_call_count}</span></span>
        <span>Time: <span class="stat-value">${fmtDuration(r.duration_seconds)}</span></span>
      </div>
    </div>
  `).join('');
}

async function selectRun(runId) {
  selectedRunId = runId;
  competitionViewActive = false;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  renderRunList();
  const panel = document.getElementById('run-detail');
  panel.innerHTML = '<div class="run-detail-empty">Loading...</div>';
  try {
    const res = await fetch(`/api/runs/${runId}`);
    const data = await res.json();
    if (data.error) { panel.innerHTML = `<div class="run-detail-empty">${data.error}</div>`; return; }
    renderDetail(data);
  } catch (e) {
    panel.innerHTML = `<div class="run-detail-empty">Error: ${e.message}</div>`;
  }
}

function renderDetail(data) {
  const { summary: s, events } = data;
  const panel = document.getElementById('run-detail');

  let callLogHtml = '';
  if (s.tripletex_call_log && s.tripletex_call_log.length > 0) {
    callLogHtml = `<div class="call-log"><h3>API Call Log</h3>${
      s.tripletex_call_log.map(c => {
        const sc = c.status_code || 0;
        const cls = sc >= 500 ? 'call-5xx' : sc >= 400 ? 'call-4xx' : 'call-2xx';
        return `<div class="call-log-entry">
          <span class="call-method">${c.method}</span>
          <span class="call-path">${c.path}</span>
          <span class="call-status ${cls}">${sc}${c.cache_hit ? ' (cache)' : ''}</span>
        </div>`;
      }).join('')
    }</div>`;
  }

  const brief = s.execution_brief || {};
  const briefHtml = Object.keys(brief).length > 0 ? `
    <details style="background:var(--surface);border:1px solid var(--border);border-radius:8px;margin-bottom:12px;overflow:hidden;">
      <summary style="padding:12px 16px;cursor:pointer;font-weight:600;font-size:14px;background:var(--surface2);">
        Execution Brief (${(brief.planned_endpoints||[]).length} endpoints, ${(brief.field_rules||[]).length} rule sets)
      </summary>
      <div style="padding:16px;font-size:13px;font-family:'SF Mono','Menlo',monospace;overflow-x:auto;">
        <div style="margin-bottom:12px;">
          <strong style="color:var(--blue);">Planned Endpoints:</strong><br>
          ${(brief.planned_endpoints||[]).map(e => `<div>${e.method} ${e.path} <span style="color:var(--text2)">// ${e.summary}</span></div>`).join('')}
        </div>
        <div style="margin-bottom:12px;">
          <strong style="color:var(--purple);">Prefetched Schemas:</strong> ${(brief.prefetched_schema_names||[]).join(', ')}
        </div>
        <div style="margin-bottom:12px;">
          <strong style="color:var(--yellow);">Computed Dates:</strong> ${JSON.stringify(brief.computed_dates||{})}
        </div>
        ${brief.has_trace_example ? `<div style="margin-bottom:12px;"><strong style="color:var(--green);">Trace Example Injected:</strong> ${brief.trace_example_task}</div>` : ''}
        ${(brief.field_rules||[]).length > 0 ? `
          <div>
            <strong style="color:var(--red);">Field Rules:</strong>
            ${(brief.field_rules||[]).map(r => `
              <div style="margin-left:12px;margin-top:4px;">
                <div style="font-weight:600;">${r.endpoint}</div>
                ${(r.rules||[]).map(rule => `<div style="color:var(--text2);margin-left:12px;">- ${rule}</div>`).join('')}
              </div>
            `).join('')}
          </div>
        ` : ''}
      </div>
    </details>
  ` : '';

  const promptHtml = s.executor_system_prompt ? `
    <details style="background:var(--surface);border:1px solid var(--border);border-radius:8px;margin-bottom:12px;overflow:hidden;">
      <summary style="padding:12px 16px;cursor:pointer;font-weight:600;font-size:14px;background:var(--surface2);">
        System Prompt
      </summary>
      <div style="padding:16px;background:var(--bg);max-height:400px;overflow:auto;">
        <pre style="font-family:'SF Mono','Menlo',monospace;font-size:11px;color:var(--text2);white-space:pre-wrap;">${s.executor_system_prompt}</pre>
      </div>
    </details>
  ` : '';

  const plannerHtml = s.planner_payload ? `
    <details style="background:var(--surface);border:1px solid var(--border);border-radius:8px;margin-bottom:12px;overflow:hidden;">
      <summary style="padding:12px 16px;cursor:pointer;font-weight:600;font-size:14px;background:var(--surface2);">
        Planner Output
      </summary>
      <div style="padding:16px;background:var(--bg);max-height:500px;overflow:auto;">
        <pre style="font-family:'SF Mono','Menlo',monospace;font-size:11px;color:var(--text2);">${JSON.stringify(s.planner_payload, null, 2)}</pre>
      </div>
    </details>
  ` : '';

  const metaHtml = s.metadata && Object.keys(s.metadata).length > 0
    ? Object.entries(s.metadata).map(([k,v]) =>
        `<div class="detail-stat"><div class="detail-stat-label">${k}</div><div class="detail-stat-value" style="font-size:13px;">${v}</div></div>`
      ).join('')
    : '';

  panel.innerHTML = `
    <div class="detail-header">
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px;">
        <h2>${s.task_type || 'Unknown Task'}</h2>
        ${statusBadge(s.status)}
        ${sourceBadge(s.source)}
      </div>
      ${s.goal ? `<div style="color:var(--text2);font-size:13px;margin-bottom:8px;">${s.goal}</div>` : ''}
      <div class="detail-prompt">${s.prompt || '(no prompt)'}</div>
      ${s.summary ? `<div style="color:var(--green);font-size:13px;margin-bottom:8px;">&#x2714; ${s.summary}</div>` : ''}
      ${s.error_message ? `<div style="color:var(--red);font-size:13px;margin-bottom:8px;">&#x2716; ${truncate(s.error_message, 300)}</div>` : ''}
      ${s.competition_score ? (s.competition_score.status === 'in_progress' || s.competition_score.status === 'pending' || s.status === 'running' ?
        `<div style="background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.2);border-radius:8px;padding:12px 16px;margin-bottom:12px;">
           <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
             <span class="badge badge-yellow" style="font-size:14px;">Evaluating submission...</span>
           </div>
           ${s.competition_score.submission_id ? '<div style="font-size:11px;color:var(--text2);margin-top:8px;font-family:monospace;">ID: '+s.competition_score.submission_id+'</div>' : ''}
         </div>` :
        `<div style="background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.2);border-radius:8px;padding:12px 16px;margin-bottom:12px;">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
          <span title="Leaderboard Points = (Correctness × Tier) + Efficiency Bonus" style="cursor:help;font-size:24px;font-weight:700;color:var(--yellow);">&#127942; ${(s.competition_score.normalized_score || 0).toFixed(4)} <span style="font-size:14px;font-weight:600;color:var(--text2);text-transform:uppercase;">Leaderboard Pts</span></span>
          ${s.competition_score.checks_total > 0 && s.competition_score.checks_passed === s.competition_score.checks_total ? '<span class="badge badge-green" style="font-size:12px;margin-left:auto;">PERFECT</span>' : ''}
        </div>
        <div style="background:rgba(0,0,0,0.2);border-radius:6px;padding:8px 12px;margin-bottom:12px;">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
            <span style="font-size:13px;font-weight:600;color:var(--text);">Check Correctness</span>
            <span style="font-size:13px;font-weight:700;color:var(--text);">${s.competition_score.score_raw} / ${s.competition_score.score_max} pts</span>
          </div>
          <div style="font-size:12px;color:var(--text2);margin-bottom:8px;">${s.competition_score.comment}</div>
          <div class="check-pills">${(s.competition_score.checks||[]).map(c => {
            const isPass = c.toLowerCase().includes('passed');
            return '<span class="check-pill '+(isPass?'check-passed':'check-failed')+'">'+c+'</span>';
          }).join('')}</div>
        </div>
        <div style="font-size:11px;color:var(--text2);font-style:italic;line-height:1.4;">
          * Note: Leaderboard points include a dynamic Efficiency Bonus for perfect runs. The efficiency benchmark recalculates every 12h against global limits.
        </div>
        ${s.competition_score.submission_id ? '<div style="font-size:11px;color:var(--text2);margin-top:8px;font-family:monospace;">Submission ID: '+s.competition_score.submission_id+'</div>' : ''}
      </div>`) : ''}
    </div>
    <div class="detail-grid">
      <div class="detail-stat"><div class="detail-stat-label">Duration</div><div class="detail-stat-value">${fmtDuration(s.duration_seconds)}</div></div>
      <div class="detail-stat"><div class="detail-stat-label">API Calls</div><div class="detail-stat-value">${s.tripletex_call_count}</div></div>
      <div class="detail-stat"><div class="detail-stat-label">API Errors</div><div class="detail-stat-value" style="color:${s.tripletex_error_count > 0 ? 'var(--red)' : 'var(--text)'}">${s.tripletex_error_count}</div></div>
      <div class="detail-stat"><div class="detail-stat-label">Tool Calls</div><div class="detail-stat-value">${s.tool_call_count}</div></div>
      <div class="detail-stat"><div class="detail-stat-label">Preflight Blocks</div><div class="detail-stat-value" style="color:${s.preflight_rejection_count > 0 ? 'var(--orange)' : 'var(--text)'}">${s.preflight_rejection_count || 0}</div></div>
      <div class="detail-stat"><div class="detail-stat-label">Events</div><div class="detail-stat-value">${s.event_count}</div></div>
      <div class="detail-stat"><div class="detail-stat-label">Files</div><div class="detail-stat-value">${s.file_count}</div></div>
      ${metaHtml}
    </div>
    ${briefHtml}
    ${promptHtml}
    ${plannerHtml}
    ${callLogHtml}
    ${renderEnforcerLog(s)}
    ${renderPreflightLog(s)}
    <div class="timeline">
      <h3>Event Timeline (${events.length} events)</h3>
      ${events.map((e, i) => {
        const summary = eventSummary(e);
        const p = e.payload || {};
        const isErr = e.event_type === 'tool_result' && (p.result?.ok === false || p.result?.error);
        return `
          <div class="tl-event" onclick="togglePayload(${i})">
            <span class="tl-time">${fmtTime(e.timestamp)}</span>
            <span class="tl-badge">${eventBadge(e.event_type)}</span>
            <span class="tl-detail" style="${isErr ? 'color:var(--red)' : ''}">${summary}</span>
          </div>
          <div class="tl-payload" id="payload-${i}">${JSON.stringify(e.payload, null, 2)}</div>
        `;
      }).join('')}
    </div>
  `;

  // Auto-scroll to latest events for running tasks
  if (s.status === 'running') {
    requestAnimationFrame(() => {
      panel.scrollTop = panel.scrollHeight;
    });
  }
}

function renderEnforcerLog(s) {
  const det = s.enforcer_rejections || [];
  const sem = s.semantic_rejections || [];
  const ovr = s.enforcer_overrides || [];
  if (det.length === 0 && sem.length === 0 && ovr.length === 0) return '';
  let html = '<div class="call-log" style="margin-top:16px;"><h3>🛡️ Enforcer Activity</h3>';
  for (const r of det) {
    html += `<div class="call-log-entry">
      <span class="call-method badge badge-red" style="font-size:10px;">BLOCKED</span>
      <span class="call-path" style="margin-left:6px;">${r.tool_name || 'tripletex_request'}: ${r.reason || ''}</span>
      <div style="font-size:11px;color:var(--text2);margin-left:72px;margin-top:2px;">→ ${r.suggestion || ''}</div>
    </div>`;
  }
  for (const r of sem) {
    html += `<div class="call-log-entry">
      <span class="call-method badge badge-red" style="font-size:10px;">🧠 SEM</span>
      <span class="call-path" style="margin-left:6px;">${r.reason || ''}</span>
      <div style="font-size:11px;color:var(--text2);margin-left:72px;margin-top:2px;">→ ${r.suggestion || ''}</div>
    </div>`;
  }
  for (const r of ovr) {
    html += `<div class="call-log-entry">
      <span class="call-method badge badge-yellow" style="font-size:10px;">⚡ OVRD</span>
      <span class="call-path" style="margin-left:6px;">${r.reason || ''}</span>
    </div>`;
  }
  html += '</div>';
  return html;
}

function renderPreflightLog(s) {
  const rejections = s.preflight_rejections || [];
  const stripped = s.auto_stripped || [];
  if (rejections.length === 0 && stripped.length === 0) return '';
  let html = '<div class="call-log" style="margin-top:16px;"><h3>Schema Validation Issues</h3>';
  for (const r of rejections) {
    html += `<div class="call-log-entry"><span class="call-method badge badge-red" style="font-size:10px;">BLOCKED</span><span class="call-path" style="margin-left:6px;">${r.error || 'Schema validation failed'}</span></div>`;
  }
  for (const a of stripped) {
    html += `<div class="call-log-entry"><span class="call-method badge badge-yellow" style="font-size:10px;">STRIP</span><span class="call-path" style="margin-left:6px;">Auto-removed: ${(a.fields || []).join(', ')}</span></div>`;
  }
  html += '</div>';
  return html;
}

function eventSummary(e) {
  const p = e.payload || {};
  switch (e.event_type) {
    case 'init': return truncate(p.prompt || '', 60);
    case 'planner': return `${p.task_type}: ${truncate(p.goal || '', 50)}`;
    case 'tool_start': return `${p.tool_name}(${truncate(JSON.stringify(p.arguments || {}), 60)})`;
    case 'tool_result': {
      const ok = p.result?.ok;
      const prefix = ok ? '&#x2714;' : '&#x2716;';
      const detail = ok
        ? (p.result?.summary || p.result?.resource_id ? `id=${p.result.resource_id}` : 'ok')
        : truncate(p.result?.error || 'failed', 60);
      return `${prefix} ${p.tool_name}: ${detail}`;
    }
    case 'final_payload': return p.summary || p.status || '';
    case 'done': return `calls=${p.tripletex_call_count} errors=${p.tripletex_error_count}`;
    case 'error': return truncate(p.message || '', 80);
    case 'blocked_warning': return truncate(p.message || '', 80);
    case 'recovery': return 'Drift recovery triggered';
    case 'enforcer_rejected': return `🛡️ BLOCKED: ${truncate(p.reason || '', 70)} → ${truncate(p.suggestion || '', 50)}`;
    case 'semantic_enforcer_rejected': return `🧠 BLOCKED: ${truncate(p.reason || '', 80)}`;
    case 'enforcer_override': return `⚡ Override: ${truncate(p.reason || '', 80)}`;
    case 'enforcer_passed': return `🛡️✅ ${p.call || ''}`;
    case 'thinking': return `🧠 ${truncate(p.text || '', 100)}`;
    case 'assistant_reasoning': return `💬 ${truncate(p.text || '', 100)}`;
    case 'api_advisor_query': return `🔍 Advisor: ${truncate(p.question || '', 80)}`;
    case 'api_advisor_response': return `📋 Advisor: ${p.endpoints_found || 0} endpoints, ${truncate(p.response_preview || '', 60)}`;
    case 'execution_brief': return `${(p.planned_endpoints || []).length} endpoints, ${p.field_rules_count || 0} rules, trace=${p.has_trace_example ? 'yes' : 'no'}`;
    case 'competition_scoring': return `checks=${p.checks_passed || 0}/${p.checks_total || 0}`;
    case 'error_summary': return `${(p.errors || []).length} error(s)`;
    case 'attachments_prepared': return `${(p.attachments || []).length} attachment(s)`;
    case 'metadata_update': return `tier=${p.task_tier || '?'} executor=${(p.executor_model || '').split('/').pop() || '?'}`;
    default: return JSON.stringify(p).slice(0, 60);
  }
}

function togglePayload(id) {
  const elId = typeof id === 'number' ? `payload-${id}` : id;
  const el = document.getElementById(elId);
  if (el) el.classList.toggle('open');
}

// Filter buttons
document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    if (btn.id === 'btn-scores') return; // Handled separately
    competitionViewActive = false;
    document.getElementById('btn-scores').classList.remove('active');
    
    if (btn.dataset.filter === 'raw') {
      showRawRequests();
      return;
    }
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeFilter = btn.dataset.filter;
    renderRunList();
  });
});

async function showRawRequests() {
  const panel = document.getElementById('run-detail');
  panel.innerHTML = '<div class="run-detail-empty">Loading raw requests...</div>';
  try {
    const res = await fetch('/api/raw-requests');
    const data = await res.json();
    const reqs = data.requests || [];
    if (reqs.length === 0) {
      panel.innerHTML = '<div class="run-detail-empty">No raw requests logged yet</div>';
      return;
    }
    const statusCls = (sc) => sc >= 500 ? 'call-5xx' : sc >= 400 ? 'call-4xx' : 'call-2xx';
    panel.innerHTML = `
      <div class="detail-header">
        <h2>Raw HTTP Requests (${reqs.length})</h2>
        <div style="color:var(--text2);font-size:12px;margin-top:4px;">All non-dashboard requests logged by middleware</div>
      </div>
      <div style="font-family:'SF Mono','Menlo',monospace;font-size:12px;">
        ${reqs.map((r, i) => `
          <div class="tl-event" onclick="togglePayload('raw-${i}')" style="align-items:flex-start;">
            <span class="tl-time">${fmtTime(r.timestamp)}</span>
            <span style="min-width:55px;font-weight:600;">${r.method}</span>
            <span style="min-width:200px;">${r.path}${r.query ? '?' + r.query : ''}</span>
            <span class="${statusCls(r.status_code)}" style="min-width:40px;">${r.status_code}</span>
            <span style="color:var(--text2);">${r.elapsed_seconds}s</span>
            <span style="color:var(--text2);margin-left:8px;">${r.client || ''}</span>
          </div>
          <div class="tl-payload" id="raw-${i}" style="margin-left:80px;">Headers: ${JSON.stringify(r.headers || {}, null, 2)}\\n\\nBody: ${r.body_preview ? r.body_preview.slice(0, 1000) : '(empty)'}</div>
        `).join('')}
      </div>
    `;
  } catch (e) {
    panel.innerHTML = '<div class="run-detail-empty">Error: ' + e.message + '</div>';
  }
}



async function fetchCompetitionData(force) {
  const isRunningOrPending = allRuns.some(r => r.status === 'running') || globalHasPendingSubmissions;
  const ttl = isRunningOrPending ? 15000 : 120000;
  if (!force && Date.now() - lastCompetitionRefresh < ttl) return;
  try {
    const res = await fetch('/api/competition/submissions');
    const data = await res.json();
    lastCompetitionRefresh = Date.now();
    
    if (data.summary) {
       const best = data.summary.best_normalized_score || 0;
       const bestElem = document.getElementById('hdr-best-score');
       const bestCont = document.getElementById('hdr-best-score-container');
       if (bestElem && bestCont) {
         bestElem.textContent = best.toFixed(4);
         bestCont.style.display = 'inline-block';
       }
       
       const currentCount = data.summary.total || 0;
       if (lastSubmissionCount > 0 && currentCount > lastSubmissionCount) {
         playCompletionSound();
         flashHeader('done');
       }
       lastSubmissionCount = currentCount;
    }

    if (competitionViewActive) {
      renderCompetitionView(data);
    }
  } catch(e) { console.error('Comp fetch error:', e); }
}

function showCompetitionScores() {
  competitionViewActive = true;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('btn-scores').classList.add('active');
  document.getElementById('run-detail').innerHTML = '<div class="run-detail-empty">Loading scores...</div>';
  fetchCompetitionData(true);
}

async function submitNewRun() {
  const url = document.getElementById('submit-endpoint-url').value.trim();
  if (!url) return;
  const btn = document.getElementById('btn-submit-run');
  const statusEl = document.getElementById('submit-status');
  btn.disabled = true;
  btn.textContent = 'Submitting...';
  statusEl.style.display = 'block';
  statusEl.style.color = 'var(--text2)';
  statusEl.textContent = 'Sending submission request...';
  try {
    const res = await fetch('/api/competition/submit', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({endpoint_url: url, endpoint_api_key: null})
    });
    const data = await res.json();
    if (data.error) {
      statusEl.style.color = 'var(--red)';
      statusEl.textContent = 'Error: ' + data.error;
    } else {
      statusEl.style.color = 'var(--green)';
      statusEl.textContent = 'Queued! ID: ' + data.submission_id + ' (used ' + data.daily_submissions_used + '/' + data.daily_submissions_max + ' daily)';
      playStartSound();
      flashHeader('start');
      // Force refresh competition data
      lastCompetitionRefresh = 0;
      setTimeout(() => fetchCompetitionData(true), 5000);
    }
  } catch(e) {
    statusEl.style.color = 'var(--red)';
    statusEl.textContent = 'Network error: ' + e.message;
  }
  btn.disabled = false;
  btn.textContent = 'Submit Run';
}

function renderCompetitionView(data) {
  const panel = document.getElementById('run-detail');
  const s = data.summary || {};
  
  const gridHtml = `
    <div class="detail-grid">
      <div class="detail-stat"><div class="detail-stat-label">Best Score</div><div class="detail-stat-value" style="color:var(--green);font-size:20px;">${(s.best_normalized_score||0).toFixed(4)}</div></div>
      <div class="detail-stat"><div class="detail-stat-label">Perfect Runs</div><div class="detail-stat-value">${s.perfect_runs || 0}</div></div>
      <div class="detail-stat"><div class="detail-stat-label">Total Submissions</div><div class="detail-stat-value">${s.total || 0}</div></div>
      <div class="detail-stat"><div class="detail-stat-label">Check Pass Rate</div><div class="detail-stat-value">${((s.check_pass_rate || 0)*100).toFixed(1)}%</div></div>
    </div>
  `;
  
  const listHtml = (data.submissions || []).map(sub => {
    const feedback = sub.feedback || {};
    const checks = (feedback.checks || []).map(c => {
      const isPass = c.toLowerCase().includes('passed');
      return `<span class="check-pill ${isPass ? 'check-passed' : 'check-failed'}">${c}</span>`;
    }).join('');
    
    const err = sub.error ? `<div style="color:var(--red);font-size:12px;margin-bottom:6px;">${sub.error}</div>` : '';
    const comment = feedback.comment ? `<div style="font-size:13px;color:var(--text2);margin-bottom:8px;">${feedback.comment}</div>` : '';
    const norm = typeof sub.normalized_score === 'number' ? sub.normalized_score.toFixed(4) : '0.0000';
    const dur = sub.duration_ms ? fmtDuration(sub.duration_ms/1000) : '-';
    
    return `
      <div class="competition-card">
        <div class="competition-card-top">
          <div style="font-weight:600;font-size:14px;">
             ${fmtTime(sub.queued_at)} 
             <span style="color:var(--text2);font-weight:400;margin-left:8px;font-size:12px;">(${dur})</span>
          </div>
          <div style="font-weight:600;color:var(--yellow);">
            ${sub.score_raw}/${sub.score_max} <span style="opacity:0.7">(${norm})</span>
          </div>
        </div>
        ${err}
        ${comment}
        <div class="check-pills">${checks}</div>
      </div>
    `;
  }).join('');
  
  panel.innerHTML = `
    <div class="detail-header">
      <h2>&#127942; Competition Submissions</h2>
      <div style="display:flex;gap:12px;align-items:center;margin-bottom:16px;">
        <input type="text" id="submit-endpoint-url" placeholder="Endpoint URL" 
          value="https://annamae-subseptate-nonveraciously.ngrok-free.dev/solve"
          style="flex:1;background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:8px 12px;color:var(--text);font-size:13px;outline:none;">
        <button onclick="submitNewRun()" id="btn-submit-run" 
          ${allRuns.some(r => r.status === 'running') ? 'disabled style="background:var(--surface2);color:var(--text2);border:none;border-radius:6px;padding:8px 16px;font-weight:600;font-size:13px;cursor:not-allowed;white-space:nowrap;"' : 'style="background:var(--yellow);color:var(--bg);border:none;border-radius:6px;padding:8px 16px;font-weight:600;font-size:13px;cursor:pointer;white-space:nowrap;"'}>
          Submit Run
        </button>
        <input type="number" id="batch-count" value="10" min="1" max="50" 
          style="width:50px;background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:8px;color:var(--text);font-size:13px;text-align:center;">
        <button onclick="startBatch()" id="btn-batch"
          style="background:var(--green);color:var(--bg);border:none;border-radius:6px;padding:8px 16px;font-weight:600;font-size:13px;cursor:pointer;white-space:nowrap;">
          Batch Run
        </button>
        <button onclick="stopBatch()" id="btn-batch-stop"
          style="background:var(--red);color:#fff;border:none;border-radius:6px;padding:8px 12px;font-weight:600;font-size:13px;cursor:pointer;white-space:nowrap;display:none;">
          Stop
        </button>
      </div>
      <div id="batch-status" style="font-size:12px;margin-bottom:4px;display:none;color:var(--text2);"></div>
      <div id="submit-status" style="font-size:12px;margin-bottom:12px;display:none;"></div>
    </div>
    ${gridHtml}
    <div style="margin-top:20px;">${listHtml}</div>
  `;
}

document.getElementById('btn-scores').addEventListener('click', showCompetitionScores);
document.getElementById('search').addEventListener('input', () => renderRunList());

// Auto-refresh
async function refresh() {
  fetchCompetitionData(false);
  try {
    const [runsRes, settingsRes] = await Promise.all([
      fetch('/api/runs'),
      fetch('/api/settings'),
    ]);
    const runsData = await runsRes.json();
    const settingsData = await settingsRes.json();

    const newRuns = runsData.runs || [];

    // --- Sound & flash triggers: detect state transitions ---
    const newStatuses = {};
    const newScoreStatuses = {};
    for (const run of newRuns) {
      newStatuses[run.run_id] = run.status;
      const prev = previousRunStatuses[run.run_id];
      if (!prev && run.status === 'running') {
        // New run just started
        playStartSound();
        flashHeader('start');
      } else if (prev === 'running' && run.status === 'completed') {
        // Run just completed successfully
        playCompletionSound();
        flashHeader('done');
        fetch('/api/runs/enrich', {method: 'POST'}).catch(e => console.error(e));
      } else if (prev === 'running' && run.status === 'error') {
        // Run just failed
        playErrorSound();
        flashHeader('error');
        fetch('/api/runs/enrich', {method: 'POST'}).catch(e => console.error(e));
      }

      if (run.competition_score) {
        newScoreStatuses[run.run_id] = run.competition_score.status;
        const prevScore = previousScoreStatuses[run.run_id];
        if (prevScore && ['in_progress', 'pending'].includes(prevScore) && run.competition_score.status === 'completed') {
          playCompletionSound();
          flashHeader('done');
          fetch('/api/runs/enrich', {method: 'POST'}).catch(e => console.error(e));
        }
      }
    }
    previousRunStatuses = newStatuses;
    previousScoreStatuses = newScoreStatuses;

    allRuns = newRuns;
    globalHasPendingSubmissions = runsData.has_pending_submissions || false;
    document.getElementById('hdr-active').textContent = runsData.active_count;
    document.getElementById('hdr-total').textContent = runsData.total_count;
    const shortModel = (m) => m ? m.split('/').pop().replace(':exacto','') : '-';
    document.getElementById('hdr-planner').textContent = shortModel(settingsData.planner_model);
    document.getElementById('hdr-t1').textContent = shortModel(settingsData.tier1_executor_model);
    document.getElementById('hdr-t2').textContent = shortModel(settingsData.tier2_executor_model);
    document.getElementById('hdr-t3').textContent = shortModel(settingsData.tier3_executor_model);

    renderRunList();

    if (competitionViewActive) {
      const isRunning = allRuns.some(r => r.status === 'running');
      const btn = document.getElementById('btn-submit-run');
      if (btn) {
        btn.disabled = isRunning;
        btn.style.background = isRunning ? 'var(--surface2)' : 'var(--yellow)';
        btn.style.color = isRunning ? 'var(--text2)' : 'var(--bg)';
        btn.style.cursor = isRunning ? 'not-allowed' : 'pointer';
      }
    }

    // Auto-refresh active run detail (smooth — no loading flash)
    if (selectedRunId) {
      const selectedRun = allRuns.find(r => r.run_id === selectedRunId);
      if (selectedRun && (selectedRun.status === 'running' || (selectedRun.competition_score && ['in_progress', 'pending'].includes(selectedRun.competition_score.status)))) {
        try {
          const res = await fetch(`/api/runs/${selectedRunId}`);
          const data = await res.json();
          if (!data.error) renderDetail(data);
        } catch (e) { /* ignore refresh errors for detail */ }
      }
    }
  } catch (e) {
    console.error('Refresh error:', e);
  }
}

refresh();
setInterval(refresh, 4000);
setInterval(() => { fetch('/api/runs/enrich', {method: 'POST'}).catch(e => console.error(e)); }, 30000);

let batchPollInterval = null;

async function startBatch() {
  const count = parseInt(document.getElementById('batch-count')?.value || '10');
  const btn = document.getElementById('btn-batch');
  const stopBtn = document.getElementById('btn-batch-stop');
  const statusEl = document.getElementById('batch-status');
  btn.disabled = true;
  btn.style.opacity = '0.5';
  stopBtn.style.display = 'inline-block';
  statusEl.style.display = 'block';
  statusEl.textContent = `Starting batch of ${count} runs...`;
  try {
    const res = await fetch('/api/competition/batch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({count, delay_seconds: 5}),
    });
    const data = await res.json();
    if (data.error) { statusEl.textContent = data.error; return; }
    statusEl.textContent = `Batch started: ${count} runs`;
    batchPollInterval = setInterval(pollBatchStatus, 3000);
  } catch(e) { statusEl.textContent = 'Error: ' + e; }
}

async function stopBatch() {
  try {
    await fetch('/api/competition/batch/stop', {method: 'POST'});
    document.getElementById('batch-status').textContent = 'Stopping...';
  } catch(e) {}
}

async function pollBatchStatus() {
  try {
    const res = await fetch('/api/competition/batch/status');
    const data = await res.json();
    const p = data.progress;
    const statusEl = document.getElementById('batch-status');
    const btn = document.getElementById('btn-batch');
    const stopBtn = document.getElementById('btn-batch-stop');
    statusEl.style.display = 'block';
    const results = data.results || [];
    const totalErrors = results.reduce((s,r) => s + (r.errors || 0), 0);
    const totalCalls = results.reduce((s,r) => s + (r.calls || 0), 0);
    statusEl.textContent = `${p.status} | ${p.completed}/${p.total} done | ${totalCalls} API calls, ${totalErrors} errors`;
    if (!data.active) {
      clearInterval(batchPollInterval);
      batchPollInterval = null;
      btn.disabled = false;
      btn.style.opacity = '1';
      stopBtn.style.display = 'none';
      statusEl.textContent += ' ✅';
    }
  } catch(e) {}
}
</script>
</body>
</html>"""
