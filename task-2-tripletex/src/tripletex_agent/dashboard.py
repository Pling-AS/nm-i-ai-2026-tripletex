"""Monitoring dashboard for Tripletex agent runs."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import secrets
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.routing import WebSocketRoute

from tripletex_agent.config import get_settings, Settings
from tripletex_agent.postmortem import (
    append_postmortem_event,
    build_analysis_payload,
    generate_postmortem,
)
from tripletex_agent.run_store import get_run_store
from tripletex_agent.trace import RUNS_DIR, RunTrace, register_trace_callback

logger = logging.getLogger(__name__)

security = HTTPBasic(auto_error=False)


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.append(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections = [c for c in self._connections if c is not websocket]

    async def broadcast(self, message: dict) -> None:
        async with self._lock:
            dead: list[WebSocket] = []
            for conn in self._connections:
                try:
                    await conn.send_json(message)
                except Exception:
                    dead.append(conn)
            for connection in dead:
                self._connections = [
                    c for c in self._connections if c is not connection
                ]

    @property
    def connection_count(self) -> int:
        return len(self._connections)


_ws_manager: ConnectionManager | None = None

from collections import deque

_pending_submissions: deque[dict[str, Any]] = deque(maxlen=50)
_PENDING_FILE = RUNS_DIR / ".pending_submissions.json"


def _load_pending_from_disk() -> None:
    try:
        if _PENDING_FILE.exists():
            data = json.loads(_PENDING_FILE.read_text(encoding="utf-8"))
            for entry in data:
                if isinstance(entry, dict) and "submission_id" in entry:
                    _pending_submissions.append(entry)
    except Exception:
        pass


def _save_pending_to_disk() -> None:
    try:
        _PENDING_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PENDING_FILE.write_text(
            json.dumps(list(_pending_submissions), ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


_load_pending_from_disk()


def register_pending_submission(submission_id: str) -> None:
    _pending_submissions.append(
        {
            "submission_id": submission_id,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _save_pending_to_disk()


def pop_pending_submission() -> str | None:
    if _pending_submissions:
        entry = _pending_submissions.popleft()
        _save_pending_to_disk()
        return entry["submission_id"]
    return None


def get_ws_manager() -> ConnectionManager:
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = ConnectionManager()
    return _ws_manager


def verify_dashboard_access(
    credentials: HTTPBasicCredentials | None = Depends(security),
    settings: Settings = Depends(get_settings),
) -> str:
    """Verify dashboard access using Basic Auth against APP_API_KEY.

    If APP_API_KEY is not set, allow access (dev mode).
    If set, require Basic Auth with any username and password=APP_API_KEY.
    """
    if not settings.app_api_key:
        return "admin"

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )

    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = settings.app_api_key.encode("utf8")
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )

    if not is_correct_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# --- Cached competition submissions for run matching ---
_submissions_cache: list[dict[str, Any]] = []
_submissions_cache_ts: float = 0
_enrichment_retry_state: dict[str, dict[str, float | int]] = {}
_enrich_lock = asyncio.Lock()


async def _get_cached_submissions(force_refresh: bool = False) -> list[dict[str, Any]]:
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
    ttl = 10.0 if (active_count > 0 or has_pending) else 60.0

    if not force_refresh and _submissions_cache and (now - _submissions_cache_ts) < ttl:
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
    run_endpoint_url: str | None = None,
) -> dict[str, Any] | None:
    """Match a local run to a competition submission by timestamp.

    Uses platform_started_at when available and falls back to queued_at.
    """
    if not run_started_at or not submissions:
        return None
    try:
        run_ts = datetime.fromisoformat(run_started_at)
    except ValueError:
        return None

    best_match = None
    best_delta = 30.0

    normalized_run_endpoint = (run_endpoint_url or "").rstrip("/").lower()
    best_endpoint_mismatch = 1

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
        if delta > 30.0:
            continue

        sub_endpoint = (sub.get("endpoint_url") or "").rstrip("/").lower()
        endpoint_mismatch = 1
        if normalized_run_endpoint and sub_endpoint:
            endpoint_mismatch = 0 if normalized_run_endpoint == sub_endpoint else 1

        if (endpoint_mismatch, delta) < (best_endpoint_mismatch, best_delta):
            best_endpoint_mismatch = endpoint_mismatch
            best_delta = delta
            best_match = sub

    return best_match


def _attach_competition_scores(
    runs: list[dict[str, Any]], submissions: list[dict[str, Any]]
) -> None:
    used_sub_ids: set[str] = set()

    def _match_and_attach(run: dict[str, Any]) -> None:
        run_sub_id = run.get("metadata", {}).get("submission_id")
        match = None
        if run_sub_id:
            match = next((s for s in submissions if s.get("id") == run_sub_id), None)
        if not match:
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

    for run in runs:
        if run.get("source") != "competition":
            run["competition_score"] = None
            continue
        if run.get("status") != "error":
            _match_and_attach(run)

    for run in runs:
        if run.get("source") == "competition" and run.get("status") == "error":
            _match_and_attach(run)


router = APIRouter(dependencies=[Depends(verify_dashboard_access)])


async def dashboard_ws(websocket: WebSocket) -> None:
    manager = get_ws_manager()
    await manager.connect(websocket)
    _ensure_file_poll_started()
    try:
        store = get_run_store()
        runs = store.list_summaries()

        submissions = await _get_cached_submissions()
        _attach_competition_scores(runs, submissions)

        active_count = len(RunTrace.get_active_runs())
        await websocket.send_json(
            {
                "type": "snapshot",
                "payload": {
                    "runs": runs,
                    "active_count": active_count,
                    "total_count": len(runs),
                },
            }
        )

        s = get_settings()
        await websocket.send_json(
            {
                "type": "settings",
                "payload": {
                    "model": s.openrouter_model,
                    "planner_model": s.planner_model,
                    "tier1_executor_model": s.tier1_executor_model,
                    "tier2_executor_model": s.tier2_executor_model,
                    "tier3_executor_model": s.tier3_executor_model,
                    "max_steps": s.agent_max_steps,
                    "temperature": s.agent_model_temperature,
                },
            }
        )

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await manager.disconnect(websocket)


async def broadcast_run_update(
    run_id: str, update_type: str, data: dict | None = None
) -> None:
    manager = get_ws_manager()
    if manager.connection_count == 0:
        return

    message = {
        "type": update_type,
        "payload": {"run_id": run_id, **(data or {})},
    }
    await manager.broadcast(message)


async def broadcast_run_list_update() -> None:
    manager = get_ws_manager()
    if manager.connection_count == 0:
        return

    store = get_run_store()
    runs = store.list_summaries()

    submissions = await _get_cached_submissions()
    _attach_competition_scores(runs, submissions)

    active_count = len(RunTrace.get_active_runs())
    await manager.broadcast(
        {
            "type": "snapshot",
            "payload": {
                "runs": runs,
                "active_count": active_count,
                "total_count": len(runs),
            },
        }
    )


async def _on_trace_event(trace: RunTrace, event_type: str, payload: dict) -> None:
    manager = get_ws_manager()
    if manager.connection_count == 0:
        return

    store = get_run_store()

    if event_type == "_close":
        store.invalidate_run(trace.run_id)
        await broadcast_run_list_update()
        return

    if event_type in (
        "init",
        "done",
        "error",
        "planner",
        "metadata_update",
        "competition_scoring",
    ):
        store.refresh_run(trace.path)
        await broadcast_run_list_update()
        return

    await broadcast_run_update(
        trace.run_id,
        "run_event",
        {
            "event_type": event_type,
            "timestamp": payload.get("timestamp", ""),
        },
    )


register_trace_callback(_on_trace_event)
router.routes.append(WebSocketRoute("/dashboard/ws", endpoint=dashboard_ws))


_file_poll_task: asyncio.Task | None = None  # type: ignore[type-arg]


async def _poll_runs_directory() -> None:
    store = get_run_store()
    last_file_set: set[str] = set()
    last_mtime_sum: float = 0.0

    while True:
        await asyncio.sleep(5)
        try:
            manager = get_ws_manager()
            if manager.connection_count == 0:
                continue

            if not RUNS_DIR.exists():
                continue

            current_files: set[str] = set()
            mtime_sum: float = 0.0
            for f in RUNS_DIR.glob("*.jsonl"):
                if f.name == "raw_requests.jsonl":
                    continue
                current_files.add(f.name)
                mtime_sum += f.stat().st_mtime

            if current_files != last_file_set or mtime_sum != last_mtime_sum:
                last_file_set = current_files
                last_mtime_sum = mtime_sum
                store.invalidate_all()
                await broadcast_run_list_update()
        except Exception:
            pass


def _ensure_file_poll_started() -> None:
    global _file_poll_task
    if _file_poll_task is None or _file_poll_task.done():
        try:
            loop = asyncio.get_running_loop()
            _file_poll_task = loop.create_task(_poll_runs_directory())
        except RuntimeError:
            pass


def _parse_trace_file(path: Path) -> list[dict[str, Any]]:
    return get_run_store().parse_trace_file(path)


def _summarize_run(path: Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    return get_run_store().summarize_run(path, events)


@router.get("/api/runs")
async def list_runs() -> JSONResponse:
    store = get_run_store()
    runs: list[dict[str, Any]] = store.list_summaries()

    # Also add active runs not yet on disk
    for run_id, trace in RunTrace.get_active_runs().items():
        if not any(r["run_id"] == run_id for r in runs):
            detail = store.get_detail(run_id)
            summary = (
                detail[0] if detail else store.get_active_run_summary(run_id, trace)
            )
            summary["status"] = "running"
            runs.insert(0, summary)

    submissions = await _get_cached_submissions()
    _attach_competition_scores(runs, submissions)

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
    async with _enrich_lock:
        return await _do_enrich_runs()


async def _do_enrich_runs() -> JSONResponse:
    import platform as _platform

    local_hostname = _platform.node()

    store = get_run_store()
    submissions = await _get_cached_submissions()
    enriched_count = 0
    skipped_count = 0
    pending_count = 0
    postmortem_count = 0

    if RUNS_DIR.exists():
        for path in sorted(RUNS_DIR.glob("*.jsonl"), reverse=True):
            if path.name == "raw_requests.jsonl":
                continue
            run_id = store.get_run_id(path)
            detail = store.get_detail(run_id)
            events = detail[1] if detail else store.parse_trace_file(path)
            if not events:
                continue
            summary = detail[0] if detail else {}
            run_hostname = summary.get("metadata", {}).get("hostname", "")
            if run_hostname and run_hostname != local_hostname:
                skipped_count += 1
                continue
            result = _enrich_run_file(path, events, submissions)
            if result and result.get("pending_match"):
                pending_count += 1
            if result and result.get("wrote_events"):
                enriched_count += 1
                store.invalidate_run(run_id)
            else:
                skipped_count += 1

        from tripletex_agent.openrouter import OpenRouterClient

        settings = get_settings()
        openrouter = OpenRouterClient(settings)
        try:
            for path in sorted(RUNS_DIR.glob("*.jsonl"), reverse=True):
                if path.name == "raw_requests.jsonl":
                    continue
                run_id = store.get_run_id(path)
                detail = store.get_detail(run_id)
                if not detail:
                    continue

                summary, events = detail
                run_hostname = summary.get("metadata", {}).get("hostname", "")
                if run_hostname and run_hostname != local_hostname:
                    continue
                if summary.get("source") != "competition":
                    continue
                if summary.get("status") not in ("completed", "error"):
                    continue

                event_types = {e["event_type"] for e in events}
                if "post_mortem" in event_types:
                    continue
                if "competition_scoring" not in event_types:
                    continue
                scoring_evt = next(
                    e for e in events if e["event_type"] == "competition_scoring"
                )
                if scoring_evt.get("payload", {}).get("score_raw") is None:
                    continue

                logger.debug(
                    "Generating post-mortem for run_id=%s payload_chars=%d",
                    run_id,
                    len(build_analysis_payload(events)),
                )
                analysis = await generate_postmortem(
                    events=events,
                    openrouter_client=openrouter,
                    model=settings.enforcer_model,
                )
                if analysis:
                    fresh_events = store.parse_trace_file(path)
                    if not any(
                        e.get("event_type") == "post_mortem" for e in fresh_events
                    ):
                        append_postmortem_event(path, analysis)
                        store.invalidate_run(run_id)
                        postmortem_count += 1
        finally:
            await openrouter.close()

    overview = _generate_overview()

    if enriched_count > 0 or postmortem_count > 0:
        await broadcast_run_list_update()

    return JSONResponse(
        {
            "enriched": enriched_count,
            "skipped": skipped_count,
            "pending_match": pending_count,
            "postmortem_generated": postmortem_count,
            "overview_runs": overview.get("total_runs", 0),
        }
    )


@router.get("/api/runs/overview")
async def runs_overview() -> JSONResponse:
    overview_path = RUNS_DIR / "overview.json"
    if overview_path.exists() and not _is_overview_stale():
        try:
            with overview_path.open("r", encoding="utf-8") as f:
                return JSONResponse(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    overview = _generate_overview()
    return JSONResponse(overview)


@router.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> JSONResponse:
    store = get_run_store()
    result = store.get_detail(run_id)
    if result:
        summary, events = result
        if summary.get("source") == "competition":
            submissions = await _get_cached_submissions()
            _attach_competition_scores([summary], submissions)
        return JSONResponse(
            {
                "summary": summary,
                "events": events,
            }
        )
    return JSONResponse({"error": "Run not found"}, status_code=404)


@router.delete("/api/runs/{run_id}")
async def delete_run(run_id: str) -> JSONResponse:
    store = get_run_store()
    cached = store._by_run_id.get(run_id)
    if not cached:
        for s in store.list_summaries():
            if s["run_id"] == run_id:
                cached = store._by_run_id.get(run_id)
                break
    if not cached:
        return JSONResponse({"error": "Run not found"}, status_code=404)
    try:
        cached.path.unlink()
        store.invalidate_run(run_id)
        return JSONResponse({"status": "deleted", "run_id": run_id})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/runs/{run_id}/postmortem")
async def generate_run_postmortem(run_id: str) -> JSONResponse:
    store = get_run_store()
    detail = store.get_detail(run_id)
    if not detail:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    summary, events = detail
    if summary.get("source") != "competition":
        return JSONResponse(
            {"error": "Post-mortem is only supported for competition runs"},
            status_code=400,
        )

    event_types = {e["event_type"] for e in events}

    if "post_mortem" in event_types:
        pm_event = next(e for e in events if e["event_type"] == "post_mortem")
        return JSONResponse({"status": "exists", "post_mortem": pm_event["payload"]})

    if "competition_scoring" not in event_types:
        return JSONResponse(
            {"error": "Run not yet scored — enrich first"},
            status_code=400,
        )

    from tripletex_agent.openrouter import OpenRouterClient

    settings = get_settings()
    openrouter = OpenRouterClient(settings)
    try:
        logger.debug(
            "Generating post-mortem for single run_id=%s payload_chars=%d",
            run_id,
            len(build_analysis_payload(events)),
        )
        analysis = await generate_postmortem(
            events=events,
            openrouter_client=openrouter,
            model=settings.enforcer_model,
        )
    finally:
        await openrouter.close()

    if not analysis:
        return JSONResponse({"error": "Post-mortem generation failed"}, status_code=500)

    cached = store._by_run_id.get(run_id)
    if cached:
        fresh_events = store.parse_trace_file(cached.path)
        if any(e.get("event_type") == "post_mortem" for e in fresh_events):
            pm_event = next(e for e in fresh_events if e["event_type"] == "post_mortem")
            return JSONResponse(
                {"status": "exists", "post_mortem": pm_event["payload"]}
            )
        append_postmortem_event(cached.path, analysis)
        store.invalidate_run(run_id)

    return JSONResponse({"status": "generated", "post_mortem": analysis})


def _randomize_prompt(prompt: str) -> str:
    import random as _rnd
    import re as _re

    _FIRST_NAMES = [
        "Emma",
        "Liam",
        "Sofia",
        "Noah",
        "Olivia",
        "Henrik",
        "Ingrid",
        "Magnus",
        "Astrid",
        "Erik",
        "Freya",
        "Lars",
        "Marta",
        "Oscar",
        "Linnea",
        "Aksel",
        "Nora",
        "Jonas",
        "Sigrid",
        "Tobias",
    ]
    _LAST_NAMES = [
        "Andersen",
        "Berg",
        "Dahl",
        "Eriksen",
        "Hansen",
        "Johansen",
        "Larsen",
        "Nilsen",
        "Olsen",
        "Pedersen",
        "Schmidt",
        "Weber",
        "Fischer",
        "Müller",
        "Dubois",
        "Martin",
        "Silva",
        "Costa",
    ]
    _COMPANY_SUFFIXES = ["AS", "ASA", "Lda", "SARL", "GmbH", "SL", "Ltd"]
    _COMPANY_WORDS = [
        "Nordic",
        "Fjord",
        "Summit",
        "Arctic",
        "Baltic",
        "Coastal",
        "Alpine",
        "Terra",
        "Skyline",
        "Vertex",
        "Horizon",
        "Crest",
    ]

    seen_names: dict[str, str] = {}

    def _replace_name(match: _re.Match) -> str:  # type: ignore[type-arg]
        original = match.group(0)
        if original in seen_names:
            return seen_names[original]
        replacement = f"{_rnd.choice(_FIRST_NAMES)} {_rnd.choice(_LAST_NAMES)}"
        seen_names[original] = replacement
        return replacement

    name_pattern = _re.compile(
        r"\b(?:[A-ZÆØÅÉÈÊËÀÂÄÖÜ][a-zæøåéèêëàâäöü]+)\s+"
        r"(?:[A-ZÆØÅÉÈÊËÀÂÄÖÜ][a-zæøåéèêëàâäöü]+)\b"
    )

    skip_words = {
        "Tripletex",
        "Cloud Run",
        "Google Cloud",
        "Clas Ohlson",
        "Porto Alegre",
    }
    result = prompt
    names_found = name_pattern.findall(result)
    for name in names_found:
        if name not in skip_words and "@" not in name:
            new_name = f"{_rnd.choice(_FIRST_NAMES)} {_rnd.choice(_LAST_NAMES)}"
            result = result.replace(name, new_name, 1)

    org_pattern = _re.compile(r"\b(\d{9})\b")
    for match in org_pattern.finditer(result):
        old = match.group(1)
        new_org = str(_rnd.randint(800000000, 999999999))
        result = result.replace(old, new_org, 1)

    email_pattern = _re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
    for match in email_pattern.finditer(result):
        old_email = match.group(0)
        new_email = f"{_rnd.choice(_FIRST_NAMES).lower()}.{_rnd.choice(_LAST_NAMES).lower()}@example.org"
        result = result.replace(old_email, new_email, 1)

    return result


@router.post("/api/runs/simulate")
async def simulate_run() -> JSONResponse:
    import asyncio
    import random as _random

    from tripletex_agent.agent import TripletexAccountingAgent
    from tripletex_agent.config import get_settings as _get_settings
    from tripletex_agent.schemas import SolveRequest, TripletexCredentials

    settings = _get_settings()
    if (
        not settings.tripletex_sandbox_api_url
        or not settings.tripletex_sandbox_api_session_token
    ):
        return JSONResponse(
            {"error": "No sandbox credentials configured in .env"}, status_code=400
        )

    store = get_run_store()
    summaries = store.list_summaries()
    competition_prompts = [
        s["prompt"]
        for s in summaries
        if s.get("source") == "competition" and s.get("prompt")
    ]
    if not competition_prompts:
        return JSONResponse(
            {"error": "No previous competition prompts to simulate from"},
            status_code=400,
        )

    prompt = _randomize_prompt(_random.choice(competition_prompts))

    async def _run_sim() -> None:
        agent = TripletexAccountingAgent(settings)
        try:
            req = SolveRequest(
                prompt=prompt,
                tripletex_credentials=TripletexCredentials(
                    base_url=str(settings.tripletex_sandbox_api_url or ""),
                    session_token=str(
                        settings.tripletex_sandbox_api_session_token or ""
                    ),
                ),
            )
            await agent.solve(req)
        except Exception as exc:
            logger.warning("Simulation run failed: %s", exc)

    asyncio.create_task(_run_sim())
    return JSONResponse({"status": "started", "prompt_preview": prompt[:200]})


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
            "local_solve_url": s.local_solve_url,
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
    endpoint_url = body.get("endpoint_url") or settings.local_solve_url
    endpoint_api_key = body.get("endpoint_api_key") or settings.app_api_key

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

    sub_id = data.get("id", "")
    if sub_id:
        register_pending_submission(sub_id)

    return JSONResponse(
        {
            "submission_id": sub_id,
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
    "running": 0,
    "concurrency": 0,
    "current": None,
    "status": "idle",
}
_batch_runner_task: asyncio.Task[None] | None = None
_batch_runner_worker_tasks: set[asyncio.Task[Any]] = set()


@router.post("/api/competition/batch")
async def competition_batch(request_body: dict | None = None) -> JSONResponse:
    global _batch_runner_active, _batch_runner_results, _batch_runner_progress
    global _batch_runner_task
    body = request_body or {}
    count = max(1, min(int(body.get("count", 5)), 50))
    concurrency = max(1, min(int(body.get("concurrency", 8)), 10))
    concurrency = min(concurrency, count)

    if _batch_runner_active:
        return JSONResponse(
            {
                "error": "Batch runner already active",
                "progress": _batch_runner_progress,
            },
            status_code=409,
        )

    _batch_runner_task = asyncio.create_task(_run_batch(count, concurrency))
    return JSONResponse(
        {
            "started": True,
            "count": count,
            "concurrency": concurrency,
            "max_concurrency": 10,
        }
    )


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
    global _batch_runner_active, _batch_runner_task, _batch_runner_worker_tasks
    _batch_runner_active = False
    _batch_runner_progress["status"] = "stopping"

    if _batch_runner_task and not _batch_runner_task.done():
        _batch_runner_task.cancel()

    for task in list(_batch_runner_worker_tasks):
        if not task.done():
            task.cancel()

    return JSONResponse({"stopped": True, "progress": _batch_runner_progress})


@router.get("/api/raw-requests")
async def get_raw_requests() -> JSONResponse:
    from tripletex_agent.main import RAW_LOG_PATH

    entries: list[dict] = []
    if RAW_LOG_PATH.exists():
        try:
            with RAW_LOG_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            pass
    entries.reverse()
    return JSONResponse({"requests": entries[:200]})


def _batch_get_active_run_id_for_submission(submission_id: str) -> str | None:
    for run_id, trace in RunTrace.get_active_runs().items():
        if trace.metadata.get("submission_id") == submission_id:
            return run_id
    return None


def _batch_get_summary_for_submission(submission_id: str) -> dict[str, Any] | None:
    store = get_run_store()
    for summary in store.list_summaries():
        if summary.get("metadata", {}).get("submission_id") == submission_id:
            return summary
    return None


async def _wait_for_batch_submission(
    submission_id: str,
    *,
    poll_seconds: int = 5,
    start_timeout_seconds: int = 60,
    finish_timeout_seconds: int = 5 * 60,
) -> str:
    start_checks = max(1, start_timeout_seconds // poll_seconds)
    active_run_id: str | None = None

    for attempt in range(start_checks):
        if not _batch_runner_active:
            return "stopped"

        active_run_id = _batch_get_active_run_id_for_submission(submission_id)
        if active_run_id:
            break

        summary = _batch_get_summary_for_submission(submission_id)
        if summary:
            status = summary.get("status", "")
            if status in ("completed", "error", "incomplete"):
                return "done"
            if status == "running":
                active_run_id = summary.get("run_id")
                break

        if attempt < start_checks - 1:
            await asyncio.sleep(poll_seconds)
    else:
        return "timeout"

    finish_checks = max(1, finish_timeout_seconds // poll_seconds)
    for _ in range(finish_checks):
        if not _batch_runner_active:
            return "stopped"

        active_runs = RunTrace.get_active_runs()
        if active_run_id:
            if active_run_id not in active_runs:
                return "done"
        elif not any(
            trace.metadata.get("submission_id") == submission_id
            for trace in active_runs.values()
        ):
            return "done"

        await asyncio.sleep(poll_seconds)

    return "timeout"


async def _resolve_batch_endpoint_url(settings: Settings) -> str:
    endpoint_url = None
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            ngrok_resp = await c.get("http://localhost:4040/api/tunnels")
            tunnels = ngrok_resp.json().get("tunnels", [])
            for tunnel in tunnels:
                public_url = tunnel.get("public_url", "")
                if "ngrok" in public_url:
                    endpoint_url = public_url + "/solve"
                    break
    except Exception:
        pass

    return endpoint_url or settings.local_solve_url


async def _run_batch(count: int, concurrency: int) -> None:
    global _batch_runner_active, _batch_runner_results, _batch_runner_progress
    global _batch_runner_task, _batch_runner_worker_tasks

    _batch_runner_active = True
    _batch_runner_results = []
    _batch_runner_progress = {
        "total": count,
        "completed": 0,
        "running": 0,
        "concurrency": concurrency,
        "current": None,
        "status": f"starting batch: {count} submissions @ concurrency {concurrency}",
    }

    settings = get_settings()
    token = settings.ainm_jwt_token
    task_id = settings.ainm_tripletex_task_id
    endpoint_api_key = settings.app_api_key
    endpoint_url = await _resolve_batch_endpoint_url(settings)
    semaphore = asyncio.Semaphore(concurrency)
    progress_lock = asyncio.Lock()

    if not token:
        _batch_runner_progress["status"] = "failed: AINM_JWT_TOKEN not configured"
        _batch_runner_active = False
        _batch_runner_task = None
        return

    async def _submit_one(run_number: int) -> None:
        nonlocal endpoint_url, token, task_id, endpoint_api_key

        async with semaphore:
            if not _batch_runner_active:
                return

            async with progress_lock:
                _batch_runner_progress["running"] += 1
                _batch_runner_progress["current"] = run_number
                _batch_runner_progress["status"] = (
                    f"running {_batch_runner_progress['running']}/{concurrency} workers, "
                    f"{_batch_runner_progress['completed']}/{count} done"
                )

            sub_id = ""
            outcome = "error"

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
                    sub_id = sub_data.get("id", "")

                if sub_id:
                    register_pending_submission(sub_id)
                    outcome = await _wait_for_batch_submission(sub_id)
                    _batch_runner_results.append(
                        {
                            "run": run_number,
                            "submission_id": sub_id[:12],
                            "status": outcome,
                        }
                    )
                else:
                    _batch_runner_results.append(
                        {
                            "run": run_number,
                            "status": "error",
                            "error": "missing submission id",
                        }
                    )

            except asyncio.CancelledError:
                _batch_runner_results.append(
                    {
                        "run": run_number,
                        "submission_id": sub_id[:12] if sub_id else "",
                        "status": "cancelled",
                    }
                )
                raise
            except Exception as exc:
                _batch_runner_results.append(
                    {
                        "run": run_number,
                        "submission_id": sub_id[:12] if sub_id else "",
                        "status": "error",
                        "error": str(exc)[:200],
                    }
                )
            finally:
                async with progress_lock:
                    _batch_runner_progress["running"] = max(
                        0, _batch_runner_progress["running"] - 1
                    )
                    _batch_runner_progress["completed"] += 1
                    status_word = "stopping" if not _batch_runner_active else "running"
                    _batch_runner_progress["status"] = (
                        f"{status_word}: {_batch_runner_progress['running']}/{concurrency} workers, "
                        f"{_batch_runner_progress['completed']}/{count} done"
                    )

    _batch_runner_worker_tasks = {
        asyncio.create_task(_submit_one(i + 1)) for i in range(count)
    }

    try:
        await asyncio.gather(*_batch_runner_worker_tasks, return_exceptions=True)
    finally:
        _batch_runner_worker_tasks.clear()
        _batch_runner_progress["running"] = 0
        _batch_runner_progress["status"] = (
            "completed"
            if _batch_runner_active
            else f"stopped ({_batch_runner_progress['completed']}/{count} completed)"
        )
        _batch_runner_active = False
        _batch_runner_task = None


def _enrich_run_file(
    path: Path,
    events: list[dict[str, Any]],
    submissions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Append competition_scoring and error_summary events to a run JSONL if missing.

    Returns the enrichment data if newly added, None if already enriched or not applicable.
    """
    existing_types = {e["event_type"] for e in events}

    summary = _summarize_run(path, events)
    run_id = summary.get("run_id", path.stem)

    # Only enrich completed competition runs
    if summary["source"] != "competition" or summary["status"] not in (
        "completed",
        "error",
    ):
        return None

    scoring_events = [e for e in events if e["event_type"] == "competition_scoring"]
    existing_scoring = next(
        (
            e
            for e in scoring_events
            if e.get("payload", {}).get("score_raw") is not None
        ),
        scoring_events[-1] if scoring_events else None,
    )
    has_final_scoring = (
        existing_scoring is not None
        and existing_scoring.get("payload", {}).get("status") in ("completed", "failed")
        and existing_scoring.get("payload", {}).get("score_raw") is not None
    )
    has_error_summary = "error_summary" in existing_types

    run_endpoint_url = ""
    init_event = next((e for e in events if e["event_type"] == "init"), None)
    if init_event:
        payload = init_event.get("payload", {})
        metadata = payload.get("metadata", {})
        run_endpoint_url = (
            metadata.get("endpoint_url")
            or payload.get("endpoint_url")
            or payload.get("solve_url")
            or ""
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

    import time

    pending_match = False
    scoring_data: dict[str, Any] = {}
    should_write_scoring = False

    if not has_final_scoring:
        retry_state = _enrichment_retry_state.get(
            run_id, {"attempts": 0, "next_retry_at": 0.0}
        )
        now_monotonic = time.monotonic()
        next_retry_at = float(retry_state.get("next_retry_at", 0.0))
        attempts = int(retry_state.get("attempts", 0))

        if now_monotonic >= next_retry_at:
            run_sub_id = summary.get("metadata", {}).get("submission_id")
            match = None
            if run_sub_id:
                match = next(
                    (s for s in submissions if s.get("id") == run_sub_id), None
                )
            if not match:
                match = _match_submission_to_run(
                    summary["started_at"],
                    summary.get("duration_seconds"),
                    submissions,
                    run_endpoint_url=run_endpoint_url,
                )

            if match:
                sub_status = match.get("status", "")
                if sub_status in ("completed", "failed"):
                    feedback = match.get("feedback", {})
                    checks = feedback.get("checks", [])
                    passed = sum(1 for c in checks if "passed" in c.lower())
                    scoring_data = {
                        "submission_id": match.get("id"),
                        "status": sub_status,
                        "score_raw": match.get("score_raw", 0),
                        "score_max": match.get("score_max", 0),
                        "normalized_score": match.get("normalized_score", 0),
                        "checks_passed": passed,
                        "checks_total": len(checks),
                        "checks": checks,
                        "comment": feedback.get("comment", ""),
                    }
                    should_write_scoring = True
                    _enrichment_retry_state.pop(run_id, None)
                else:
                    pending_match = True
            else:
                pending_match = True
                attempts += 1
                backoff_seconds = min(30.0, 5.0 * attempts)
                _enrichment_retry_state[run_id] = {
                    "attempts": attempts,
                    "next_retry_at": now_monotonic + backoff_seconds,
                }
        else:
            pending_match = True
    else:
        _enrichment_retry_state.pop(run_id, None)

    should_write_error_summary = bool(errors) and not has_error_summary
    if not should_write_scoring and not should_write_error_summary:
        return {"wrote_events": False, "pending_match": pending_match}

    now = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as f:
        if should_write_scoring:
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
        if should_write_error_summary:
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

    return {
        "wrote_events": True,
        "pending_match": pending_match,
        "scoring": scoring_data if should_write_scoring else None,
        "error_count": len(errors),
    }


def _is_overview_stale() -> bool:
    overview_path = RUNS_DIR / "overview.json"
    if not overview_path.exists():
        return True
    overview_mtime = overview_path.stat().st_mtime
    run_files = list(RUNS_DIR.glob("*.jsonl"))
    run_files = [f for f in run_files if f.name != "raw_requests.jsonl"]
    if not run_files:
        return False
    try:
        with overview_path.open() as f:
            overview = json.load(f)
        overview_count = overview.get("total_runs", 0)
    except Exception:
        return True
    if len(run_files) != overview_count:
        return True
    newest_run_mtime = max(f.stat().st_mtime for f in run_files)
    return newest_run_mtime > overview_mtime


def _generate_overview() -> dict[str, Any]:
    """Generate overview.json with one entry per run."""
    runs_data: list[dict[str, Any]] = []
    store = get_run_store()
    for summary in store.list_summaries():
        detail = store.get_detail(summary["run_id"])
        events = detail[1] if detail else []
        if not events:
            continue

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
        postmortem_event = next(
            (e for e in events if e["event_type"] == "post_mortem"), None
        )

        runs_data.append(
            {
                "filename": summary["filename"],
                "run_id": summary["run_id"],
                "started_at": summary["started_at"],
                "task_type": summary["task_type"],
                "task_tier": summary.get("metadata", {}).get("task_tier"),
                "executor_model": summary.get("metadata", {}).get("executor_model", ""),
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
                "post_mortem": postmortem_event.get("payload", {})
                if postmortem_event
                else None,
                "total_errors": error_data.get("total_errors", 0),
                "errors": [e.get("error", "") for e in error_data.get("errors", [])],
                "prompt_preview": summary["prompt"][:120],
            }
        )

    overview = {
        "runs": runs_data,
        "total_runs": len(runs_data),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Write to disk
    overview_path = RUNS_DIR / "overview.json"
    with overview_path.open("w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    return overview


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    nextjs_index = (
        Path(__file__).resolve().parents[2] / "dashboard" / "out" / "index.html"
    )
    if nextjs_index.exists():
        return HTMLResponse(nextjs_index.read_text(encoding="utf-8"))
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
.ws-status { font-size: 11px; padding: 2px 8px; border-radius: 10px; margin-left: auto; }
.ws-live { color: var(--green); }
.ws-reconnecting { color: var(--yellow); animation: pulse 1.5s infinite; }
.ws-offline { color: var(--red); }
.ws-connecting { color: var(--text2); }
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
    <span id="ws-status" class="ws-status ws-connecting" title="WebSocket status">&#9679; connecting</span>
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
let defaultSolveUrl = '';
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

  // Extract post-mortem from events
  const pmEvent = events.find(e => e.event_type === 'post_mortem');
  const pm = pmEvent ? pmEvent.payload : null;

  // Post-mortem card HTML
  const postMortemHtml = pm ? `
    <div style="background:linear-gradient(135deg, rgba(99,102,241,.08), rgba(168,85,247,.08));border:1px solid rgba(139,92,246,.25);border-radius:10px;padding:16px 20px;margin-bottom:16px;">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
        <span style="font-size:16px;">🔍</span>
        <h3 style="margin:0;font-size:15px;font-weight:700;color:var(--purple);">Post-Mortem Analysis</h3>
        <span class="badge" style="margin-left:auto;background:rgba(139,92,246,.15);color:var(--purple);font-size:10px;">${pm.confidence || 'unknown'} confidence</span>
        <span class="badge" style="background:rgba(139,92,246,.1);color:var(--text2);font-size:10px;">${pm.root_cause_category || ''}</span>
      </div>
      <div style="font-size:14px;font-weight:600;color:var(--text);margin-bottom:12px;">${pm.headline || ''}</div>
      ${pm.what_went_wrong && pm.what_went_wrong.length > 0 ? `
        <div style="margin-bottom:10px;">
          <div style="font-size:11px;font-weight:600;color:var(--red);text-transform:uppercase;margin-bottom:4px;">Issues Found</div>
          ${pm.what_went_wrong.map(w => `<div style="font-size:13px;color:var(--text);padding:4px 0;padding-left:12px;border-left:2px solid var(--red);">• ${w}</div>`).join('')}
        </div>
      ` : ''}
      ${pm.what_worked && pm.what_worked.length > 0 ? `
        <div style="margin-bottom:10px;">
          <div style="font-size:11px;font-weight:600;color:var(--green);text-transform:uppercase;margin-bottom:4px;">What Worked</div>
          ${pm.what_worked.map(w => `<div style="font-size:13px;color:var(--text);padding:4px 0;padding-left:12px;border-left:2px solid var(--green);">✓ ${w}</div>`).join('')}
        </div>
      ` : ''}
      ${pm.what_to_try_next && pm.what_to_try_next.length > 0 ? `
        <div>
          <div style="font-size:11px;font-weight:600;color:var(--blue);text-transform:uppercase;margin-bottom:4px;">Recommendations</div>
          ${pm.what_to_try_next.map(w => `<div style="font-size:13px;color:var(--text);padding:4px 0;padding-left:12px;border-left:2px solid var(--blue);">→ ${w}</div>`).join('')}
        </div>
      ` : ''}
    </div>
  ` : (s.source === 'competition' && s.status !== 'running' ? `
    <div style="background:var(--surface);border:1px dashed var(--border);border-radius:10px;padding:12px 16px;margin-bottom:16px;text-align:center;">
      <span style="color:var(--text2);font-size:13px;">No post-mortem yet</span>
      <button onclick="triggerPostMortem('${s.run_id}')" style="margin-left:12px;background:var(--purple);color:#fff;border:none;border-radius:6px;padding:6px 14px;font-size:12px;font-weight:600;cursor:pointer;">Generate Analysis</button>
    </div>
  ` : '');

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
    ${postMortemHtml}
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

async function triggerPostMortem(runId) {
  try {
    const res = await fetch(`/api/runs/${runId}/postmortem`, {method: 'POST'});
    const data = await res.json();
    if (data.status === 'generated' || data.status === 'exists') {
      selectRun(runId); // Refresh detail to show the analysis
    }
  } catch (e) {
    console.error('Post-mortem error:', e);
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
          value="${defaultSolveUrl || 'https://tripletex-agent-nmiai.ngrok-free.dev/solve'}"
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
    if (settingsData.local_solve_url && !defaultSolveUrl) defaultSolveUrl = settingsData.local_solve_url;
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

// WebSocket connection
let ws = null;
let wsReconnectTimer = null;
let wsReconnectDelay = 1000;

function updateConnectionStatus(state) {
  const el = document.getElementById('ws-status');
  if (!el) return;
  el.className = 'ws-status ws-' + state;
  const labels = { live: '● live', reconnecting: '● reconnecting', offline: '● offline', connecting: '● connecting' };
  el.innerHTML = labels[state] || state;
  
  if (state === 'live') {
    stopHttpFallback();
  } else if (state === 'reconnecting' || state === 'offline') {
    startHttpFallback();
  }
}

function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/dashboard/ws`;
  
  updateConnectionStatus('connecting');
  ws = new WebSocket(wsUrl);
  
  ws.onopen = () => {
    updateConnectionStatus('live');
    wsReconnectDelay = 1000; // Reset backoff
  };
  
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    handleWsMessage(msg);
  };
  
  ws.onclose = () => {
    updateConnectionStatus('reconnecting');
    wsReconnectTimer = setTimeout(() => {
      wsReconnectDelay = Math.min(wsReconnectDelay * 1.5, 15000);
      connectWebSocket();
    }, wsReconnectDelay);
  };
  
  ws.onerror = () => {
    ws.close();
  };
  
  // Keepalive ping every 25s
  setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send('ping');
    }
  }, 25000);
}

function handleWsMessage(msg) {
  switch (msg.type) {
    case 'snapshot':
      handleSnapshot(msg.payload);
      break;
    case 'settings':
      handleSettings(msg.payload);
      break;
    case 'run_upsert':
    case 'run_status_changed':
      handleRunUpdate(msg.payload);
      break;
    case 'run_event':
      handleRunEvent(msg.payload);
      break;
    case 'heartbeat':
    case 'pong':
      break; // Connection alive
  }
}

function handleSnapshot(payload) {
  const newRuns = payload.runs || [];
  
  // Sound & flash triggers (detect state transitions)
  const newStatuses = {};
  const newScoreStatuses = {};
  for (const run of newRuns) {
    newStatuses[run.run_id] = run.status;
    const prev = previousRunStatuses[run.run_id];
    if (!prev && run.status === 'running') {
      playStartSound();
      flashHeader('start');
    } else if (prev === 'running' && run.status === 'completed') {
      playCompletionSound();
      flashHeader('done');
    } else if (prev === 'running' && run.status === 'error') {
      playErrorSound();
      flashHeader('error');
    }
    
    if (run.competition_score) {
      newScoreStatuses[run.run_id] = run.competition_score.status;
      const prevScore = previousScoreStatuses[run.run_id];
      if (prevScore && ['in_progress', 'pending'].includes(prevScore) && run.competition_score.status === 'completed') {
        playCompletionSound();
        flashHeader('done');
      }
    }
  }
  previousRunStatuses = newStatuses;
  previousScoreStatuses = newScoreStatuses;
  
  allRuns = newRuns;
  document.getElementById('hdr-active').textContent = payload.active_count;
  document.getElementById('hdr-total').textContent = payload.total_count;
  
  renderRunList();
  
  // Auto-refresh selected run if running
  if (selectedRunId) {
    const selectedRun = allRuns.find(r => r.run_id === selectedRunId);
    if (selectedRun && (selectedRun.status === 'running' || (selectedRun.competition_score && ['in_progress', 'pending'].includes(selectedRun.competition_score.status)))) {
      selectRun(selectedRunId); // Refresh detail
    }
  }
}

function handleSettings(payload) {
  const shortModel = (m) => m ? m.split('/').pop().replace(':exacto','') : '-';
  document.getElementById('hdr-planner').textContent = shortModel(payload.planner_model);
  document.getElementById('hdr-t1').textContent = shortModel(payload.tier1_executor_model);
  document.getElementById('hdr-t2').textContent = shortModel(payload.tier2_executor_model);
  document.getElementById('hdr-t3').textContent = shortModel(payload.tier3_executor_model);
}

function handleRunUpdate(payload) {
  // A single run was updated — request fresh snapshot
  // (The server sends full snapshots on significant events, so this is just a trigger)
}

function handleRunEvent(payload) {
  // Lightweight event for in-progress runs
  // If it's the selected run, refresh detail
  if (payload.run_id === selectedRunId) {
    selectRun(selectedRunId);
  }
}

// Fallback: HTTP polling only when WebSocket is unavailable
let httpFallbackInterval = null;

function startHttpFallback() {
  if (httpFallbackInterval) return;
  httpFallbackInterval = setInterval(refresh, 4000);
  refresh(); // Initial load
}

function stopHttpFallback() {
  if (httpFallbackInterval) {
    clearInterval(httpFallbackInterval);
    httpFallbackInterval = null;
  }
}

// Initialize: try WebSocket first, fall back to HTTP polling
connectWebSocket();
// Periodic enrichment
setInterval(() => { fetch('/api/runs/enrich', {method: 'POST'}).catch(e => console.error(e)); }, 15000);

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
