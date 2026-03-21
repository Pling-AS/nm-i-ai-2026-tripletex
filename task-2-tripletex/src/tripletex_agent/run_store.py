from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tripletex_agent.prompts import EXECUTOR_SYSTEM_PROMPT
from tripletex_agent.trace import RUNS_DIR, RunTrace


@dataclass
class CachedRun:
    path: Path
    size: int
    mtime: float
    summary: dict[str, Any]
    events: list[dict[str, Any]]


class RunStore:
    """In-memory cache of run summaries and events, backed by JSONL files on disk."""

    def __init__(self, runs_dir: Path = RUNS_DIR) -> None:
        self._runs_dir = runs_dir
        self._by_run_id: dict[str, CachedRun] = {}
        self._path_to_run_id: dict[Path, str] = {}

    def get_run_id(self, path: Path) -> str:
        filename = path.name
        return (
            filename.rsplit("_", 1)[-1].replace(".jsonl", "")
            if "_" in filename
            else filename
        )

    def list_summaries(self) -> list[dict[str, Any]]:
        if not self._runs_dir.exists():
            self.invalidate_all()
            return []

        seen_paths: set[Path] = set()
        for path in self._iter_run_paths():
            seen_paths.add(path)
            run_id = self.get_run_id(path)
            cached = self._by_run_id.get(run_id)
            if cached and cached.path == path and self._is_fresh(cached, path):
                continue
            self.refresh_run(path)

        stale_paths = set(self._path_to_run_id) - seen_paths
        for stale_path in stale_paths:
            stale_run_id = self._path_to_run_id.get(stale_path)
            if stale_run_id:
                self.invalidate_run(stale_run_id)

        summaries = [cached.summary for cached in self._by_run_id.values()]
        summaries.sort(key=lambda s: s.get("started_at", ""), reverse=True)
        return summaries

    def get_detail(
        self, run_id: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        cached = self._by_run_id.get(run_id)
        if cached:
            if cached.path.exists() and self._is_fresh(cached, cached.path):
                return cached.summary, cached.events
            if cached.path.exists():
                self.refresh_run(cached.path)
                refreshed = self._by_run_id.get(run_id)
                if refreshed:
                    return refreshed.summary, refreshed.events

        if self._runs_dir.exists():
            for path in self._iter_run_paths():
                if self.get_run_id(path) == run_id:
                    self.refresh_run(path)
                    refreshed = self._by_run_id.get(run_id)
                    if refreshed:
                        return refreshed.summary, refreshed.events
                    return None
        return None

    def refresh_run(self, path: Path) -> None:
        run_id = self.get_run_id(path)
        if not path.exists() or path.name == "raw_requests.jsonl":
            self.invalidate_run(run_id)
            return

        try:
            stat = path.stat()
        except OSError:
            self.invalidate_run(run_id)
            return

        events = self._parse_trace_file(path)
        if not events:
            self.invalidate_run(run_id)
            return

        summary = self._summarize_run(path, events)
        self._by_run_id[run_id] = CachedRun(
            path=path,
            size=stat.st_size,
            mtime=stat.st_mtime,
            summary=summary,
            events=events,
        )
        self._path_to_run_id[path] = run_id

    def invalidate_run(self, run_id: str) -> None:
        cached = self._by_run_id.pop(run_id, None)
        if cached:
            self._path_to_run_id.pop(cached.path, None)

    def invalidate_all(self) -> None:
        self._by_run_id.clear()
        self._path_to_run_id.clear()

    def get_active_run_summary(self, run_id: str, trace: RunTrace) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "filename": trace.path.name,
            "started_at": trace.started_at.isoformat(),
            "prompt": trace.prompt,
            "task_type": "",
            "goal": "",
            "status": "running",
            "source": trace.metadata.get("source", "unknown"),
            "duration_seconds": round(
                (datetime.now(timezone.utc) - trace.started_at).total_seconds(), 1
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

    def parse_trace_file(self, path: Path) -> list[dict[str, Any]]:
        return self._parse_trace_file(path)

    def summarize_run(self, path: Path, events: list[dict[str, Any]]) -> dict[str, Any]:
        return self._summarize_run(path, events)

    def _iter_run_paths(self) -> list[Path]:
        return [
            path
            for path in sorted(self._runs_dir.glob("*.jsonl"), reverse=True)
            if path.name != "raw_requests.jsonl"
        ]

    def _is_fresh(self, cached: CachedRun, path: Path) -> bool:
        try:
            stat = path.stat()
        except OSError:
            return False
        return cached.size == stat.st_size and cached.mtime == stat.st_mtime

    def _parse_trace_file(self, path: Path) -> list[dict[str, Any]]:
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

    def _summarize_run(
        self, path: Path, events: list[dict[str, Any]]
    ) -> dict[str, Any]:
        filename = path.name
        run_id = self.get_run_id(path)

        init_event = next((e for e in events if e["event_type"] == "init"), None)
        planner_event = next((e for e in events if e["event_type"] == "planner"), None)
        execution_brief_event = next(
            (e for e in events if e["event_type"] == "execution_brief"), None
        )
        done_event = next((e for e in events if e["event_type"] == "done"), None)
        error_event = next((e for e in events if e["event_type"] == "error"), None)
        final_event = next(
            (e for e in events if e["event_type"] == "final_payload"), None
        )

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

        status = "unknown"
        if done_event:
            status = "completed"
        elif error_event:
            status = "error"
        elif run_id in RunTrace.get_active_runs():
            status = "running"
        elif events:
            last_event = events[-1]["event_type"]
            status = "error" if last_event == "error" else "incomplete"

        call_count = 0
        error_count = 0
        call_log: list[dict[str, Any]] = []
        if done_event:
            payload = done_event.get("payload", {})
            call_count = payload.get("tripletex_call_count", 0)
            error_count = payload.get("tripletex_error_count", 0)
            call_log = payload.get("tripletex_call_log", [])

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

        source = metadata.get("source", "")
        if not source:
            source = "competition" if "tx-proxy" in base_url else "simulation"

        tool_starts = [e for e in events if e["event_type"] == "tool_start"]
        tool_errors = [
            e
            for e in events
            if e["event_type"] == "tool_result"
            and not e.get("payload", {}).get("result", {}).get("ok", True)
        ]

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

        summary_text = ""
        if final_event:
            summary_text = final_event.get("payload", {}).get("summary", "")

        error_message = ""
        if error_event:
            error_message = error_event.get("payload", {}).get("message", "")

        enforcer_rejections = [
            e.get("payload", {})
            for e in events
            if e["event_type"] == "enforcer_rejected"
        ]
        semantic_rejections = [
            e.get("payload", {})
            for e in events
            if e["event_type"] == "semantic_enforcer_rejected"
        ]
        enforcer_overrides = [
            e.get("payload", {})
            for e in events
            if e["event_type"] == "enforcer_override"
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


_store: RunStore | None = None


def get_run_store() -> RunStore:
    global _store
    if _store is None:
        _store = RunStore()
    return _store
