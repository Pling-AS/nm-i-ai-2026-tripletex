import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, ClassVar, Coroutine
from uuid import uuid4

from tripletex_agent.config import BASE_DIR

RUNS_DIR = BASE_DIR / "runs"

_trace_callbacks: list[
    Callable[["RunTrace", str, dict], Coroutine[Any, Any, None]]
] = []


def register_trace_callback(
    cb: Callable[["RunTrace", str, dict], Coroutine[Any, Any, None]],
) -> None:
    _trace_callbacks.append(cb)


async def _invoke_trace_callback(
    cb: Callable[["RunTrace", str, dict], Coroutine[Any, Any, None]],
    trace: "RunTrace",
    event_type: str,
    payload: dict,
) -> None:
    await cb(trace, event_type, payload)


@dataclass(slots=True)
class RunTrace:
    run_id: str = field(default_factory=lambda: uuid4().hex)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    path: Path = field(init=False)
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    _closed: bool = field(default=False, repr=False)

    # Class-level registry of active (in-progress) runs
    _active: ClassVar[dict[str, "RunTrace"]] = {}

    def __post_init__(self) -> None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = self.started_at.strftime("%Y%m%d-%H%M%S")
        self.path = RUNS_DIR / f"{timestamp}_{self.run_id}.jsonl"
        RunTrace._active[self.run_id] = self

    def write(self, event_type: str, payload: dict) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

        for cb in _trace_callbacks:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_invoke_trace_callback(cb, self, event_type, payload))
            except RuntimeError:
                pass

    def update_metadata(self, updates: dict[str, Any]) -> None:
        """Merge additional routing metadata and emit a trace event."""
        self.metadata.update(updates)
        self.write("metadata_update", updates)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            RunTrace._active.pop(self.run_id, None)
            for cb in _trace_callbacks:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(_invoke_trace_callback(cb, self, "_close", {}))
                except RuntimeError:
                    pass

    @classmethod
    def get_active_runs(cls) -> dict[str, "RunTrace"]:
        return dict(cls._active)

    @classmethod
    def get_runs_dir(cls) -> Path:
        return RUNS_DIR
