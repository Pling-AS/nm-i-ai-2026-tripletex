import json
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from tripletex_agent.config import BASE_DIR

RUNS_DIR = BASE_DIR / "runs"


@dataclass(slots=True)
class RunTrace:
    run_id: str = field(default_factory=lambda: uuid4().hex)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
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
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def update_metadata(self, updates: dict[str, Any]) -> None:
        """Merge additional routing metadata and emit a trace event."""
        self.metadata.update(updates)
        self.write("metadata_update", updates)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            RunTrace._active.pop(self.run_id, None)

    @classmethod
    def get_active_runs(cls) -> dict[str, "RunTrace"]:
        return dict(cls._active)

    @classmethod
    def get_runs_dir(cls) -> Path:
        return RUNS_DIR
