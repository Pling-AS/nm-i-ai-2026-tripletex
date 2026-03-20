import json
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from uuid import uuid4

from tripletex_agent.config import BASE_DIR


@dataclass(slots=True)
class RunTrace:
    run_id: str = field(default_factory=lambda: uuid4().hex)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    path: Path = field(init=False)

    def __post_init__(self) -> None:
        runs_dir = BASE_DIR / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = self.started_at.strftime("%Y%m%d-%H%M%S")
        self.path = runs_dir / f"{timestamp}_{self.run_id}.jsonl"

    def write(self, event_type: str, payload: dict) -> None:
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
