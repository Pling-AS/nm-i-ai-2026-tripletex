"""API client for Astar Island endpoints."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE = "https://api.ainm.no"
MIN_REQUEST_INTERVAL = 0.5  # 2 req/s, safely under 5 req/s limit


@dataclass
class SimulationResult:
    """Parsed result from a /simulate call."""

    grid: list[list[int]]
    settlements: list[dict[str, Any]]
    viewport: dict[str, int]
    width: int
    height: int
    queries_used: int
    queries_max: int


@dataclass
class RoundDetail:
    """Parsed round details including initial states."""

    id: str
    round_number: int
    status: str
    map_width: int
    map_height: int
    seeds_count: int
    initial_states: list[dict[str, Any]]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


class AstarClient:
    """Thin REST client for the Astar Island API."""

    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._base = (base_url or os.getenv("API_BASE", DEFAULT_BASE)).rstrip("/")
        self._token = token or os.environ["ACCESS_TOKEN"]
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {self._token}"
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _rate_limit(self) -> None:
        """Enforce minimum interval between requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    def _get(self, path: str) -> Any:
        self._rate_limit()
        url = f"{self._base}{path}"
        resp = self._session.get(url, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        self._rate_limit()
        url = f"{self._base}{path}"
        resp = self._session.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_rounds(self) -> list[dict[str, Any]]:
        """GET /astar-island/rounds — list all rounds."""
        return self._get("/astar-island/rounds")

    def get_active_round(self) -> dict[str, Any] | None:
        """Return the first active round, or None."""
        rounds = self.get_rounds()
        return next((r for r in rounds if r["status"] == "active"), None)

    def get_round_detail(self, round_id: str) -> RoundDetail:
        """GET /astar-island/rounds/{round_id} — full detail with initial states."""
        data = self._get(f"/astar-island/rounds/{round_id}")
        return RoundDetail(
            id=data["id"],
            round_number=data["round_number"],
            status=data["status"],
            map_width=data["map_width"],
            map_height=data["map_height"],
            seeds_count=data["seeds_count"],
            initial_states=data["initial_states"],
            raw=data,
        )

    def get_budget(self) -> dict[str, Any]:
        """GET /astar-island/budget — remaining query budget."""
        return self._get("/astar-island/budget")

    def simulate(
        self,
        round_id: str,
        seed_index: int,
        viewport_x: int = 0,
        viewport_y: int = 0,
        viewport_w: int = 15,
        viewport_h: int = 15,
    ) -> SimulationResult:
        """POST /astar-island/simulate — run one stochastic simulation.

        Costs 1 query from the budget.
        """
        data = self._post(
            "/astar-island/simulate",
            {
                "round_id": round_id,
                "seed_index": seed_index,
                "viewport_x": viewport_x,
                "viewport_y": viewport_y,
                "viewport_w": viewport_w,
                "viewport_h": viewport_h,
            },
        )
        return SimulationResult(
            grid=data["grid"],
            settlements=data["settlements"],
            viewport=data["viewport"],
            width=data["width"],
            height=data["height"],
            queries_used=data["queries_used"],
            queries_max=data["queries_max"],
        )

    def submit(
        self,
        round_id: str,
        seed_index: int,
        prediction: list[list[list[float]]],
    ) -> dict[str, Any]:
        """POST /astar-island/submit — submit prediction for one seed.

        prediction: H x W x 6 probability tensor as nested lists.
        """
        return self._post(
            "/astar-island/submit",
            {
                "round_id": round_id,
                "seed_index": seed_index,
                "prediction": prediction,
            },
        )

    def get_my_rounds(self) -> list[dict[str, Any]]:
        """GET /astar-island/my-rounds — rounds with scores and budget."""
        return self._get("/astar-island/my-rounds")

    def get_my_predictions(self, round_id: str) -> list[dict[str, Any]]:
        """GET /astar-island/my-predictions/{round_id} — predictions with argmax/confidence."""
        return self._get(f"/astar-island/my-predictions/{round_id}")

    def get_analysis(self, round_id: str, seed_index: int) -> dict[str, Any]:
        """GET /astar-island/analysis/{round_id}/{seed_index} — post-round analysis."""
        return self._get(f"/astar-island/analysis/{round_id}/{seed_index}")

    def get_leaderboard(self) -> list[dict[str, Any]]:
        """GET /astar-island/leaderboard."""
        return self._get("/astar-island/leaderboard")
