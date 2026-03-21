import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, UTC
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from tripletex_agent.agent import TripletexAccountingAgent
from tripletex_agent.config import Settings, get_settings
from tripletex_agent.dashboard import router as dashboard_router
from tripletex_agent.openrouter import OpenRouterError
from tripletex_agent.schemas import SolveRequest, SolveResponse, TripletexCredentials
from tripletex_agent.trace import RUNS_DIR

logger = logging.getLogger(__name__)

# Raw request log file (all HTTP requests, not just /solve)
RAW_LOG_PATH = RUNS_DIR / "raw_requests.jsonl"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO)
    )
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Tripletex Accounting Agent", lifespan=lifespan)
app.include_router(dashboard_router)

_nextjs_static = Path(__file__).resolve().parents[2] / "dashboard" / "out"
if (_nextjs_static / "_next").exists():
    app.mount(
        "/dashboard/_next",
        StaticFiles(directory=str(_nextjs_static / "_next")),
        name="nextjs-static",
    )


@app.middleware("http")
async def log_all_requests(request: Request, call_next):
    path = request.url.path
    # Skip logging for dashboard and high-frequency API routes
    if path in (
        "/api/runs",
        "/api/settings",
        "/api/competition/submissions",
        "/api/competition/submit",
        "/dashboard",
        "/dashboard/ws",
        "/favicon.ico",
        "/health",
    ):
        return await call_next(request)

    start = time.monotonic()
    body_bytes = b""
    # Only read body for POST/PUT — skip for GET to avoid stream exhaustion
    if request.method in ("POST", "PUT"):
        body_bytes = await request.body()

    response = await call_next(request)
    elapsed = round(time.monotonic() - start, 3)

    body_preview = ""
    if body_bytes:
        try:
            body_preview = body_bytes[:2000].decode("utf-8", errors="replace")
        except Exception:
            body_preview = f"<{len(body_bytes)} bytes>"

    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "method": request.method,
        "path": path,
        "query": str(request.url.query) if request.url.query else "",
        "client": request.client.host if request.client else "unknown",
        "status_code": response.status_code,
        "elapsed_seconds": elapsed,
        "headers": {
            k: (v if k.lower() != "authorization" else "[REDACTED]")
            for k, v in request.headers.items()
            if k.lower()
            in (
                "content-type",
                "authorization",
                "user-agent",
                "x-forwarded-for",
                "x-real-ip",
                "ngrok-skip-browser-warning",
            )
        },
        "body_preview": _redact_body(body_preview),
    }
    logger.info(
        "HTTP %s %s → %d (%.3fs)", request.method, path, response.status_code, elapsed
    )
    try:
        with RAW_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass
    return response


def _redact_body(body_str: str) -> str:
    """Redact sensitive fields from JSON body preview."""
    if not body_str.strip().startswith("{"):
        return body_str
    try:
        data = json.loads(body_str)
        if "tripletex_credentials" in data:
            creds = data["tripletex_credentials"]
            if isinstance(creds, dict):
                # Redact session_token if present
                if "session_token" in creds:
                    creds["session_token"] = "[REDACTED]"
                # Also prevent logging base_url if sensitive (usually not, but consistent)
            data["tripletex_credentials"] = creds
        return json.dumps(data)
    except Exception:
        return body_str


def get_agent(
    settings: Settings = Depends(get_settings),
) -> TripletexAccountingAgent:
    return TripletexAccountingAgent(settings)


def verify_api_key(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.app_api_key:
        return
    expected = f"Bearer {settings.app_api_key}"
    if authorization != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/solve",
    response_model=SolveResponse,
    dependencies=[Depends(verify_api_key)],
)
async def solve_endpoint(
    payload: SolveRequest,
    settings: Settings = Depends(get_settings),
    agent: TripletexAccountingAgent = Depends(get_agent),
) -> SolveResponse:
    if payload.tripletex_credentials is None:
        if (
            not settings.tripletex_sandbox_api_url
            or not settings.tripletex_sandbox_api_session_token
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Tripletex credentials are required either in the request body "
                    "or via TRIPLETEX_SANDBOX_API_URL and "
                    "TRIPLETEX_SANDBOX_API_SESSION_TOKEN in .env"
                ),
            )
        payload = payload.model_copy(
            update={
                "tripletex_credentials": TripletexCredentials(
                    base_url=settings.tripletex_sandbox_api_url,
                    session_token=settings.tripletex_sandbox_api_session_token,
                )
            }
        )
    try:
        await agent.solve(payload)
    except OpenRouterError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return SolveResponse(status="completed")


# Fallback: accept POST to root as well (some platforms POST to base URL)
@app.post(
    "/",
    response_model=SolveResponse,
    dependencies=[Depends(verify_api_key)],
)
async def root_solve_endpoint(
    payload: SolveRequest,
    settings: Settings = Depends(get_settings),
    agent: TripletexAccountingAgent = Depends(get_agent),
) -> SolveResponse:
    return await solve_endpoint(payload, settings, agent)
