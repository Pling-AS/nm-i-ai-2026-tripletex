import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, status

from tripletex_agent.agent import TripletexAccountingAgent
from tripletex_agent.config import Settings, get_settings
from tripletex_agent.openrouter import OpenRouterError
from tripletex_agent.schemas import SolveRequest, SolveResponse, TripletexCredentials


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO)
    )
    yield


app = FastAPI(title="Tripletex Accounting Agent", lifespan=lifespan)


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
