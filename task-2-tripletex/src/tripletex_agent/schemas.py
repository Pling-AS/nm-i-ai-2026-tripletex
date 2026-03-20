from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class SolveFile(BaseModel):
    filename: str
    content_base64: str
    mime_type: str


class TripletexCredentials(BaseModel):
    base_url: HttpUrl | str
    session_token: str = Field(min_length=1)


class SolveRequest(BaseModel):
    prompt: str = Field(min_length=1)
    files: list[SolveFile] = Field(default_factory=list)
    tripletex_credentials: TripletexCredentials | None = None


class SolveResponse(BaseModel):
    status: Literal["completed"]


class AttachmentSummary(BaseModel):
    filename: str
    mime_type: str
    kind: Literal["image", "pdf", "text", "binary"]
    extracted_text: str | None = None
    notes: str | None = None


class PlannerOutput(BaseModel):
    task_type: str
    goal: str
    likely_resources: list[str] = Field(default_factory=list)
    success_checks: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    suggested_first_action: str


class ToolExecutionResult(BaseModel):
    ok: bool
    name: str
    payload: dict[str, Any]
