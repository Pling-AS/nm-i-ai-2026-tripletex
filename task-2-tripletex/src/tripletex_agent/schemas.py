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


class EntitySlot(BaseModel):
    """A named entity extracted from the prompt with its field values."""

    role: str  # e.g. "customer", "supplier", "employee", "project_manager"
    name: str | None = None
    organization_number: str | None = None
    email: str | None = None
    phone: str | None = None
    date_of_birth: str | None = None  # YYYY-MM-DD
    extra_fields: dict[str, Any] = Field(default_factory=dict)


class LineItemSlot(BaseModel):
    """A product/service line item extracted from the prompt."""

    description: str
    product_number: str | None = None  # explicit product number from prompt
    quantity: float = 1.0
    unit_price_excluding_vat: float | None = None
    vat_rate_percent: float | None = None  # 25, 15, 0, etc.


class PlannerOutput(BaseModel):
    task_type: str
    goal: str
    likely_resources: list[str] = Field(default_factory=list)
    success_checks: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    suggested_first_action: str

    # --- Structured slots (enriched planning) ---
    entities: list[EntitySlot] = Field(default_factory=list)
    line_items: list[LineItemSlot] = Field(default_factory=list)
    actions: list[str] = Field(
        default_factory=list,
        description="Ordered action flags: create_customer, create_product, "
        "create_order, create_invoice, send_invoice, register_payment, "
        "reverse_payment, create_voucher, etc.",
    )
    extracted_dates: dict[str, str] = Field(
        default_factory=dict,
        description="Dates explicitly stated in the prompt. Keys: invoice_date, "
        "due_date, start_date, end_date, payment_date, birth_date, etc. "
        "Values: YYYY-MM-DD format.",
    )
    ordered_steps: list[str] = Field(
        default_factory=list,
        description="Ordered list of concrete API steps the executor should follow.",
    )
    alternative_task_type: str | None = Field(
        default=None,
        description="Second-best task_type classification if the primary is uncertain.",
    )


class ToolExecutionResult(BaseModel):
    ok: bool
    name: str
    payload: dict[str, Any]
