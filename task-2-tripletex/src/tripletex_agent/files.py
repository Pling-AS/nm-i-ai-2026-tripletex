import base64
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from pypdf import PdfReader

from tripletex_agent.schemas import AttachmentSummary, SolveFile

logger = logging.getLogger(__name__)

try:
    import pymupdf4llm  # pyright: ignore[reportMissingImports]

    _HAS_PYMUPDF4LLM = True
except ImportError:
    _HAS_PYMUPDF4LLM = False


@dataclass(slots=True)
class PreparedAttachments:
    summaries: list[AttachmentSummary]
    executor_content_parts: list[dict[str, Any]]


def prepare_attachments(
    files: list[SolveFile],
    max_text_chars: int,
) -> PreparedAttachments:
    summaries: list[AttachmentSummary] = []
    executor_content_parts: list[dict[str, Any]] = []

    for file in files:
        payload = base64.b64decode(file.content_base64)
        kind = detect_kind(file.mime_type)
        extracted_text: str | None = None
        notes: str | None = None

        if kind == "pdf":
            extracted_text = extract_pdf_text(payload, max_text_chars)
            if not extracted_text:
                notes = "No extractable PDF text found"
        elif kind == "text":
            extracted_text = decode_text(payload, max_text_chars)
        elif kind == "image":
            data_url = build_data_url(file.mime_type, file.content_base64)
            executor_content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            )
            notes = f"Image attachment included: {file.filename}"
        else:
            notes = f"Binary attachment included: {file.filename}"

        summary = AttachmentSummary(
            filename=file.filename,
            mime_type=file.mime_type,
            kind=kind,
            extracted_text=extracted_text,
            notes=notes,
        )
        summaries.append(summary)

        text_block = render_attachment_text(summary)
        if text_block:
            executor_content_parts.append({"type": "text", "text": text_block})

    return PreparedAttachments(
        summaries=summaries,
        executor_content_parts=executor_content_parts,
    )


def detect_kind(mime_type: str) -> str:
    if mime_type.startswith("image/"):
        return "image"
    if mime_type == "application/pdf":
        return "pdf"
    if mime_type.startswith("text/") or mime_type in {
        "application/json",
        "text/csv",
        "application/xml",
    }:
        return "text"
    return "binary"


def extract_pdf_text(payload: bytes, max_text_chars: int) -> str | None:
    if _HAS_PYMUPDF4LLM:
        try:
            import pymupdf  # pyright: ignore[reportMissingImports]

            doc = pymupdf.open(stream=payload, filetype="pdf")
            md = pymupdf4llm.to_markdown(doc)
            doc.close()
            if md and md.strip():
                return md.strip()[:max_text_chars]
        except Exception as exc:
            logger.warning("pymupdf4llm failed, falling back to pypdf: %s", exc)

    try:
        reader = PdfReader(BytesIO(payload))
    except Exception:
        return None

    fragments: list[str] = []
    remaining = max_text_chars
    for page in reader.pages:
        if remaining <= 0:
            break
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if not text.strip():
            continue
        fragment = text[:remaining]
        fragments.append(fragment)
        remaining -= len(fragment)
    combined = "\n\n".join(fragments).strip()
    return combined or None


def decode_text(payload: bytes, max_text_chars: int) -> str:
    try:
        return payload.decode("utf-8")[:max_text_chars]
    except UnicodeDecodeError:
        return payload.decode("latin-1", errors="ignore")[:max_text_chars]


def build_data_url(mime_type: str, content_base64: str) -> str:
    return f"data:{mime_type};base64,{content_base64}"


def render_attachment_text(summary: AttachmentSummary) -> str:
    parts = [
        f"Attachment: {summary.filename}",
        f"MIME type: {summary.mime_type}",
        f"Kind: {summary.kind}",
    ]
    if summary.extracted_text:
        parts.append("Extracted text:")
        parts.append(summary.extracted_text)
    if summary.notes:
        parts.append(f"Notes: {summary.notes}")
    return "\n".join(parts)
