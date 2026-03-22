import asyncio
import base64
import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from pypdf import PdfReader

from tripletex_agent.schemas import AttachmentSummary, SolveFile

logger = logging.getLogger(__name__)

try:
    import pymupdf4llm  # pyright: ignore[reportMissingImports]

    _HAS_PYMUPDF4LLM = True
except ImportError:
    _HAS_PYMUPDF4LLM = False

_DATALAB_API_URL = "https://www.datalab.to/api/v1/marker"
_CACHE_DIR = Path(__file__).resolve().parents[2] / "pdf_cache"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _get_cached_extraction(sha: str) -> dict[str, Any] | None:
    cache_file = _CACHE_DIR / f"{sha}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_cache(sha: str, data: dict[str, Any]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"{sha}.json"
    cache_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _save_pdf_file(sha: str, payload: bytes, filename: str) -> Path:
    pdf_dir = _CACHE_DIR / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    safe_name = filename.replace("/", "_").replace("\\", "_")
    dest = pdf_dir / f"{sha}_{safe_name}"
    if not dest.exists():
        dest.write_bytes(payload)
    return dest


@dataclass(slots=True)
class ExtractionResult:
    text: str | None
    method: str
    sha256: str
    cached: bool
    filename: str
    pdf_path: str


async def extract_pdf_with_cache(
    payload: bytes,
    filename: str,
    api_key: str,
    max_text_chars: int,
) -> ExtractionResult:
    sha = _sha256(payload)

    pdf_path = str(_save_pdf_file(sha, payload, filename))

    cached = _get_cached_extraction(sha)
    if cached:
        text = cached.get("text", "")
        if text:
            text = text[:max_text_chars]
        return ExtractionResult(
            text=text or None,
            method=cached.get("method", "cached"),
            sha256=sha,
            cached=True,
            filename=filename,
            pdf_path=pdf_path,
        )

    text: str | None = None
    method = "none"

    if api_key:
        text = await _extract_via_datalab(payload, filename, api_key, max_text_chars)
        if text:
            method = "datalab"

    if not text:
        text = extract_pdf_text(payload, max_text_chars)
        if text:
            method = "pymupdf4llm" if _HAS_PYMUPDF4LLM else "pypdf"

    _save_cache(
        sha,
        {
            "text": text,
            "method": method,
            "filename": filename,
            "sha256": sha,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "char_count": len(text) if text else 0,
        },
    )

    return ExtractionResult(
        text=text,
        method=method,
        sha256=sha,
        cached=False,
        filename=filename,
        pdf_path=pdf_path,
    )


async def _extract_via_datalab(
    payload: bytes, filename: str, api_key: str, max_text_chars: int
) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                _DATALAB_API_URL,
                files={"file": (filename, payload, "application/pdf")},
                data={
                    "output_format": "markdown",
                    "mode": "fast",
                    "langs": "no,en,de,es,fr,pt",
                },
                headers={"X-API-Key": api_key},
            )
            resp.raise_for_status()
            data = resp.json()

            check_url = data.get("request_check_url")
            if not check_url:
                return None

            for _ in range(30):
                await asyncio.sleep(1)
                poll = await client.get(check_url, headers={"X-API-Key": api_key})
                poll.raise_for_status()
                result = poll.json()

                if result.get("status") == "complete":
                    md = result.get("markdown", "")
                    if md and md.strip():
                        return md.strip()[:max_text_chars]
                    return None
                if result.get("status") == "failed":
                    logger.warning("Datalab extraction failed: %s", result.get("error"))
                    return None
    except Exception as exc:
        logger.warning("Datalab PDF extraction failed: %s", exc)
    return None


@dataclass(slots=True)
class PreparedAttachments:
    summaries: list[AttachmentSummary]
    executor_content_parts: list[dict[str, Any]]
    extraction_details: list[dict[str, Any]]


async def prepare_attachments_async(
    files: list[SolveFile],
    max_text_chars: int,
    datalab_api_key: str = "",
) -> PreparedAttachments:
    summaries: list[AttachmentSummary] = []
    executor_content_parts: list[dict[str, Any]] = []
    extraction_details: list[dict[str, Any]] = []

    for file in files:
        payload = base64.b64decode(file.content_base64)
        kind = detect_kind(file.mime_type)
        extracted_text: str | None = None
        notes: str | None = None

        if kind == "pdf":
            result = await extract_pdf_with_cache(
                payload, file.filename, datalab_api_key, max_text_chars
            )
            extracted_text = result.text
            notes = f"Extracted via {result.method}" + (
                " (cached)" if result.cached else ""
            )
            extraction_details.append(
                {
                    "filename": file.filename,
                    "sha256": result.sha256,
                    "method": result.method,
                    "cached": result.cached,
                    "char_count": len(extracted_text) if extracted_text else 0,
                    "pdf_path": result.pdf_path,
                }
            )
            if not extracted_text:
                notes = "No extractable PDF text found"
        elif kind == "text":
            extracted_text = decode_text(payload, max_text_chars)
        elif kind == "image":
            data_url = build_data_url(file.mime_type, file.content_base64)
            executor_content_parts.append(
                {"type": "image_url", "image_url": {"url": data_url}}
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
        extraction_details=extraction_details,
    )


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
                {"type": "image_url", "image_url": {"url": data_url}}
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
        extraction_details=[],
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
