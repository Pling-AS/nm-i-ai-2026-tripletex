import base64

from tripletex_agent.files import prepare_attachments
from tripletex_agent.schemas import SolveFile


def test_prepare_text_attachment_extracts_text() -> None:
    payload = base64.b64encode("hei fra norge".encode("utf-8")).decode("utf-8")
    solve_file = SolveFile(
        filename="input.txt",
        content_base64=payload,
        mime_type="text/plain",
    )

    prepared = prepare_attachments([solve_file], max_text_chars=100)

    assert prepared.summaries[0].kind == "text"
    assert prepared.summaries[0].extracted_text == "hei fra norge"
    assert prepared.executor_content_parts
