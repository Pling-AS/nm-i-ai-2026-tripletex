import httpx
import pytest

from tripletex_agent.tripletex import TripletexClient


@pytest.mark.asyncio
async def test_request_clears_get_cache_after_successful_mutation() -> None:
    responses = [
        httpx.Response(
            200,
            json={"values": [{"id": 1, "version": 1, "userType": "STANDARD"}]},
        ),
        httpx.Response(
            200,
            json={"value": {"id": 1, "version": 2, "userType": "EXTENDED"}},
        ),
        httpx.Response(
            200,
            json={"values": [{"id": 1, "version": 2, "userType": "EXTENDED"}]},
        ),
    ]
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, str(request.url)))
        return responses.pop(0)

    client = TripletexClient(
        base_url="https://example.test/v2",
        session_token="token",
        timeout=5.0,
    )
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url="https://example.test/v2",
        timeout=5.0,
        auth=httpx.BasicAuth("0", "token"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        transport=httpx.MockTransport(handler),
    )

    try:
        first_lookup = await client.request(
            method="GET",
            path="/employee",
            params={"id": "1", "fields": "id,version,userType"},
        )
        await client.request(
            method="PUT",
            path="/employee/1",
            json_body={"id": 1, "version": 1, "userType": "EXTENDED"},
        )
        second_lookup = await client.request(
            method="GET",
            path="/employee",
            params={"id": "1", "fields": "id,version,userType"},
        )
    finally:
        await client.close()

    assert first_lookup["values"][0]["version"] == 1
    assert second_lookup["values"][0]["version"] == 2
    assert len(requests) == 3


def test_parse_response_body_accepts_successful_empty_json_response() -> None:
    client = TripletexClient(
        base_url="https://example.test/v2",
        session_token="token",
        timeout=5.0,
    )
    response = httpx.Response(
        200,
        content=b"",
        headers={"content-type": "application/json"},
    )

    try:
        assert client._parse_response_body(response) is None
    finally:
        import asyncio

        asyncio.run(client.close())
