import json
from typing import Any

import httpx  # pyright: ignore[reportMissingImports]
from tenacity import (  # pyright: ignore[reportMissingImports]
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class TripletexError(Exception):
    pass


class TripletexRetriableError(TripletexError):
    pass


class TripletexApiError(TripletexError):
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Tripletex request failed with status {status_code}: {body}")


class TripletexClient:
    def __init__(
        self,
        *,
        base_url: str,
        session_token: str,
        timeout: float,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            auth=httpx.BasicAuth("0", session_token),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        self.call_count = 0
        self.error_count = 0
        self.call_log: list[dict[str, Any]] = []
        self._get_cache: dict[str, dict[str, Any] | list[Any] | str | None] = {}

    async def close(self) -> None:
        await self._client.aclose()

    async def request(
        self,
        *,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | list[Any] | None = None,
    ) -> dict[str, Any] | list[Any] | str | None:
        normalized_method = method.upper()
        normalized_path = path if path.startswith("/") else f"/{path}"
        import re as _re

        # Strip /v2 prefix if present — base_url already includes /v2
        normalized_path = _re.sub(r"^/v2/", "/", normalized_path)
        normalized_path = _re.sub(r"^/v2$", "/", normalized_path)
        # Fix URL-encoded > in whoAmI path
        normalized_path = normalized_path.replace("%3E", ">").replace("%3e", ">")
        normalized_params = _normalize_mapping(params)
        cache_key = None
        if normalized_method == "GET":
            cache_key = _build_get_cache_key(normalized_path, normalized_params)
            cached_response = self._get_cache.get(cache_key)
            if cache_key in self._get_cache:
                self.call_log.append(
                    {
                        "method": normalized_method,
                        "path": normalized_path,
                        "status_code": 200,
                        "cache_hit": True,
                    }
                )
                return cached_response

        is_mutation = normalized_method in {"POST", "PUT", "DELETE"}
        response = None
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(
                (httpx.TransportError, TripletexRetriableError)
            ),
            reraise=True,
        ):
            with attempt:
                response = await self._client.request(
                    normalized_method,
                    normalized_path,
                    params=normalized_params,
                    json=json_body,
                )
                self.call_count += 1
                self.call_log.append(
                    {
                        "method": normalized_method,
                        "path": normalized_path,
                        "status_code": response.status_code,
                    }
                )
                if response.status_code == 429:
                    raise TripletexRetriableError(response.text)
                # Only retry 5xx for safe reads — mutations should fail fast
                # to avoid burning 3 API calls on a single bad write.
                if response.status_code >= 500 and not is_mutation:
                    raise TripletexRetriableError(response.text)

        if response is None:
            raise TripletexError("Tripletex request produced no response")

        body = self._parse_response_body(response)
        # Fast-fail on expired proxy tokens — retrying is pointless
        if response.status_code >= 400:
            body_str = str(body) if body else ""
            if "invalid or expired proxy token" in body_str.lower() or (
                "expired" in body_str.lower() and "proxy" in body_str.lower()
            ):
                self.error_count += 1
                raise TripletexApiError(response.status_code, body)
        if response.status_code >= 400:
            self.error_count += 1
            raise TripletexApiError(response.status_code, body)

        compacted = compact_response(body)
        if cache_key is not None:
            self._get_cache[cache_key] = compacted
        elif normalized_method in {"POST", "PUT", "DELETE"}:
            self._get_cache.clear()
        return compacted

    async def grant_entitlements(
        self,
        *,
        employee_id: int,
        template: str,
        ensure_user_type_extended: bool = False,
    ) -> dict[str, Any]:
        changes: list[Any] = []
        if ensure_user_type_extended:
            employee_lookup = await self.request(
                method="GET",
                path="/employee",
                params={"id": str(employee_id), "fields": "id,version,userType"},
            )
            values = (
                (employee_lookup or {}).get("values", [])
                if isinstance(employee_lookup, dict)
                else []
            )
            employee = values[0] if values else None
            if employee and employee.get("userType") != "EXTENDED":
                changes.append(
                    await self.request(
                        method="PUT",
                        path=f"/employee/{employee_id}",
                        json_body={
                            "id": employee_id,
                            "version": employee.get("version"),
                            "userType": "EXTENDED",
                        },
                    )
                )

        entitlement_result = await self.request(
            method="PUT",
            path="/employee/entitlement/:grantEntitlementsByTemplate",
            params={"employeeId": employee_id, "template": template},
        )
        changes.append(entitlement_result)
        return {"updated": True, "actions": changes}

    def _parse_response_body(
        self,
        response: httpx.Response,
    ) -> dict[str, Any] | list[Any] | str | None:
        if response.status_code == 204:
            return None
        if not response.content or not response.content.strip():
            return None
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                return response.json()
            except json.JSONDecodeError:
                return response.text or None
        return response.text


def compact_response(
    value: Any,
    *,
    max_depth: int = 6,
    max_items: int = 15,
    max_string: int = 800,
) -> Any:
    if max_depth <= 0:
        return "<trimmed>"
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                compacted["__trimmed__"] = True
                break
            if key == "values" and isinstance(item, list):
                compacted[key] = [
                    compact_response(
                        entry,
                        max_depth=max_depth - 1,
                        max_items=max_items,
                        max_string=max_string,
                    )
                    for entry in item[:max_items]
                ]
                if len(item) > max_items:
                    compacted["values_trimmed"] = len(item) - max_items
            else:
                compacted[key] = compact_response(
                    item,
                    max_depth=max_depth - 1,
                    max_items=max_items,
                    max_string=max_string,
                )
        return compacted
    if isinstance(value, list):
        compacted_items = [
            compact_response(
                item,
                max_depth=max_depth - 1,
                max_items=max_items,
                max_string=max_string,
            )
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            compacted_items.append("<trimmed>")
        return compacted_items
    if isinstance(value, str):
        return value[:max_string]
    return value


def _normalize_mapping(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {key: item for key, item in value.items() if item is not None}


def _build_get_cache_key(
    path: str,
    params: dict[str, Any] | None,
) -> str:
    return json.dumps(
        {
            "path": path,
            "params": params or {},
        },
        sort_keys=True,
        ensure_ascii=False,
    )
