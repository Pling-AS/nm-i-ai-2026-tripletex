import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EndpointEntry:
    method: str
    path: str
    tags: list[str]
    summary: str
    description: str
    operation_id: str
    request_schema: str | None
    response_schema: str | None


class TripletexSpecIndex:
    def __init__(self, spec_path: Path) -> None:
        self._spec_path = spec_path
        with spec_path.open("r", encoding="utf-8") as file:
            spec = json.load(file)
        self._paths = spec.get("paths", {})
        self._schemas = spec.get("components", {}).get("schemas", {})
        self._endpoints = self._build_endpoints()
        self._endpoint_map = {
            (entry.method, entry.path): entry for entry in self._endpoints
        }

    def search_endpoints(
        self,
        query: str,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        tokens = self._tokenize(query)
        scored: list[tuple[int, EndpointEntry]] = []
        for entry in self._endpoints:
            path_lower = entry.path.lower()
            summary_lower = entry.summary.lower()
            description_lower = entry.description.lower()
            operation_id_lower = entry.operation_id.lower()
            tags_lower = [tag.lower() for tag in entry.tags]
            request_schema_lower = (entry.request_schema or "").lower()
            response_schema_lower = (entry.response_schema or "").lower()
            haystack = " ".join(
                [
                    entry.path,
                    entry.summary,
                    entry.description,
                    entry.operation_id,
                    " ".join(entry.tags),
                    entry.request_schema or "",
                    entry.response_schema or "",
                ]
            ).lower()
            score = 0
            for token in tokens:
                exact_path_token = f"/{token}"
                if path_lower == exact_path_token:
                    score += 12
                elif exact_path_token in path_lower:
                    score += 10
                elif token in path_lower:
                    score += 5
                if token in summary_lower:
                    score += 4
                if any(token == tag or token in tag for tag in tags_lower):
                    score += 4
                if token in request_schema_lower or token in response_schema_lower:
                    score += 3
                if token in operation_id_lower:
                    score += 2
                if token in description_lower:
                    score += 1
                if token in haystack:
                    score += 1
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda item: (-item[0], item[1].path, item[1].method))
        return [self._serialize_endpoint(entry) for _, entry in scored[:limit]]

    def get_schema(
        self,
        schema_name: str,
        max_properties: int = 40,
    ) -> dict[str, Any]:
        schema = self._schemas.get(schema_name)
        if schema is None:
            raise KeyError(f"Unknown schema: {schema_name}")
        return self._summarize_schema(
            schema_name,
            schema,
            max_properties=max_properties,
        )

    def get_schema_definition(self, schema_name: str) -> dict[str, Any]:
        schema = self._schemas.get(schema_name)
        if schema is None:
            raise KeyError(f"Unknown schema: {schema_name}")
        return schema

    def get_endpoint(self, method: str, path: str) -> dict[str, Any]:
        normalized_method = method.upper()
        normalized_path = path if path.startswith("/") else f"/{path}"
        entry = self._endpoint_map.get((normalized_method, normalized_path))
        if entry is None:
            entry = self._find_templated_endpoint(
                normalized_method,
                normalized_path,
            )
        if entry is None:
            raise KeyError(f"Unknown endpoint: {normalized_method} {normalized_path}")
        return self._serialize_endpoint(entry)

    def _find_templated_endpoint(
        self,
        method: str,
        path: str,
    ) -> EndpointEntry | None:
        for entry in self._endpoints:
            if entry.method != method:
                continue
            pattern = re.sub(r"\{[^/]+\}", r"[^/]+", entry.path)
            if re.fullmatch(pattern, path):
                return entry
        return None

    def _build_endpoints(self) -> list[EndpointEntry]:
        entries: list[EndpointEntry] = []
        for path, methods in self._paths.items():
            if not isinstance(methods, dict):
                continue
            for method, operation in methods.items():
                if method.lower() not in {"get", "post", "put", "delete"}:
                    continue
                if not isinstance(operation, dict):
                    continue
                request_schema = self._extract_request_schema(operation)
                response_schema = self._extract_response_schema(operation)
                entries.append(
                    EndpointEntry(
                        method=method.upper(),
                        path=path,
                        tags=list(operation.get("tags") or []),
                        summary=str(operation.get("summary") or ""),
                        description=str(operation.get("description") or ""),
                        operation_id=str(operation.get("operationId") or ""),
                        request_schema=request_schema,
                        response_schema=response_schema,
                    )
                )
        return entries

    def _extract_request_schema(self, operation: dict[str, Any]) -> str | None:
        request_body = operation.get("requestBody") or {}
        content = request_body.get("content") or {}
        for media_type in content.values():
            schema = media_type.get("schema") or {}
            ref = schema.get("$ref")
            if ref:
                return self._schema_name_from_ref(ref)
        return None

    def _extract_response_schema(self, operation: dict[str, Any]) -> str | None:
        responses = operation.get("responses") or {}
        for response in responses.values():
            content = response.get("content") or {}
            for media_type in content.values():
                schema = media_type.get("schema") or {}
                ref = schema.get("$ref")
                if ref:
                    return self._schema_name_from_ref(ref)
        return None

    def _summarize_schema(
        self,
        schema_name: str,
        schema: dict[str, Any],
        max_properties: int,
    ) -> dict[str, Any]:
        properties = schema.get("properties") or {}
        summary_properties: dict[str, Any] = {}
        for index, (name, definition) in enumerate(properties.items()):
            if index >= max_properties:
                break
            summary_properties[name] = self._summarize_property(definition)
        return {
            "schema": schema_name,
            "type": schema.get("type"),
            "description": schema.get("description"),
            "required": schema.get("required", []),
            "properties": summary_properties,
        }

    def _summarize_property(self, definition: dict[str, Any]) -> dict[str, Any]:
        ref = definition.get("$ref")
        summary: dict[str, Any] = {}
        if ref:
            summary["ref"] = self._schema_name_from_ref(ref)
            return summary
        if "type" in definition:
            summary["type"] = definition["type"]
        if "format" in definition:
            summary["format"] = definition["format"]
        if "enum" in definition:
            summary["enum"] = definition["enum"]
        if definition.get("readOnly"):
            summary["readOnly"] = True
        items = definition.get("items")
        if isinstance(items, dict):
            items_ref = items.get("$ref")
            if items_ref:
                summary["items"] = {"ref": self._schema_name_from_ref(items_ref)}
            elif items.get("type"):
                summary["items"] = {"type": items.get("type")}
        return summary

    def _serialize_endpoint(self, entry: EndpointEntry) -> dict[str, Any]:
        return {
            "method": entry.method,
            "path": entry.path,
            "tags": entry.tags,
            "summary": entry.summary,
            "description": entry.description,
            "operationId": entry.operation_id,
            "requestSchema": entry.request_schema,
            "responseSchema": entry.response_schema,
        }

    def _schema_name_from_ref(self, ref: str) -> str:
        return ref.rsplit("/", maxsplit=1)[-1]

    def _tokenize(self, value: str) -> list[str]:
        raw_tokens = [
            token for token in re.split(r"[^a-zA-Z0-9_/:-]+", value.lower()) if token
        ]
        expanded_tokens: list[str] = []
        seen: set[str] = set()
        for token in raw_tokens:
            for candidate in (token, self._singularize_token(token)):
                if not candidate or candidate in seen:
                    continue
                expanded_tokens.append(candidate)
                seen.add(candidate)
        return expanded_tokens

    def _singularize_token(self, token: str) -> str:
        if len(token) > 4 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
        return token
