import logging
import re
from typing import Any


logger = logging.getLogger(__name__)


_FIELD_OVERRIDES: dict[str, dict[str, str]] = {
    "/project": {"fixedprice": "fixedPrice"},
}


def validate_and_fix_payload(
    method: str,
    path: str,
    payload: dict,
    spec_index: Any,
) -> tuple[dict, list[str]]:
    if not isinstance(payload, dict):
        return payload, []

    normalized_method = (method or "").upper()
    normalized_path = path if str(path).startswith("/") else f"/{path}"
    corrections: list[str] = []

    overrides = _FIELD_OVERRIDES.get(normalized_path, {})
    if overrides:
        for wrong, right in overrides.items():
            if wrong in payload and right not in payload:
                payload[right] = payload.pop(wrong)
                corrections.append(
                    f"Renamed '{wrong}' to '{right}' (hardcoded override)."
                )

    try:
        cache = _get_runtime_cache(spec_index)
        request_schema_node = _lookup_request_schema_node(
            spec_index,
            normalized_method,
            normalized_path,
            cache,
        )
        if request_schema_node is None:
            return payload, []

        context = _SchemaContext(spec_index=spec_index, cache=cache)
        fixed_payload = _transform_value(
            value=payload,
            schema_node=request_schema_node,
            path_prefix="",
            context=context,
            corrections=corrections,
        )
        if not isinstance(fixed_payload, dict):
            return payload, []

        for message in corrections:
            logger.info(
                "Payload correction for %s %s: %s",
                normalized_method,
                normalized_path,
                message,
            )
        return fixed_payload, corrections
    except Exception as exc:
        logger.debug(
            "Schema validation fallback for %s %s: %s",
            normalized_method,
            normalized_path,
            exc,
        )
        return payload, []


class _SchemaContext:
    def __init__(self, spec_index: Any, cache: dict[str, Any]) -> None:
        self.spec_index = spec_index
        self.cache = cache
        self._resolving_refs: set[str] = set()


def _get_runtime_cache(spec_index: Any) -> dict[str, Any]:
    cache = getattr(spec_index, "_schema_validator_cache", None)
    if isinstance(cache, dict):
        cache.setdefault("endpoint_schema", {})
        cache.setdefault("schema_definition", {})
        cache.setdefault("resolved_ref", {})
        cache.setdefault("object_properties", {})
        return cache

    cache = {
        "endpoint_schema": {},
        "schema_definition": {},
        "resolved_ref": {},
        "object_properties": {},
    }
    try:
        setattr(spec_index, "_schema_validator_cache", cache)
    except Exception:
        pass
    return cache


def _lookup_request_schema_node(
    spec_index: Any,
    method: str,
    path: str,
    cache: dict[str, Any],
) -> dict[str, Any] | None:
    cache_key = (method, path)
    cached = cache["endpoint_schema"].get(cache_key)
    if isinstance(cached, dict):
        return cached
    if cache_key in cache["endpoint_schema"]:
        return None

    endpoint_path: str | None = None
    endpoint = None
    try:
        endpoint = spec_index.get_endpoint(method, path)
    except Exception:
        endpoint = None

    if isinstance(endpoint, dict):
        endpoint_path = (
            endpoint.get("path") if isinstance(endpoint.get("path"), str) else None
        )
        request_schema_name = endpoint.get("requestSchema")
        if isinstance(request_schema_name, str) and request_schema_name:
            schema_node = {"$ref": f"#/components/schemas/{request_schema_name}"}
            cache["endpoint_schema"][cache_key] = schema_node
            return schema_node

    operation = _find_operation(spec_index, method, endpoint_path or path)
    schema_node = _extract_operation_request_schema(operation)
    cache["endpoint_schema"][cache_key] = schema_node
    return schema_node


def _find_operation(spec_index: Any, method: str, path: str) -> dict[str, Any] | None:
    paths = getattr(spec_index, "_paths", None)
    if not isinstance(paths, dict):
        return None

    normalized_path = path if path.startswith("/") else f"/{path}"
    method_key = method.lower()

    methods = paths.get(normalized_path)
    if isinstance(methods, dict):
        operation = methods.get(method_key)
        if isinstance(operation, dict):
            return operation

    normalized_lower = normalized_path.lower()
    for candidate_path, candidate_methods in paths.items():
        if not isinstance(candidate_path, str) or not isinstance(
            candidate_methods, dict
        ):
            continue
        if candidate_path.lower() != normalized_lower:
            continue
        operation = candidate_methods.get(method_key)
        if isinstance(operation, dict):
            return operation

    for candidate_path, candidate_methods in paths.items():
        if not isinstance(candidate_path, str) or not isinstance(
            candidate_methods, dict
        ):
            continue
        operation = candidate_methods.get(method_key)
        if not isinstance(operation, dict):
            continue
        pattern = re.sub(r"\{[^/]+\}", r"[^/]+", candidate_path)
        if re.fullmatch(pattern, normalized_path):
            return operation

    return None


def _extract_operation_request_schema(
    operation: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(operation, dict):
        return None

    request_body = operation.get("requestBody")
    if not isinstance(request_body, dict):
        return None
    content = request_body.get("content")
    if not isinstance(content, dict):
        return None

    for media_type in content.values():
        if not isinstance(media_type, dict):
            continue
        schema = media_type.get("schema")
        if isinstance(schema, dict):
            return schema
    return None


def _transform_value(
    value: Any,
    schema_node: dict[str, Any] | None,
    path_prefix: str,
    context: _SchemaContext,
    corrections: list[str],
) -> Any:
    if schema_node is None:
        return value

    resolved_schema = _resolve_schema(schema_node, context)
    if not isinstance(resolved_schema, dict):
        return value

    if isinstance(value, dict):
        return _transform_object(
            value=value,
            schema=resolved_schema,
            path_prefix=path_prefix,
            context=context,
            corrections=corrections,
        )

    if isinstance(value, list):
        items_schema = resolved_schema.get("items")
        if not isinstance(items_schema, dict):
            return value
        return [
            _transform_value(
                item,
                items_schema,
                f"{path_prefix}[{index}]" if path_prefix else f"[{index}]",
                context,
                corrections,
            )
            for index, item in enumerate(value)
        ]

    return value


def _transform_object(
    value: dict[str, Any],
    schema: dict[str, Any],
    path_prefix: str,
    context: _SchemaContext,
    corrections: list[str],
) -> dict[str, Any]:
    properties = _collect_properties(schema, context)
    if not properties:
        return {k: v for k, v in value.items()}

    folded_map: dict[str, str | None] = {}
    for property_name in properties:
        key = property_name.lower()
        if key not in folded_map:
            folded_map[key] = property_name
        elif folded_map[key] != property_name:
            folded_map[key] = None

    transformed: dict[str, Any] = {}
    for original_key, original_value in value.items():
        field_path = _join_field_path(path_prefix, original_key)
        canonical_key = original_key
        matched_schema = properties.get(original_key)

        if matched_schema is None:
            case_match = folded_map.get(original_key.lower())
            if case_match and case_match != original_key:
                canonical_key = case_match
                matched_schema = properties.get(case_match)
                corrections.append(
                    f"Renamed field '{field_path}' to '{_join_field_path(path_prefix, canonical_key)}'."
                )

        if isinstance(matched_schema, dict) and _is_read_only(matched_schema, context):
            continue

        output_key = canonical_key
        if output_key in transformed and output_key != original_key:
            corrections.append(
                f"Skipped renaming field '{field_path}' to '{_join_field_path(path_prefix, output_key)}' due to key conflict."
            )
            output_key = original_key

        transformed[output_key] = _transform_value(
            value=original_value,
            schema_node=matched_schema if isinstance(matched_schema, dict) else None,
            path_prefix=_join_field_path(path_prefix, output_key),
            context=context,
            corrections=corrections,
        )

    return transformed


def _collect_properties(
    schema: dict[str, Any],
    context: _SchemaContext,
) -> dict[str, dict[str, Any]]:
    schema_key = _schema_cache_key(schema)
    cached = context.cache["object_properties"].get(schema_key)
    if isinstance(cached, dict):
        return cached

    merged: dict[str, dict[str, Any]] = {}
    for composite_key in ("allOf", "anyOf", "oneOf"):
        variants = schema.get(composite_key)
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            variant_schema = _resolve_schema(variant, context)
            if not isinstance(variant_schema, dict):
                continue
            merged.update(_collect_properties(variant_schema, context))

    direct_props = schema.get("properties")
    if isinstance(direct_props, dict):
        for prop_name, prop_schema in direct_props.items():
            if isinstance(prop_name, str) and isinstance(prop_schema, dict):
                merged[prop_name] = prop_schema

    context.cache["object_properties"][schema_key] = merged
    return merged


def _resolve_schema(
    schema_node: dict[str, Any],
    context: _SchemaContext,
) -> dict[str, Any] | None:
    ref = schema_node.get("$ref") if isinstance(schema_node, dict) else None
    if not isinstance(ref, str):
        return schema_node if isinstance(schema_node, dict) else None

    cached = context.cache["resolved_ref"].get(ref)
    if isinstance(cached, dict):
        return cached
    if ref in context._resolving_refs:
        return None

    context._resolving_refs.add(ref)
    try:
        schema_name = ref.rsplit("/", maxsplit=1)[-1]
        definition = context.cache["schema_definition"].get(schema_name)
        if definition is None:
            try:
                definition = context.spec_index.get_schema_definition(schema_name)
            except Exception:
                definition = None
            context.cache["schema_definition"][schema_name] = definition

        if isinstance(definition, dict):
            context.cache["resolved_ref"][ref] = definition
            return definition
        return None
    finally:
        context._resolving_refs.discard(ref)


def _is_read_only(schema_node: dict[str, Any], context: _SchemaContext) -> bool:
    if schema_node.get("readOnly") is True:
        return True
    resolved = _resolve_schema(schema_node, context)
    return isinstance(resolved, dict) and resolved.get("readOnly") is True


def _join_field_path(prefix: str, key: str) -> str:
    if not prefix:
        return key
    return f"{prefix}.{key}"


def _schema_cache_key(schema: dict[str, Any]) -> str:
    ref = schema.get("$ref")
    if isinstance(ref, str):
        return f"ref:{ref}"
    return f"obj:{id(schema)}"
