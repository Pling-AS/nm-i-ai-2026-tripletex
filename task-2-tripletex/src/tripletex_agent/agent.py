from difflib import get_close_matches
import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from tripletex_agent.config import Settings
from tripletex_agent.files import prepare_attachments
from tripletex_agent.openrouter import OpenRouterClient, OpenRouterError
from tripletex_agent.prompts import EXECUTOR_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT
from tripletex_agent.schemas import PlannerOutput, SolveRequest
from tripletex_agent.spec_index import TripletexSpecIndex
from tripletex_agent.trace import RunTrace
from tripletex_agent.tripletex import (
    TripletexApiError,
    TripletexClient,
    compact_response,
)


@dataclass(slots=True)
class AgentRunResult:
    planner: PlannerOutput
    stats: dict[str, Any]


@dataclass(slots=True)
class ExecutionState:
    inspected_schemas: set[str] = field(default_factory=set)
    cached_tool_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    recent_tool_call_keys: list[str] = field(default_factory=list)
    recent_endpoint_families: list[str] = field(default_factory=list)


class TripletexAccountingAgent:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._spec_index = TripletexSpecIndex(settings.tripletex_api_spec_path)

    async def solve(self, request: SolveRequest) -> AgentRunResult:
        execution_state = ExecutionState()
        trace = RunTrace()
        trace.write(
            "init",
            {
                "prompt": request.prompt,
                "file_count": len(request.files),
                "base_url": str(request.tripletex_credentials.base_url),
            },
        )
        attachments = prepare_attachments(
            request.files,
            self._settings.max_attachment_text_chars,
        )
        trace.write(
            "attachments_prepared",
            {
                "attachments": [
                    summary.model_dump() for summary in attachments.summaries
                ]
            },
        )
        openrouter = OpenRouterClient(self._settings)
        tripletex = TripletexClient(
            base_url=str(request.tripletex_credentials.base_url),
            session_token=request.tripletex_credentials.session_token,
            timeout=self._settings.http_timeout_seconds,
        )
        try:
            planner = await self._plan(openrouter, request, attachments.summaries)
            trace.write("planner", planner.model_dump())
            await self._execute(
                openrouter,
                tripletex,
                request,
                planner,
                attachments.executor_content_parts,
                execution_state,
                trace,
            )
            trace.write(
                "done",
                {
                    "tripletex_call_count": tripletex.call_count,
                    "tripletex_error_count": tripletex.error_count,
                    "tripletex_call_log": tripletex.call_log,
                },
            )
            return AgentRunResult(
                planner=planner,
                stats={
                    "tripletex_call_count": tripletex.call_count,
                    "tripletex_error_count": tripletex.error_count,
                },
            )
        except Exception as exc:
            trace.write("error", {"message": str(exc)})
            raise
        finally:
            await openrouter.close()
            await tripletex.close()

    async def _plan(
        self,
        openrouter: OpenRouterClient,
        request: SolveRequest,
        attachment_summaries: list[Any],
    ) -> PlannerOutput:
        plan_prompt = {
            "prompt": request.prompt,
            "attachments": [summary.model_dump() for summary in attachment_summaries],
        }
        result = await openrouter.complete_json(
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(plan_prompt, ensure_ascii=False),
                },
            ]
        )
        try:
            return PlannerOutput.model_validate(result)
        except ValidationError as exc:
            raise OpenRouterError(f"Planner output validation failed: {exc}") from exc

    def _build_execution_brief(
        self,
        planner: PlannerOutput,
        request_prompt: str,
    ) -> dict[str, Any]:
        primary_resource, linked_resources = _infer_task_resources(
            planner=planner,
            request_prompt=request_prompt,
        )
        preferred_link_fields = _infer_relation_roles(
            planner=planner,
            request_prompt=request_prompt,
        )
        requested_field_names = _infer_requested_schema_fields(request_prompt)
        explicit_prompt_values = _extract_explicit_prompt_values(request_prompt)
        queries = _build_execution_brief_queries(planner, request_prompt)
        candidate_endpoints: list[dict[str, Any]] = []
        seen_endpoints: set[tuple[str, str]] = set()

        for query in queries:
            for endpoint in self._spec_index.search_endpoints(query, limit=4):
                endpoint_key = (
                    str(endpoint.get("method") or ""),
                    str(endpoint.get("path") or ""),
                )
                if endpoint_key in seen_endpoints:
                    continue

                candidate = {
                    "query": query,
                    "method": endpoint.get("method"),
                    "path": endpoint.get("path"),
                    "summary": endpoint.get("summary"),
                    "tags": endpoint.get("tags"),
                    "requestSchema": endpoint.get("requestSchema"),
                    "responseSchema": endpoint.get("responseSchema"),
                }

                request_schema = endpoint.get("requestSchema")
                if isinstance(request_schema, str) and request_schema:
                    candidate["request_schema_summary"] = self._spec_index.get_schema(
                        request_schema,
                        max_properties=12,
                    )
                    requested_field_paths = _find_requested_field_paths(
                        self._spec_index,
                        request_schema,
                        requested_field_names,
                    )
                    if requested_field_paths:
                        candidate["requested_field_paths"] = requested_field_paths

                candidate_endpoints.append(candidate)
                seen_endpoints.add(endpoint_key)
                if len(candidate_endpoints) >= 8:
                    break
            if len(candidate_endpoints) >= 8:
                break

        return {
            "goal": planner.goal,
            "task_type": planner.task_type,
            "primary_resource": primary_resource,
            "linked_resources": linked_resources,
            "preferred_link_fields": preferred_link_fields,
            "requested_field_names": requested_field_names,
            "explicit_prompt_values": explicit_prompt_values,
            "suggested_first_action": planner.suggested_first_action,
            "success_checks": planner.success_checks[:5],
            "risk_notes": planner.risk_notes[:5],
            "candidate_queries": queries,
            "candidate_endpoints": candidate_endpoints,
            "working_rules": [
                (
                    "Preserve explicit facts from the prompt, such as names, emails, "
                    "and dates. If a requested fact is not top-level in the primary "
                    "schema, inspect nested schema paths or linked subresources before "
                    "declaring the task complete."
                ),
                (
                    "When a prompt names another entity to link, prefer schema relation "
                    "fields whose semantic role matches the prompt wording, such as "
                    "customer, employee, projectManager, or contact."
                ),
                (
                    "Keep actions anchored to the primary target resource. Only explore "
                    "other entity families when they are clearly required linked "
                    "resources for the task."
                ),
                "Prefer existing entities when conflicts or duplicates are plausible.",
                (
                    "For JSON writes, rely on inspected request schemas and send "
                    "only supported fields."
                ),
                (
                    "If a failure indicates a missing linked entity, only create "
                    "or fetch the prerequisite if it directly serves the requested "
                    "task."
                ),
                (
                    "If docs or API responses indicate external or company-level "
                    "setup that the prompt did not ask for, stop instead of "
                    "inventing unrelated configuration work."
                ),
            ],
        }

    def _grounded_candidate_endpoints(
        self,
        planner: PlannerOutput,
        request_prompt: str,
    ) -> list[dict[str, Any]]:
        primary_resource, linked_resources = _infer_task_resources(
            planner=planner,
            request_prompt=request_prompt,
        )
        grounded_queries = [primary_resource, *(linked_resources[:2])]
        candidate_endpoints: list[dict[str, Any]] = []
        seen_endpoints: set[tuple[str, str]] = set()

        for query in grounded_queries:
            for endpoint in self._spec_index.search_endpoints(query, limit=4):
                endpoint_key = (
                    str(endpoint.get("method") or ""),
                    str(endpoint.get("path") or ""),
                )
                if endpoint_key in seen_endpoints:
                    continue
                candidate_endpoints.append(
                    {
                        "query": query,
                        "method": endpoint.get("method"),
                        "path": endpoint.get("path"),
                        "summary": endpoint.get("summary"),
                    }
                )
                seen_endpoints.add(endpoint_key)
                if len(candidate_endpoints) >= 6:
                    return candidate_endpoints

        return candidate_endpoints

    async def _execute(
        self,
        openrouter: OpenRouterClient,
        tripletex: TripletexClient,
        request: SolveRequest,
        planner: PlannerOutput,
        attachment_parts: list[dict[str, Any]],
        execution_state: ExecutionState,
        trace: RunTrace,
    ) -> None:
        tools = build_tool_definitions(
            planner_task_type=planner.task_type,
            request_prompt=request.prompt,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": EXECUTOR_SYSTEM_PROMPT}
        ]
        execution_brief = self._build_execution_brief(
            planner=planner,
            request_prompt=request.prompt,
        )
        user_parts = [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "task_prompt": request.prompt,
                        "planner": planner.model_dump(),
                        "execution_brief": execution_brief,
                        "instructions": [
                            "Use the fewest Tripletex calls possible.",
                            (
                                "Prefer execution_brief candidate_endpoints before "
                                "issuing fresh broad search queries."
                            ),
                            "Use inspect_tripletex_endpoint before risky writes when possible.",
                            "Do not retry identical failed mutations unchanged.",
                            (
                                "Do not repeat the same search or endpoint inspection "
                                "unchanged when prior results were unhelpful."
                            ),
                            "Return completion JSON only when the task is actually done.",
                        ],
                    },
                    ensure_ascii=False,
                ),
            }
        ]
        user_parts.extend(attachment_parts)
        messages.append({"role": "user", "content": user_parts})

        max_steps = max(self._settings.agent_max_steps, 28)
        for _ in range(max_steps):
            message = await openrouter.chat_completion(
                messages=messages,
                tools=tools,
                max_tokens=1800,
            )
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": message.get("content") or "",
                        "tool_calls": tool_calls,
                    }
                )
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    arguments = json.loads(
                        tool_call["function"].get("arguments") or "{}"
                    )
                    tool_call_key = _build_tool_call_key(tool_name, arguments)
                    trace.write(
                        "tool_start",
                        {
                            "tool_call_key": tool_call_key,
                            "tool_name": tool_name,
                            "arguments": arguments,
                        },
                    )
                    tool_result = await self._run_tool(
                        tool_name,
                        arguments,
                        tripletex,
                        execution_state,
                    )
                    trace.write(
                        "tool_result",
                        {
                            "tool_call_key": tool_call_key,
                            "tool_name": tool_name,
                            "result": tool_result,
                        },
                    )
                    blocking_issue = _detect_blocking_issue(
                        planner=planner,
                        request_prompt=request.prompt,
                        tool_name=tool_name,
                        arguments=arguments,
                        tool_result=tool_result,
                    )
                    if blocking_issue is not None:
                        trace.write("blocked", blocking_issue)
                        raise OpenRouterError(blocking_issue["message"])
                    drift_issue = _detect_off_target_drift(
                        planner=planner,
                        request_prompt=request.prompt,
                        tool_name=tool_name,
                        arguments=arguments,
                        execution_state=execution_state,
                    )
                    if drift_issue is not None:
                        grounded_candidate_endpoints = (
                            self._grounded_candidate_endpoints(
                                planner=planner,
                                request_prompt=request.prompt,
                            )
                        )
                        recovery_result = {
                            "ok": False,
                            "error": drift_issue["message"],
                            "hint": drift_issue["hint"],
                            "grounded_primary_resource": drift_issue[
                                "grounded_primary_resource"
                            ],
                            "grounded_linked_resources": drift_issue[
                                "grounded_linked_resources"
                            ],
                            "grounded_candidate_endpoints": grounded_candidate_endpoints,
                        }
                        trace.write("recovery", recovery_result)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": json.dumps(
                                    _compact_tool_result_for_model(recovery_result),
                                    ensure_ascii=False,
                                ),
                            }
                        )
                        continue
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps(
                                _compact_tool_result_for_model(tool_result),
                                ensure_ascii=False,
                            ),
                        }
                    )
                    completion_payload = _build_deterministic_completion_payload(
                        planner=planner,
                        request_prompt=request.prompt,
                        tool_name=tool_name,
                        arguments=arguments,
                        tool_result=tool_result,
                    )
                    if completion_payload is not None:
                        trace.write("final_payload", completion_payload)
                        return
                continue

            content = message.get("content") or ""
            if isinstance(content, list):
                content = "\n".join(str(part) for part in content)
            try:
                final_payload = json.loads(content)
            except json.JSONDecodeError as exc:
                raise OpenRouterError(
                    f"Executor did not return JSON completion payload: {content}"
                ) from exc
            if final_payload.get("status") == "completed":
                trace.write("final_payload", final_payload)
                return
            raise OpenRouterError(
                f"Executor returned unexpected final payload: {final_payload}"
            )

        raise OpenRouterError("Agent reached max steps before completing the task")

    async def _run_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tripletex: TripletexClient,
        execution_state: ExecutionState,
    ) -> dict[str, Any]:
        tool_call_key = _build_tool_call_key(tool_name, arguments)
        if _should_block_repeated_exploration(
            tool_name,
            arguments,
            execution_state.recent_tool_call_keys,
        ):
            return {
                "ok": False,
                "error": "Repeated identical exploration blocked to avoid burning executor steps",
                "hint": (
                    "Reuse the previous search or inspection result, choose one of the "
                    "candidate endpoints already found, or change the query materially "
                    "before exploring again."
                ),
            }

        cached_result = execution_state.cached_tool_results.get(tool_call_key)
        if cached_result is not None:
            return cached_result

        if _should_block_repeated_mutation(
            tool_name,
            arguments,
            execution_state.recent_tool_call_keys,
        ):
            return {
                "ok": False,
                "error": "Repeated identical mutation blocked to avoid looped 4xx errors",
                "hint": (
                    "Inspect the endpoint or schema, or change the payload/params "
                    "before retrying the same mutation."
                ),
            }

        try:
            if tool_name == "search_tripletex_api":
                result = {
                    "ok": True,
                    "results": self._spec_index.search_endpoints(arguments["query"]),
                }
                execution_state.cached_tool_results[tool_call_key] = result
                return result
            if tool_name == "inspect_tripletex_endpoint":
                result = self._inspect_tripletex_endpoint(
                    arguments["method"],
                    arguments["path"],
                    execution_state,
                )
                execution_state.cached_tool_results[tool_call_key] = result
                return result
            if tool_name == "tripletex_request":
                preflight_error, endpoint = self._validate_tripletex_request(
                    arguments,
                    execution_state,
                )
                if preflight_error is not None:
                    return preflight_error
                response = await tripletex.request(
                    method=arguments["method"],
                    path=arguments["path"],
                    params=arguments.get("params"),
                    json_body=arguments.get("json_body"),
                )
                result = {
                    "ok": True,
                    "endpoint": endpoint,
                    "result": response,
                }
                resource_id = _extract_primary_resource_id(response)
                if resource_id is not None:
                    result["resource_id"] = resource_id
                summary = _build_success_summary(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=response,
                )
                if summary:
                    result["summary"] = summary
                return result
            if tool_name == "grant_employee_entitlements":
                response = await tripletex.grant_entitlements(
                    employee_id=arguments["employee_id"],
                    template=arguments["template"],
                    ensure_user_type_extended=arguments.get(
                        "ensure_user_type_extended", False
                    ),
                )
                result = {
                    "ok": True,
                    "result": response,
                    "resource_id": arguments["employee_id"],
                }
                summary = _build_success_summary(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=response,
                )
                if summary:
                    result["summary"] = summary
                return result
        except KeyError as exc:
            return {"ok": False, "error": str(exc)}
        except TripletexApiError as exc:
            result = {
                "ok": False,
                "status_code": exc.status_code,
                "error": exc.body,
            }
            validation_summary = _extract_validation_summary(exc.body)
            if validation_summary:
                result["validation_summary"] = validation_summary
                hint = _build_validation_hint(
                    tool_name=tool_name,
                    arguments=arguments,
                    validation_summary=validation_summary,
                )
                if hint:
                    result["hint"] = hint
            return result
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        return {"ok": False, "error": f"Unknown tool: {tool_name}"}

    def _inspect_tripletex_endpoint(
        self,
        method: str,
        path: str,
        execution_state: ExecutionState,
    ) -> dict[str, Any]:
        endpoint = self._spec_index.get_endpoint(method, path)
        request_schema = endpoint.get("requestSchema")
        response_schema = endpoint.get("responseSchema")
        request_schema_summary = None
        response_schema_summary = None

        if request_schema:
            request_schema_summary = self._spec_index.get_schema(request_schema)
            execution_state.inspected_schemas.add(request_schema)
        if response_schema:
            response_schema_summary = self._spec_index.get_schema(response_schema)

        return {
            "ok": True,
            "endpoint": endpoint,
            "request_schema_summary": request_schema_summary,
            "response_schema_summary": response_schema_summary,
        }

    def _validate_tripletex_request(
        self,
        arguments: dict[str, Any],
        execution_state: ExecutionState,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        method = str(arguments["method"]).upper()
        path = str(arguments["path"])
        try:
            endpoint = self._spec_index.get_endpoint(method, path)
        except KeyError as exc:
            return (
                {
                    "ok": False,
                    "error": str(exc),
                    "hint": (
                        "Call search_tripletex_api or inspect_tripletex_endpoint "
                        "before making a request to an uncertain path."
                    ),
                },
                None,
            )

        if method in {"POST", "PUT"} and arguments.get("json_body") is not None:
            request_schema = endpoint.get("requestSchema")
            if (
                request_schema
                and request_schema not in execution_state.inspected_schemas
            ):
                request_schema_summary = self._spec_index.get_schema(request_schema)
                execution_state.inspected_schemas.add(request_schema)
                return (
                    {
                        "ok": False,
                        "error": (
                            "Schema inspection required before JSON mutation "
                            "requests"
                        ),
                        "required_schema": request_schema,
                        "endpoint": endpoint,
                        "request_schema_summary": request_schema_summary,
                        "hint": (
                            "Use the attached request_schema_summary or call "
                            "inspect_tripletex_endpoint before retrying this write."
                        ),
                    },
                    endpoint,
                )

            if request_schema:
                payload_error = self._validate_json_body_against_schema(
                    request_schema,
                    arguments.get("json_body"),
                )
                if payload_error is not None:
                    payload_error["endpoint"] = endpoint
                    return payload_error, endpoint

        return None, endpoint

    def _validate_json_body_against_schema(
        self,
        schema_name: str,
        payload: Any,
    ) -> dict[str, Any] | None:
        problems = _find_schema_payload_problems(
            self._spec_index,
            schema_name,
            payload,
        )
        if not problems:
            return None

        primary_problem = problems[0]
        return {
            "ok": False,
            "error": primary_problem["message"],
            "schema_validation_errors": problems[:5],
            "hint": (
                "Adjust the JSON body to match the inspected request schema before retrying this write."
            ),
        }


def build_tool_definitions(
    *,
    planner_task_type: str | None = None,
    request_prompt: str | None = None,
) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "search_tripletex_api",
                "description": (
                    "Search the local Tripletex OpenAPI index for relevant "
                    "endpoints before making API calls. Results include "
                    "requestSchema and responseSchema names when available."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_tripletex_endpoint",
                "description": (
                    "Inspect the exact Tripletex endpoint for a method and "
                    "path, including compact request and response schema "
                    "summaries when available."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "enum": ["GET", "POST", "PUT", "DELETE"],
                        },
                        "path": {"type": "string"},
                    },
                    "required": ["method", "path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "tripletex_request",
                "description": (
                    "Send a Tripletex API request using the provided request "
                    "credentials. Use exact paths from the spec. Successful "
                    "results may include a summary and resource_id field."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "enum": ["GET", "POST", "PUT", "DELETE"],
                        },
                        "path": {"type": "string"},
                        "params": {"type": "object", "additionalProperties": True},
                        "json_body": {
                            "oneOf": [
                                {"type": "object", "additionalProperties": True},
                                {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": True,
                                    },
                                },
                                {"type": "null"},
                            ]
                        },
                    },
                    "required": ["method", "path"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    if _should_include_entitlement_tool(
        planner_task_type=planner_task_type,
        request_prompt=request_prompt,
    ):
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": "grant_employee_entitlements",
                    "description": (
                        "Grant a known Tripletex employee entitlement template, "
                        "such as ALL_PRIVILEGES, to an employee."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "employee_id": {"type": "integer"},
                            "template": {
                                "type": "string",
                                "enum": [
                                    "NONE_PRIVILEGES",
                                    "ALL_PRIVILEGES",
                                    "INVOICING_MANAGER",
                                    "PERSONELL_MANAGER",
                                    "ACCOUNTANT",
                                    "AUDITOR",
                                    "DEPARTMENT_LEADER",
                                ],
                            },
                            "ensure_user_type_extended": {"type": "boolean"},
                        },
                        "required": ["employee_id", "template"],
                        "additionalProperties": False,
                    },
                },
            }
        )

    return definitions


def _should_include_entitlement_tool(
    *,
    planner_task_type: str | None,
    request_prompt: str | None,
) -> bool:
    combined = f"{planner_task_type or ''} {request_prompt or ''}".lower()
    return any(
        phrase in combined
        for phrase in [
            "employee",
            "ansatt",
            "entitlement",
            "privilege",
            "administrator access",
            "admin access",
            "brukertilgang",
            "tilgang",
            "rolle",
            "role",
        ]
    )


def _should_block_repeated_exploration(
    tool_name: str,
    arguments: dict[str, Any],
    recent_tool_call_keys: list[str],
) -> bool:
    if tool_name not in {"search_tripletex_api", "inspect_tripletex_endpoint"}:
        return False

    tool_call_key = _build_tool_call_key(tool_name, arguments)
    return recent_tool_call_keys[-1:] == [tool_call_key]


def _build_tool_call_key(tool_name: str, arguments: dict[str, Any]) -> str:
    return json.dumps(
        {
            "tool_name": tool_name,
            "arguments": arguments,
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _compact_tool_result_for_model(tool_result: dict[str, Any]) -> dict[str, Any]:
    compacted = compact_response(
        tool_result,
        max_depth=4,
        max_items=6,
        max_string=250,
    )
    if isinstance(compacted, dict):
        return compacted
    return {"ok": False, "error": "Unexpected non-dict tool result"}


def _find_schema_payload_problems(
    spec_index: TripletexSpecIndex,
    schema_name: str,
    payload: Any,
    *,
    path: str = "json_body",
    depth: int = 0,
    max_depth: int = 4,
) -> list[dict[str, Any]]:
    if depth > max_depth:
        return []

    try:
        schema = spec_index.get_schema_definition(schema_name)
    except KeyError:
        return []

    if not isinstance(payload, dict):
        return [
            {
                "path": path,
                "message": f"`{path}` should be an object matching schema `{schema_name}`.",
            }
        ]

    properties = schema.get("properties") or {}
    writable_fields = [
        name
        for name, definition in properties.items()
        if isinstance(definition, dict) and not definition.get("readOnly")
    ]
    problems: list[dict[str, Any]] = []

    for field_name, value in payload.items():
        field_path = f"{path}.{field_name}"
        definition = properties.get(field_name)
        if not isinstance(definition, dict):
            suggestions = _suggest_schema_fields(field_name, writable_fields)
            message = f"Unsupported field `{field_path}` for schema `{schema_name}`."
            if suggestions:
                message += (
                    " Closest valid fields: "
                    + ", ".join(f"`{item}`" for item in suggestions)
                    + "."
                )
            problems.append(
                {
                    "path": field_path,
                    "message": message,
                }
            )
            continue

        if definition.get("readOnly"):
            problems.append(
                {
                    "path": field_path,
                    "message": (
                        f"`{field_path}` is read-only in schema `{schema_name}` "
                        "and should not be sent in write payloads."
                    ),
                }
            )
            continue

        ref = definition.get("$ref")
        if isinstance(ref, str):
            ref_schema_name = ref.rsplit("/", maxsplit=1)[-1]
            if not isinstance(value, dict):
                problems.append(
                    {
                        "path": field_path,
                        "message": (
                            f"`{field_path}` should be an object matching schema `{ref_schema_name}`."
                        ),
                    }
                )
                continue
            problems.extend(
                _find_schema_payload_problems(
                    spec_index,
                    ref_schema_name,
                    value,
                    path=field_path,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
            )
            continue

        field_type = definition.get("type")
        if field_type == "array":
            if not isinstance(value, list):
                problems.append(
                    {
                        "path": field_path,
                        "message": f"`{field_path}` should be an array.",
                    }
                )
                continue

            item_definition = definition.get("items")
            if not isinstance(item_definition, dict):
                continue

            item_ref = item_definition.get("$ref")
            if isinstance(item_ref, str):
                item_schema_name = item_ref.rsplit("/", maxsplit=1)[-1]
                for index, item in enumerate(value[:5]):
                    item_path = f"{field_path}[{index}]"
                    if not isinstance(item, dict):
                        problems.append(
                            {
                                "path": item_path,
                                "message": (
                                    f"`{item_path}` should be an object matching schema `{item_schema_name}`."
                                ),
                            }
                        )
                        continue
                    problems.extend(
                        _find_schema_payload_problems(
                            spec_index,
                            item_schema_name,
                            item,
                            path=item_path,
                            depth=depth + 1,
                            max_depth=max_depth,
                        )
                    )
            continue

        if field_type == "object" and isinstance(definition.get("properties"), dict):
            if not isinstance(value, dict):
                problems.append(
                    {
                        "path": field_path,
                        "message": f"`{field_path}` should be an object.",
                    }
                )

    return problems


def _suggest_schema_fields(field_name: str, allowed_fields: list[str]) -> list[str]:
    normalized_field = field_name.lower()
    suggestions: list[str] = []

    alias_map = {
        "customerid": ["customer"],
        "supplierid": ["supplier"],
        "employeeid": ["employee"],
        "departmentid": ["department"],
        "projectid": ["project"],
        "productid": ["product"],
        "invoiceid": ["invoice"],
        "quantity": ["count"],
        "qty": ["count"],
        "unitprice": ["unitPriceExcludingVatCurrency", "unitPriceIncludingVatCurrency"],
        "amount": ["unitPriceExcludingVatCurrency", "unitPriceIncludingVatCurrency"],
        "price": [
            "priceExcludingVatCurrency",
            "priceIncludingVatCurrency",
            "unitPriceExcludingVatCurrency",
        ],
        "invoicelines": ["orderLines", "orders"],
    }
    for alias in alias_map.get(normalized_field, []):
        if alias in allowed_fields and alias not in suggestions:
            suggestions.append(alias)

    fuzzy_matches = get_close_matches(field_name, allowed_fields, n=5, cutoff=0.45)
    for match in fuzzy_matches:
        if match not in suggestions:
            suggestions.append(match)

    return suggestions[:3]


def _should_block_repeated_mutation(
    tool_name: str,
    arguments: dict[str, Any],
    recent_tool_call_keys: list[str],
) -> bool:
    if tool_name == "tripletex_request":
        method = str(arguments.get("method") or "").upper()
        if method not in {"POST", "PUT", "DELETE"}:
            return False
    elif tool_name != "grant_employee_entitlements":
        return False

    tool_call_key = _build_tool_call_key(tool_name, arguments)
    return recent_tool_call_keys[-1:] == [tool_call_key]


def _extract_primary_resource_id(result: Any) -> int | None:
    if isinstance(result, dict):
        value = result.get("value")
        if isinstance(value, dict):
            value_id = value.get("id")
            if isinstance(value_id, int):
                return value_id

        result_id = result.get("id")
        if isinstance(result_id, int):
            return result_id

        values = result.get("values")
        if isinstance(values, list) and len(values) == 1:
            first_value = values[0]
            if isinstance(first_value, dict):
                first_value_id = first_value.get("id")
                if isinstance(first_value_id, int):
                    return first_value_id
    return None


def _build_deterministic_completion_payload(
    *,
    planner: PlannerOutput,
    request_prompt: str,
    tool_name: str,
    arguments: dict[str, Any],
    tool_result: dict[str, Any],
) -> dict[str, Any] | None:
    if not tool_result.get("ok"):
        return None

    if tool_name == "grant_employee_entitlements":
        summary = _build_success_summary(
            tool_name=tool_name,
            arguments=arguments,
            result=tool_result.get("result"),
        )
        if summary is None:
            summary = "Dedicated entitlement update completed successfully."
        return {"status": "completed", "summary": summary}

    if tool_name != "tripletex_request":
        return None

    method = str(arguments.get("method") or "").upper()
    path = str(arguments.get("path") or "")
    resource_id = tool_result.get("resource_id")

    expected_methods, target, verb = _infer_task_operation(planner.task_type)
    if not expected_methods or target is None or verb is None:
        inferred_target, _ = _infer_task_resources(
            planner=planner,
            request_prompt=request_prompt,
        )
        target = inferred_target
        if method == "POST":
            expected_methods = {"POST", "PUT"}
            verb = "created"
        elif method == "PUT":
            expected_methods = {"PUT", "POST"}
            verb = "updated"
        elif method == "DELETE":
            expected_methods = {"DELETE", "PUT"}
            verb = "deleted"
        else:
            return None
    if method not in expected_methods:
        return None
    if not _task_target_matches_path(target, path):
        return None

    summary_target = _humanize_task_target(target)
    if isinstance(resource_id, int) and method != "DELETE":
        return {
            "status": "completed",
            "summary": f"{summary_target} {resource_id} was {verb} successfully.",
        }
    return {
        "status": "completed",
        "summary": f"{summary_target} was {verb} successfully.",
    }

    return None


def _build_success_summary(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
) -> str | None:
    if tool_name == "grant_employee_entitlements":
        employee_id = arguments.get("employee_id")
        if isinstance(employee_id, int):
            return (
                "Employee entitlements updated successfully for employee "
                f"{employee_id}."
            )
        return "Employee entitlements updated successfully."

    if tool_name != "tripletex_request":
        return None

    method = str(arguments.get("method") or "").upper()
    path = str(arguments.get("path") or "")
    resource_id = _extract_primary_resource_id(result)

    if method == "POST" and resource_id is not None:
        return f"Created resource via POST {path} with id {resource_id}."
    if method == "PUT" and resource_id is not None:
        return f"Updated resource via PUT {path} with id {resource_id}."
    if method == "PUT":
        return f"PUT {path} succeeded."
    if method == "DELETE":
        return f"DELETE {path} succeeded."

    return None


def _extract_validation_summary(body: Any) -> list[dict[str, str]]:
    if not isinstance(body, dict):
        return []
    validation_messages = body.get("validationMessages")
    if not isinstance(validation_messages, list):
        return []

    summary: list[dict[str, str]] = []
    for item in validation_messages[:6]:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "")
        message = str(item.get("message") or "")
        summary.append(
            {
                "field": field,
                "message": message,
            }
        )
    return summary


def _build_execution_brief_queries(
    planner: PlannerOutput,
    request_prompt: str,
) -> list[str]:
    primary_resource, linked_resources = _infer_task_resources(
        planner=planner,
        request_prompt=request_prompt,
    )
    raw_queries = [
        primary_resource.replace("_", " "),
        *(resource.replace("_", " ") for resource in linked_resources[:2]),
        planner.task_type.replace("_", " "),
        planner.goal,
        planner.suggested_first_action,
        *planner.likely_resources[:3],
        request_prompt,
    ]
    queries: list[str] = []
    seen: set[str] = set()
    for raw_query in raw_queries:
        normalized_query = " ".join(str(raw_query).split())
        if not normalized_query:
            continue
        condensed_query = " ".join(normalized_query.split()[:12])
        if condensed_query in seen:
            continue
        queries.append(condensed_query)
        seen.add(condensed_query)
        if len(queries) >= 5:
            break
    return queries


def _infer_task_operation(
    task_type: str,
) -> tuple[set[str], str | None, str | None]:
    normalized = task_type.lower()
    operation_prefixes = [
        ("create_", {"POST", "PUT"}, "created"),
        ("register_", {"POST", "PUT"}, "registered"),
        ("issue_", {"POST", "PUT"}, "issued"),
        ("update_", {"PUT", "POST"}, "updated"),
        ("set_", {"PUT", "POST"}, "updated"),
        ("enable_", {"PUT", "POST"}, "updated"),
        ("delete_", {"DELETE", "PUT"}, "deleted"),
        ("remove_", {"DELETE", "PUT"}, "deleted"),
        ("reverse_", {"DELETE", "PUT", "POST"}, "reversed"),
    ]
    for prefix, methods, verb in operation_prefixes:
        if normalized.startswith(prefix):
            return methods, normalized[len(prefix) :], verb
    return set(), None, None


def _infer_task_resources(
    *,
    planner: PlannerOutput,
    request_prompt: str,
) -> tuple[str, list[str]]:
    text = " ".join(
        [
            planner.task_type,
            planner.goal,
            planner.suggested_first_action,
            *planner.likely_resources,
            request_prompt,
        ]
    ).lower()

    resource_aliases = {
        "project": ["project", "prosjekt"],
        "customer": ["customer", "kunde"],
        "invoice": ["invoice", "regning", "faktura"],
        "order": ["order", "ordre"],
        "product": ["product", "produkt"],
        "employee": ["employee", "ansatt"],
        "department": ["department", "avdeling"],
        "payment": ["payment", "betaling"],
        "travel_expense": ["travel expense", "reiseregning"],
    }

    scored_resources: list[tuple[int, str]] = []
    for resource_name, aliases in resource_aliases.items():
        score = 0
        for alias in aliases:
            if alias in request_prompt.lower():
                score += 6
            if alias in planner.goal.lower():
                score += 5
            if alias in planner.task_type.lower():
                score += 4
            if alias in text:
                score += 1
        if score > 0:
            scored_resources.append((score, resource_name))

    scored_resources.sort(key=lambda item: (-item[0], item[1]))
    if not scored_resources:
        inferred_methods, inferred_target, _ = _infer_task_operation(planner.task_type)
        if inferred_methods and inferred_target:
            return inferred_target, []
        return "resource", []

    primary_resource = scored_resources[0][1]
    linked_resources = [
        resource_name
        for _, resource_name in scored_resources[1:]
        if resource_name != primary_resource
    ]
    return primary_resource, linked_resources[:3]


def _infer_relation_roles(
    *,
    planner: PlannerOutput,
    request_prompt: str,
) -> list[str]:
    combined = " ".join(
        [
            planner.task_type,
            planner.goal,
            planner.suggested_first_action,
            *planner.likely_resources,
            request_prompt,
        ]
    ).lower()

    role_aliases = {
        "customer": ["customer", "kunde", "client", "klient"],
        "employee": ["employee", "ansatt", "medarbeider"],
        "projectManager": [
            "project manager",
            "prosjektleder",
            "projectmanager",
        ],
        "contact": ["contact", "kontakt", "contact person", "kontaktperson"],
        "department": ["department", "avdeling"],
        "supplier": ["supplier", "leverandør", "leverandor"],
    }

    scored_roles: list[tuple[int, str]] = []
    for role_name, aliases in role_aliases.items():
        score = 0
        for alias in aliases:
            if alias in request_prompt.lower():
                score += 5
            if alias in planner.goal.lower():
                score += 4
            if alias in combined:
                score += 1
        if score > 0:
            scored_roles.append((score, role_name))

    scored_roles.sort(key=lambda item: (-item[0], item[1]))
    return [role_name for _, role_name in scored_roles[:3]]


def _infer_requested_schema_fields(request_prompt: str) -> list[str]:
    prompt_lower = request_prompt.lower()
    requested_fields: list[str] = []

    field_aliases = {
        "email": ["email", "e-post", "epost", "correo", "courriel"],
        "dateOfBirth": [
            "date of birth",
            "birth date",
            "born",
            "født",
            "fødd",
            "nacido",
            "nacida",
            "geboren",
            "né",
            "nee",
            "née",
        ],
        "startDate": [
            "start date",
            "starting date",
            "fecha de inicio",
            "fecha inicial",
            "startdato",
            "oppstartsdato",
            "startdatum",
            "fecha de alta",
        ],
        "endDate": [
            "end date",
            "fecha de fin",
            "sluttdato",
            "ended",
            "ends on",
            "til dato",
            "enddatum",
        ],
    }

    if re.search(
        r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", request_prompt, re.IGNORECASE
    ):
        requested_fields.append("email")

    for field_name, aliases in field_aliases.items():
        if field_name in requested_fields:
            continue
        if any(alias in prompt_lower for alias in aliases):
            requested_fields.append(field_name)

    return requested_fields


def _extract_explicit_prompt_values(request_prompt: str) -> dict[str, list[str]]:
    prompt_values: dict[str, list[str]] = {}

    emails = re.findall(
        r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
        request_prompt,
        flags=re.IGNORECASE,
    )
    if emails:
        prompt_values["emails"] = list(dict.fromkeys(emails))

    date_matches = re.findall(
        r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\.\s*[A-Za-zÀ-ÿ]+\s+\d{4}\b",
        request_prompt,
    )
    if date_matches:
        prompt_values["dates"] = list(dict.fromkeys(date_matches))

    return prompt_values


def _find_requested_field_paths(
    spec_index: TripletexSpecIndex,
    schema_name: str,
    requested_field_names: list[str],
    *,
    path_prefix: str = "",
    depth: int = 0,
    max_depth: int = 3,
    visited_schemas: set[str] | None = None,
) -> list[str]:
    if not requested_field_names or depth > max_depth:
        return []

    visited = set(visited_schemas or set())
    if schema_name in visited:
        return []
    visited.add(schema_name)

    schema_definition = spec_index.get_schema_definition(schema_name)
    properties = schema_definition.get("properties")
    if not isinstance(properties, dict):
        return []

    matches: list[str] = []
    requested_names = set(requested_field_names)
    for field_name, definition in properties.items():
        field_path = f"{path_prefix}{field_name}"
        if field_name in requested_names and field_path not in matches:
            matches.append(field_path)

        if not isinstance(definition, dict):
            continue

        ref = definition.get("$ref")
        if isinstance(ref, str):
            ref_schema_name = ref.rsplit("/", maxsplit=1)[-1]
            for match in _find_requested_field_paths(
                spec_index,
                ref_schema_name,
                requested_field_names,
                path_prefix=f"{field_path}.",
                depth=depth + 1,
                max_depth=max_depth,
                visited_schemas=visited,
            ):
                if match not in matches:
                    matches.append(match)
            continue

        if definition.get("type") == "array":
            item_definition = definition.get("items")
            if not isinstance(item_definition, dict):
                continue
            item_ref = item_definition.get("$ref")
            if isinstance(item_ref, str):
                item_schema_name = item_ref.rsplit("/", maxsplit=1)[-1]
                for match in _find_requested_field_paths(
                    spec_index,
                    item_schema_name,
                    requested_field_names,
                    path_prefix=f"{field_path}[].",
                    depth=depth + 1,
                    max_depth=max_depth,
                    visited_schemas=visited,
                ):
                    if match not in matches:
                        matches.append(match)
            continue

        nested_properties = definition.get("properties")
        if definition.get("type") == "object" and isinstance(nested_properties, dict):
            inline_schema_name = f"{schema_name}.{field_name}"
            try:
                spec_index.get_schema_definition(inline_schema_name)
            except KeyError:
                nested_matches = _find_requested_field_paths_in_properties(
                    nested_properties,
                    requested_field_names,
                    path_prefix=f"{field_path}.",
                    depth=depth + 1,
                    max_depth=max_depth,
                )
                for match in nested_matches:
                    if match not in matches:
                        matches.append(match)

    return matches[:8]


def _find_requested_field_paths_in_properties(
    properties: dict[str, Any],
    requested_field_names: list[str],
    *,
    path_prefix: str,
    depth: int,
    max_depth: int,
) -> list[str]:
    if depth > max_depth:
        return []

    requested_names = set(requested_field_names)
    matches: list[str] = []
    for field_name, definition in properties.items():
        field_path = f"{path_prefix}{field_name}"
        if field_name in requested_names and field_path not in matches:
            matches.append(field_path)
        if not isinstance(definition, dict):
            continue
        nested_properties = definition.get("properties")
        if definition.get("type") == "object" and isinstance(nested_properties, dict):
            for match in _find_requested_field_paths_in_properties(
                nested_properties,
                requested_field_names,
                path_prefix=f"{field_path}.",
                depth=depth + 1,
                max_depth=max_depth,
            ):
                if match not in matches:
                    matches.append(match)
    return matches[:8]


def _normalize_identifier(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _task_target_matches_path(target: str, path: str) -> bool:
    normalized_target = _normalize_identifier(target)
    normalized_path = _normalize_identifier(path)
    return bool(normalized_target) and normalized_target in normalized_path


def _extract_endpoint_family(
    tool_name: str,
    arguments: dict[str, Any],
) -> str | None:
    if tool_name == "grant_employee_entitlements":
        return "employee"
    if tool_name != "tripletex_request":
        return None

    path = str(arguments.get("path") or "")
    if not path.startswith("/"):
        return None

    family = path.split("?", maxsplit=1)[0].strip("/").split("/", maxsplit=1)[0]
    return family or None


def _detect_off_target_drift(
    *,
    planner: PlannerOutput,
    request_prompt: str,
    tool_name: str,
    arguments: dict[str, Any],
    execution_state: ExecutionState,
) -> dict[str, str] | None:
    endpoint_family = _extract_endpoint_family(tool_name, arguments)
    if endpoint_family is None:
        return None

    execution_state.recent_endpoint_families.append(endpoint_family)
    if len(execution_state.recent_endpoint_families) > 8:
        execution_state.recent_endpoint_families = (
            execution_state.recent_endpoint_families[-8:]
        )

    primary_resource, linked_resources = _infer_task_resources(
        planner=planner,
        request_prompt=request_prompt,
    )
    allowed_families = {
        primary_resource,
        *(resource for resource in linked_resources),
    }
    if endpoint_family in allowed_families:
        return None

    trailing_families = execution_state.recent_endpoint_families[-3:]
    if len(trailing_families) < 3 or any(
        family != endpoint_family for family in trailing_families
    ):
        return None

    return {
        "message": (
            f"Run is drifting into the `{endpoint_family}` endpoint family even though "
            f"the grounded primary resource is `{primary_resource}`. Stop and return to "
            "the target resource or a clearly linked prerequisite."
        ),
        "hint": (
            f"Re-anchor on the grounded primary resource `{primary_resource}`. "
            "Use execution_brief candidate endpoints for that resource, or inspect a "
            "clearly linked prerequisite instead of continuing in the current endpoint "
            "family."
        ),
        "grounded_primary_resource": primary_resource,
        "grounded_linked_resources": linked_resources,
    }


def _humanize_task_target(target: str) -> str:
    return target.replace("_", " ").strip().title()


def _messages_indicate_external_blocker(messages_lower: str) -> bool:
    return any(
        phrase in messages_lower
        for phrase in [
            "requires setup done by tripletex",
            "must be activated prior",
            "cannot be created before the company has",
            "kan ikke opprettes før selskapet har",
            "company has registered",
            "not activated prior",
            "ikke aktivert",
        ]
    )


def _task_mentions_configuration_work(
    planner: PlannerOutput,
    request_prompt: str,
) -> bool:
    combined = f"{planner.task_type} {planner.goal} {request_prompt}".lower()
    return any(
        phrase in combined
        for phrase in [
            "setup",
            "configuration",
            "config",
            "module",
            "settings",
            "bank account",
            "bankkonto",
            "bankkontonummer",
            "altinn",
        ]
    )


def _detect_blocking_issue(
    *,
    planner: PlannerOutput,
    request_prompt: str,
    tool_name: str,
    arguments: dict[str, Any],
    tool_result: dict[str, Any],
) -> dict[str, str] | None:
    if tool_name != "tripletex_request" or tool_result.get("ok"):
        return None

    validation_summary = tool_result.get("validation_summary") or []
    messages = " ".join(
        str(item.get("message") or "")
        for item in validation_summary
        if isinstance(item, dict)
    )
    if not messages:
        error = tool_result.get("error")
        if isinstance(error, dict):
            messages = json.dumps(error, ensure_ascii=False)
        else:
            messages = str(error or "")

    messages_lower = messages.lower()
    if not _messages_indicate_external_blocker(messages_lower):
        return None
    if _task_mentions_configuration_work(planner, request_prompt):
        return None

    path = str(arguments.get("path") or "")
    return {
        "message": (
            "Task is blocked by an external or company-level prerequisite unrelated "
            f"to the requested outcome. Stop instead of exploring unrelated setup after {path}: {messages}"
        )
    }


def _build_validation_hint(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    validation_summary: list[dict[str, str]],
) -> str | None:
    fields = {item.get("field", "") for item in validation_summary}
    messages = " ".join(item.get("message", "") for item in validation_summary)
    messages_lower = messages.lower()

    if tool_name == "tripletex_request":
        method = str(arguments.get("method") or "").upper()
        if _messages_indicate_external_blocker(messages_lower):
            return (
                "The API is reporting external or company-level setup that is not "
                "a normal field correction. Do not invent unrelated configuration "
                "work. Confirm whether that setup is actually part of the requested "
                "task before continuing."
            )
        if method == "POST" and any(
            phrase in messages_lower
            for phrase in [
                "finnes allerede",
                "already exists",
                "allerede registrert",
                "duplicate",
                "conflict",
            ]
        ):
            return (
                "This create appears to conflict with an existing resource. Search "
                "for the existing resource using exact identifying fields from the "
                "prompt, then update or reuse it instead of retrying the same create."
            )

        if any(
            phrase in messages_lower
            for phrase in [
                'kan ikke være "0"',
                'cannot be "0"',
                "cannot be empty",
                "invalid enum",
                "must be one of",
                "must be equal to one of the allowed values",
            ]
        ):
            return (
                "A typed or enum field appears to be missing or in the wrong format. "
                "Re-inspect the request schema and use the exact supported enum "
                "literals or field shape from the schema instead of guessing numeric "
                "placeholders."
            )

        linked_reference_fields = [
            field_name
            for field_name in fields
            if field_name.endswith(".id") or field_name.lower().endswith("id")
        ]
        if linked_reference_fields:
            linked_field = linked_reference_fields[0]
            reference_name = linked_field.removesuffix(".id").removesuffix("Id")
            reference_name = reference_name or linked_field
            return (
                f"This write is missing a linked `{reference_name}` reference. "
                "Find or create that prerequisite only if it directly serves the "
                "requested task, then retry with a minimal reference object such "
                f'as `{{"{reference_name}": {{"id": ...}}}}`.'
            )

        if any(
            phrase in messages_lower
            for phrase in [
                "kan ikke være null",
                "må fylles ut",
                "must not be null",
                "required",
            ]
        ):
            return (
                "One or more required fields or prerequisite references are missing. "
                "Fix all clearly indicated required values in one retry instead of "
                "making small incremental guesses."
            )

        if any(
            phrase in messages_lower
            for phrase in [
                "eksisterer ikke i objektet",
                "does not exist in object",
                "unsupported field",
            ]
        ):
            return (
                "The payload includes unsupported fields for this schema. Remove "
                "unsupported fields, inspect the request schema again, and split "
                "adjacent actions into separate supported operations when necessary."
            )

    if tool_name == "grant_employee_entitlements":
        if "dateOfBirth" in fields:
            return (
                "The entitlement flow is failing because the employee record lacks `dateOfBirth`. "
                "Update the employee with required fields before retrying entitlements."
            )

    return None
