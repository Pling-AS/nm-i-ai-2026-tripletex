from collections import defaultdict
from difflib import get_close_matches
import json
from pathlib import Path
import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import httpx  # pyright: ignore[reportMissingImports]
from pydantic import ValidationError  # pyright: ignore[reportMissingImports]

import logging

logger = logging.getLogger(__name__)

from tripletex_agent.config import Settings
from tripletex_agent.executor_knowledge import (
    get_endpoint_chain,
    get_domain_knowledge,
    select_field_rules,
    select_successful_trace,
)
from tripletex_agent.files import prepare_attachments, prepare_attachments_async
from tripletex_agent.memory_palace import MemoryPalace  # pyright: ignore[reportMissingImports]
from tripletex_agent.middleware import ExecutionMiddleware
from tripletex_agent.mutation_fuzzer import MutationFuzzer  # pyright: ignore[reportMissingImports]
from tripletex_agent.openrouter import (
    OpenRouterClient,
    OpenRouterError,
    _is_anthropic_model,
)
from tripletex_agent.prompts import (
    EXECUTOR_PLAYBOOKS,
    PLANNER_SYSTEM_PROMPT,
    build_executor_system_prompt,
)
from tripletex_agent.schema_validator import validate_and_fix_payload
from tripletex_agent.schemas import PlannerOutput, SolveRequest
from tripletex_agent.shadow_discovery import discover_sandbox_parallel  # pyright: ignore[reportMissingImports]
from tripletex_agent.spec_index import TripletexSpecIndex
from tripletex_agent.trace import RunTrace
from tripletex_agent.trace_compiler import (  # pyright: ignore[reportMissingImports]
    compile_trace,
    match_compiled_trace,
    replay_trace,
)
from tripletex_agent.tripletex import (
    TripletexApiError,
    TripletexClient,
    compact_response,
)
from tripletex_agent.version import AGENT_VERSION

_SKIP_DISCOVERY_TASK_TYPES: frozenset[str] = frozenset(
    {
        "create_supplier",
        "create_customer",
        "create_product",
        "create_department",
    }
)

_SKIP_VERIFY_TASK_TYPES: frozenset[str] = frozenset(
    {
        "create_supplier",
        "create_customer",
        "create_product",
        "create_department",
    }
)

_DETERMINISTIC_TASK_TYPES: frozenset[str] = frozenset(
    {
        "create_supplier",
        "create_customer",
        "create_department",
    }
)


@dataclass(slots=True)
class AgentRunResult:
    planner: PlannerOutput
    stats: dict[str, Any]


@dataclass(slots=True)
class CreatedResource:
    path: str
    resource_id: int | None
    method: str
    key_fields: dict[str, Any]


@dataclass(slots=True)
class ExecutionState:
    inspected_schemas: set[str] = field(default_factory=set)
    cached_tool_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    recent_tool_call_keys: list[str] = field(default_factory=list)
    recent_endpoint_families: list[str] = field(default_factory=list)
    enforcer_override_active: bool = False
    enforcer_override_count: int = 0
    last_enforcer_suggestion: str = ""
    created_entity_keys: set[str] = field(default_factory=set)
    advisor_recovery_keys: set[str] = field(default_factory=set)
    consecutive_proxy_errors: int = 0
    start_time: float = 0.0
    total_api_errors: int = 0
    advisor_422_endpoints: set[str] = field(default_factory=set)
    created_resources: list[CreatedResource] = field(default_factory=list)
    sandbox_discovery: dict[str, Any] = field(default_factory=dict)
    pre_resolved_accounts: dict[int, int | None] = field(default_factory=dict)
    middleware: Any = None
    reclassification_done: bool = False
    off_plan_api_calls: int = 0


class TripletexAccountingAgent:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._spec_index = TripletexSpecIndex(settings.tripletex_api_spec_path)

    async def solve(
        self, request: SolveRequest, submission_id: str | None = None
    ) -> AgentRunResult:
        execution_state = ExecutionState(start_time=time.monotonic())
        credentials = request.tripletex_credentials
        if credentials is None:
            raise OpenRouterError("Missing Tripletex credentials in solve request")

        base_url = str(credentials.base_url)
        is_competition = "tx-proxy" in base_url

        is_simple_prompt = len(request.prompt) < 300 and len(request.files) == 0
        planner_chain = _build_planner_model_chain(
            self._settings, fast=is_simple_prompt
        )

        import platform as _platform

        metadata: dict[str, Any] = {
            "model": self._settings.openrouter_model,
            "planner_model": planner_chain[0],
            "max_steps": self._settings.agent_max_steps,
            "temperature": self._settings.agent_model_temperature,
            "http_timeout": self._settings.http_timeout_seconds,
            "max_attachment_chars": self._settings.max_attachment_text_chars,
            "source": "competition" if is_competition else "simulation",
            "hostname": _platform.node(),
            "agent_version": AGENT_VERSION,
        }
        if submission_id:
            metadata["submission_id"] = submission_id
        trace = RunTrace(metadata=metadata, prompt=request.prompt)
        trace.write(
            "init",
            {
                "prompt": request.prompt,
                "file_count": len(request.files),
                "base_url": base_url,
                "metadata": metadata,
            },
        )
        try:
            attachments = await prepare_attachments_async(
                request.files,
                self._settings.max_attachment_text_chars,
                datalab_api_key=self._settings.datalab_api_key,
            )
        except Exception as att_exc:
            logger.warning(
                "Async attachment prep failed, falling back to sync: %s", att_exc
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
                ],
                "pdf_extractions": attachments.extraction_details,
            },
        )
        openrouter = OpenRouterClient(self._settings)
        tripletex = TripletexClient(
            base_url=str(credentials.base_url),
            session_token=credentials.session_token,
            timeout=self._settings.http_timeout_seconds,
        )
        middleware = ExecutionMiddleware()
        execution_state.middleware = middleware
        try:
            planner = await self._plan(
                openrouter,
                request,
                attachments.summaries,
                model_chain=planner_chain,
            )
            trace.write("planner", planner.model_dump())

            corrected = _maybe_correct_task_type(planner, trace)
            if corrected:
                planner = corrected

            task_tier = _classify_task_tier(planner.task_type)

            if planner.task_type in _SKIP_DISCOVERY_TASK_TYPES:
                sandbox_discovery: dict[str, Any] = {}
                trace.write(
                    "sandbox_discovery_skipped", {"task_type": planner.task_type}
                )
            else:
                if task_tier >= 3:
                    try:
                        shadow_state = await discover_sandbox_parallel(tripletex)
                        trace.write(
                            "shadow_discovery",
                            {"summary": shadow_state.to_context_string()[:500]},
                        )
                    except Exception as e:
                        logger.debug("Shadow discovery failed: %s", e)
                sandbox_discovery = await self._discover_sandbox_entities(
                    tripletex,
                    planner,
                    execution_state,
                    trace,
                )
            executor_chain = _build_executor_model_chain(self._settings, task_tier)
            logger.info(
                "Routing: task_type=%s tier=%d planner=%s executor=%s",
                planner.task_type,
                task_tier,
                planner_chain[0],
                executor_chain[0],
            )

            trace.update_metadata(
                {
                    "task_tier": task_tier,
                    "executor_model": executor_chain[0],
                    "executor_chain": executor_chain,
                    "planner_chain": planner_chain,
                }
            )

            execution_state.sandbox_discovery = sandbox_discovery

            await self._setup_entities_deterministic(
                tripletex,
                planner,
                execution_state,
                trace,
            )

            pre_resolved_accounts = await self._batch_resolve_accounts(
                tripletex,
                request.prompt,
                execution_state,
                trace,
                planner=planner,
            )
            execution_state.pre_resolved_accounts = pre_resolved_accounts

            deterministic_result = await self._try_deterministic_execution(
                tripletex, planner, execution_state, trace
            )
            if deterministic_result is None:
                try:
                    compiled_dir = (
                        Path(__file__).parent.parent.parent / "runs" / "compiled_traces"
                    )
                    compiled = match_compiled_trace(
                        request.prompt,
                        planner.task_type,
                        compiled_dir,
                    )
                    if compiled:
                        trace.write(
                            "sniper_mode",
                            {
                                "task_type": compiled.task_type,
                                "source": compiled.source_trace,
                                "compiled_calls": compiled.total_calls,
                            },
                        )
                        calls_before_sniper = tripletex.call_count
                        registry = (
                            execution_state.middleware.registry
                            if execution_state.middleware
                            else None
                        )
                        replay_results = replay_trace(compiled, tripletex, registry)
                        calls_after_sniper = tripletex.call_count
                        sniper_success = (
                            all(
                                isinstance(r, dict) and r.get("ok", False)
                                for r in replay_results
                            )
                            and calls_after_sniper > calls_before_sniper
                        )
                        if sniper_success:
                            trace.write(
                                "sniper_complete",
                                {"steps": len(replay_results), "all_ok": True},
                            )
                            deterministic_result = {"ok": True, "sniper": True}
                except Exception as e:
                    logger.debug("Sniper mode failed, falling back to executor: %s", e)

                if deterministic_result is None:
                    await self._execute(
                        openrouter,
                        tripletex,
                        request,
                        planner,
                        attachments.executor_content_parts,
                        execution_state,
                        trace,
                        model_chain=executor_chain,
                    )

            if planner.task_type not in _SKIP_VERIFY_TASK_TYPES:
                await self._verify_and_repair(
                    openrouter,
                    tripletex,
                    planner,
                    execution_state,
                    trace,
                )
            else:
                trace.write("verify_skipped", {"task_type": planner.task_type})
            trace.write(
                "done",
                {
                    "tripletex_call_count": tripletex.call_count,
                    "tripletex_error_count": tripletex.error_count,
                    "tripletex_call_log": tripletex.call_log,
                },
            )

            try:
                runs_dir = Path(__file__).parent.parent.parent / "runs"
                palace = MemoryPalace(runs_dir)
                palace.add_trace(Path(trace.path))
                try:
                    compile_trace(Path(trace.path))
                except Exception:
                    pass
            except Exception:
                pass

            try:
                fuzzer = MutationFuzzer()
                mutations = fuzzer.analyze_near_miss(Path(trace.path))
                if mutations:
                    trace.write(
                        "mutation_candidates",
                        {
                            "count": len(mutations),
                            "types": [m.mutation_type for m in mutations],
                            "descriptions": [m.description for m in mutations[:3]],
                        },
                    )
            except Exception:
                pass

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
            trace.close()
            await openrouter.close()
            await tripletex.close()
            _refresh_overview_json()

    async def _complete_json_with_fallback(
        self,
        openrouter: OpenRouterClient,
        *,
        messages: list[dict[str, Any]],
        model_chain: list[str],
    ) -> dict[str, Any]:
        """Call complete_json trying each model in chain on retryable failures.

        If a model produces unparseable JSON (likely truncated output), retry
        the same model once with a higher max_tokens budget before falling back
        to the next model in the chain.
        """
        last_exc: Exception | None = None
        for model in model_chain:
            try:
                return await openrouter.complete_json(
                    messages=messages,
                    model_override=model,
                )
            except OpenRouterError as exc:
                is_parse_error = "Failed to parse JSON from LLM output" in str(exc)
                if is_parse_error:
                    # Likely truncated — retry same model with higher token budget
                    logger.warning(
                        "Model %s JSON parse failed (likely truncated), retrying with higher max_tokens",
                        model,
                    )
                    try:
                        return await openrouter.complete_json(
                            messages=messages,
                            model_override=model,
                            max_tokens=4096,
                        )
                    except Exception as retry_exc:
                        last_exc = retry_exc
                        if (
                            _should_fallback_on_error(retry_exc)
                            and model != model_chain[-1]
                        ):
                            logger.warning(
                                "Model %s retry also failed (%s), falling back",
                                model,
                                retry_exc,
                            )
                            continue
                        raise
                last_exc = exc
                if _should_fallback_on_error(exc) and model != model_chain[-1]:
                    logger.warning(
                        "Model %s failed (%s), falling back to next in chain",
                        model,
                        exc,
                    )
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                if _should_fallback_on_error(exc) and model != model_chain[-1]:
                    logger.warning(
                        "Model %s failed (%s), falling back to next in chain",
                        model,
                        exc,
                    )
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise OpenRouterError("Empty model chain")

    async def _chat_completion_with_fallback(
        self,
        openrouter: OpenRouterClient,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        model_chain: list[str],
        enable_thinking: bool = False,
    ) -> dict[str, Any]:
        last_exc: Exception | None = None
        for model in model_chain:
            try:
                return await openrouter.chat_completion(
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    model_override=model,
                    enable_thinking=enable_thinking and _is_anthropic_model(model),
                )
            except Exception as exc:
                last_exc = exc
                if _should_fallback_on_error(exc) and model != model_chain[-1]:
                    logger.warning(
                        "Model %s failed (%s), falling back to next in chain",
                        model,
                        exc,
                    )
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise OpenRouterError("Empty model chain")

    async def _plan(
        self,
        openrouter: OpenRouterClient,
        request: SolveRequest,
        attachment_summaries: list[Any],
        *,
        model_chain: list[str] | None = None,
    ) -> PlannerOutput:
        plan_prompt = {
            "prompt": request.prompt,
            "attachments": [summary.model_dump() for summary in attachment_summaries],
        }
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(plan_prompt, ensure_ascii=False),
            },
        ]
        result = await self._complete_json_with_fallback(
            openrouter,
            messages=messages,
            model_chain=model_chain or [],
        )
        try:
            return PlannerOutput.model_validate(result)
        except ValidationError as exc:
            raise OpenRouterError(f"Planner output validation failed: {exc}") from exc

    async def _semantic_enforce(
        self,
        openrouter: OpenRouterClient,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        planner: PlannerOutput,
        request_prompt: str,
        execution_history: list[str],
    ) -> dict[str, Any] | None:
        if tool_name != "tripletex_request":
            return None
        method = (arguments.get("method") or "").upper()
        if method not in ("POST", "PUT"):
            return None

        path = arguments.get("path", "")
        body = arguments.get("json_body") or {}
        params = arguments.get("params") or {}
        body_str = json.dumps(body, ensure_ascii=False)
        params_str = json.dumps(params, ensure_ascii=False)
        combined = body_str + params_str

        has_amounts = any(
            kw in combined.lower()
            for kw in (
                "amount",
                "price",
                "gross",
                "cost",
                "rate",
                "fixedprice",
                "paidamount",
                "count",
            )
        )
        is_payment = "/:payment" in path
        is_voucher = "/ledger/voucher" in path and path.rstrip("/").endswith("/voucher")
        is_orderline = "/orderline" in path
        is_invoice = path.rstrip("/").endswith("/invoice") and method == "POST"

        if not (has_amounts or is_payment or is_voucher or is_orderline or is_invoice):
            return None

        body_summary = body_str[:1500]
        params_summary = params_str[:500]

        line_items_json = json.dumps(
            [li.model_dump() for li in planner.line_items], ensure_ascii=False
        )[:800]

        prompt = (
            "You are a MATH-ONLY checker for Tripletex API calls. "
            "You ONLY check if NUMERIC VALUES are correct. "
            "Do NOT check call ordering, entity IDs, prerequisites, or sequencing.\n\n"
            f"TASK PROMPT: {request_prompt}\n\n"
            f"LINE ITEMS FROM PROMPT: {line_items_json}\n\n"
            f"PROPOSED CALL:\n"
            f"  {method} {path}\n"
            f"  Body: {body_summary}\n"
            f"  Params: {params_summary}\n\n"
            "CHECK ONLY:\n"
            "1. Do the AMOUNTS/PRICES match what the task prompt specifies?\n"
            "2. Is the VAT rate correct? (25%=id:3, 15%=id:5, 0%=id:6. 'sin IVA'/'uten MVA'/'ohne MwSt'=0%)\n"
            "3. For payments: does paidAmount equal the invoice total INCLUDING VAT?\n"
            "4. For percentages: is the math correct? (e.g., 75% of 342600 = 256950)\n"
            "5. For orderlines: does unitPriceExcludingVatCurrency match the line item price?\n\n"
            "DO NOT reject for: missing fields, wrong IDs, call ordering, prerequisites, or sequencing.\n"
            "ASSUME all entity IDs are correct — they come from prior API responses.\n\n"
            'Return JSON: {"allowed": true} or {"allowed": false, "reason": "...", "suggestion": "..."}\n'
            "Only reject for CLEAR NUMERIC ERRORS. When in doubt, ALLOW."
        )

        try:
            result = await openrouter.complete_json(
                messages=[
                    {
                        "role": "system",
                        "content": "You check ONLY math and amounts. Return JSON. Never reject for sequencing or IDs.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                model_override=self._settings.enforcer_model,
            )
            if not result.get("allowed", True):
                return {
                    "rejected": True,
                    "reason": result.get("reason", "Semantic check failed"),
                    "suggestion": result.get(
                        "suggestion",
                        "Review the values against the task prompt.",
                    ),
                }
        except Exception as exc:
            logger.debug("Semantic enforcer failed (non-fatal): %s", exc)
        return None

    async def _ask_api_advisor(
        self,
        openrouter: OpenRouterClient,
        question: str,
        planner: PlannerOutput,
        request_prompt: str,
    ) -> dict[str, Any]:
        search_terms = question.lower().split()[:5]
        relevant_endpoints: list[dict[str, Any]] = []
        seen: set[str] = set()
        for term in search_terms:
            for ep in self._spec_index.search_endpoints(term, limit=3):
                key = f"{ep.get('method', '')} {ep.get('path', '')}"
                if key not in seen:
                    seen.add(key)
                    relevant_endpoints.append(ep)

        schemas: dict[str, Any] = {}
        for ep in relevant_endpoints[:10]:
            for schema_key in ("requestSchema", "responseSchema"):
                name = ep.get(schema_key)
                if isinstance(name, str) and name and name not in schemas:
                    try:
                        schemas[name] = self._spec_index.get_schema(
                            name, max_properties=40
                        )
                    except KeyError:
                        pass

        endpoints_text = json.dumps(
            [
                {
                    "method": e.get("method"),
                    "path": e.get("path"),
                    "summary": e.get("summary"),
                    "requestSchema": e.get("requestSchema"),
                }
                for e in relevant_endpoints[:10]
            ],
            ensure_ascii=False,
        )
        schemas_text = json.dumps(schemas, ensure_ascii=False)[:8000]

        from tripletex_agent.executor_knowledge import select_field_rules

        endpoint_keys = [
            f"{e.get('method', '')} {e.get('path', '')}" for e in relevant_endpoints
        ]
        field_rules = select_field_rules(endpoint_keys)
        rules_text = json.dumps(field_rules, ensure_ascii=False)[:3000]

        advisor_prompt = (
            "You are a Tripletex API expert advisor. A developer needs help making the right API call.\n\n"
            f"TASK CONTEXT: {request_prompt}\n\n"
            f"DEVELOPER QUESTION: {question}\n\n"
            f"RELEVANT ENDPOINTS:\n{endpoints_text}\n\n"
            f"SCHEMAS:\n{schemas_text}\n\n"
            f"FIELD RULES (known gotchas):\n{rules_text}\n\n"
            "Give a SPECIFIC, ACTIONABLE recommendation:\n"
            "1. Which endpoint to use (method + path)\n"
            "2. The EXACT body or params structure with correct field names\n"
            "3. Any required fields that are easy to miss\n"
            "4. Common pitfalls for this specific endpoint\n"
            "Be concise and practical — the developer will use your answer directly."
        )

        try:
            result = await openrouter.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Tripletex REST API expert. Give precise, actionable answers.",
                    },
                    {"role": "user", "content": advisor_prompt},
                ],
                max_tokens=2000,
                model_override=self._settings.enforcer_model,
            )
            return {
                "ok": True,
                "advisor_response": result.get("content", ""),
                "endpoints_found": len(relevant_endpoints),
                "schemas_provided": list(schemas.keys()),
            }
        except Exception as exc:
            return {"ok": False, "error": f"Advisor call failed: {exc}"}

    def _resolve_planned_endpoints(
        self,
        planner: PlannerOutput,
    ) -> list[dict[str, Any]]:
        endpoint_keys: list[str] = get_endpoint_chain(planner.task_type)

        if planner.task_type == "create_invoice":
            prompt_lower = " ".join(
                [s.lower() for s in planner.ordered_steps] + [planner.goal.lower()]
            )
            timesheet_keywords = [
                "timer",
                "hours",
                "horas",
                "stunden",
                "heures",
                "timesheet",
                "registrer",
                "registe",
                "aktivitet",
                "activity",
            ]
            if any(kw in prompt_lower for kw in timesheet_keywords):
                ts_chain = get_endpoint_chain("create_invoice_timesheet")
                if ts_chain:
                    for ts_key in ts_chain:
                        if ts_key not in endpoint_keys:
                            endpoint_keys.append(ts_key)

        for step in planner.ordered_steps:
            match = re.search(r"(GET|POST|PUT|DELETE)\s+(/\S+)", step)
            if match:
                key = f"{match.group(1)} {match.group(2)}"
                if key not in endpoint_keys:
                    endpoint_keys.append(key)

        resolved: list[dict[str, Any]] = []
        seen: set[str] = set()
        for key in endpoint_keys:
            if key in seen:
                continue
            seen.add(key)
            parts = key.split(" ", 1)
            if len(parts) != 2:
                continue
            method, path = parts
            try:
                endpoint = self._spec_index.get_endpoint(method, path)
                resolved.append(endpoint)
            except KeyError:
                resolved.append(
                    {"method": method, "path": path, "summary": "Not found in spec"}
                )
        return resolved

    def _build_prefetched_schemas(
        self,
        planned_endpoints: list[dict[str, Any]],
    ) -> dict[str, Any]:
        schemas: dict[str, Any] = {}
        for ep in planned_endpoints:
            for schema_key in ("requestSchema", "responseSchema"):
                schema_name = ep.get(schema_key)
                if not isinstance(schema_name, str) or not schema_name:
                    continue
                if schema_name in schemas:
                    continue
                try:
                    schemas[schema_name] = self._spec_index.get_schema(
                        schema_name,
                        max_properties=60,
                    )
                except KeyError:
                    pass
        return schemas

    def _build_execution_brief(
        self,
        planner: PlannerOutput,
        request_prompt: str,
    ) -> dict[str, Any]:
        today = date.today()
        computed_dates = {
            "today_iso": today.isoformat(),
            "due_date_iso": (today + timedelta(days=30)).isoformat(),
        }
        computed_dates = {
            **computed_dates,
            **planner.extracted_dates,
        }
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

        planned_endpoints = self._resolve_planned_endpoints(planner)
        prefetched_schemas = self._build_prefetched_schemas(planned_endpoints)

        endpoint_keys = [
            f"{ep.get('method', '')} {ep.get('path', '')}" for ep in planned_endpoints
        ]
        field_rules = select_field_rules(endpoint_keys)
        trace_example = select_successful_trace(planner.task_type)

        candidate_endpoints = self._grounded_candidate_endpoints(
            planner=planner,
            request_prompt=request_prompt,
        )

        return {
            "goal": planner.goal,
            "task_type": planner.task_type,
            "primary_resource": primary_resource,
            "linked_resources": linked_resources,
            "candidate_queries": [primary_resource, *(linked_resources[:2])],
            "candidate_endpoints": candidate_endpoints,
            "preferred_link_fields": preferred_link_fields,
            "requested_field_names": requested_field_names,
            "explicit_prompt_values": explicit_prompt_values,
            "suggested_first_action": planner.suggested_first_action,
            "success_checks": planner.success_checks[:5],
            "risk_notes": planner.risk_notes[:5],
            "planned_endpoints": [
                {
                    "method": ep.get("method"),
                    "path": ep.get("path"),
                    "summary": ep.get("summary"),
                    "requestSchema": ep.get("requestSchema"),
                    "responseSchema": ep.get("responseSchema"),
                }
                for ep in planned_endpoints
            ],
            "prefetched_schemas": prefetched_schemas,
            "field_rules": field_rules,
            "successful_trace_example": trace_example,
            "computed_dates": computed_dates,
            "entities": [entity.model_dump() for entity in planner.entities],
            "line_items": [line_item.model_dump() for line_item in planner.line_items],
            "actions": planner.actions,
            "ordered_steps": planner.ordered_steps,
            "domain_knowledge": get_domain_knowledge(),
            "working_rules": [
                "OBEY field_rules BEFORE your first write — they prevent known 422 errors.",
                "Use prefetched_schemas to know exact field names and types — do NOT call inspect_tripletex_endpoint unless prefetched schemas are missing.",
                "Follow the successful_trace_example as your primary execution template when available.",
                "Prefer existing entities when conflicts or duplicates are plausible.",
                "Keep actions anchored to the primary target resource.",
                "If blocked by external or company-level setup, stop and report it.",
                "DOMAIN KNOWLEDGE: execution_brief.domain_knowledge has standard Norwegian accounts, VAT types, travel expense categories, and salary prerequisite chain. Use this instead of API lookups.",
                "MISSING ACCOUNTS: If an account number from the prompt is NOT in pre_resolved_accounts or returned None, CREATE it with POST /ledger/account before using it.",
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

        requested_field_names = _infer_requested_schema_fields(request_prompt)

        for query in grounded_queries:
            for endpoint in self._spec_index.search_endpoints(query, limit=4):
                endpoint_key = (
                    str(endpoint.get("method") or ""),
                    str(endpoint.get("path") or ""),
                )
                if endpoint_key in seen_endpoints:
                    continue

                request_schema = endpoint.get("requestSchema")
                requested_paths: list[str] = []
                if request_schema:
                    requested_paths = _find_requested_field_paths(
                        self._spec_index,
                        request_schema,
                        requested_field_names,
                    )

                candidate_endpoints.append(
                    {
                        "query": query,
                        "method": endpoint.get("method"),
                        "path": endpoint.get("path"),
                        "summary": endpoint.get("summary"),
                        "requestSchema": request_schema,
                        "responseSchema": endpoint.get("responseSchema"),
                        "requested_field_paths": requested_paths,
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
        *,
        model_chain: list[str] | None = None,
    ) -> None:
        tools = build_tool_definitions(
            planner_task_type=planner.task_type,
            request_prompt=request.prompt,
        )
        focused_prompt = build_executor_system_prompt(planner.task_type)
        is_salary_task = planner.task_type in ("create_voucher",) and any(
            kw in request.prompt.lower()
            for kw in (
                "salary",
                "lønn",
                "gehalt",
                "nómina",
                "paie",
                "folha",
                "salário",
                "payroll",
                "lønns",
            )
        )
        if is_salary_task:
            salary_cheat_sheet = """
## SALARY/PAYROLL TASK — SPECIAL INSTRUCTIONS
This is a SALARY task. Do NOT use POST /ledger/voucher — it scores 0 points.

USE POST /salary/transaction — the ONLY correct endpoint for salary processing.

PREREQUISITE CHAIN (follow in order, skip steps where entities already exist):
1. Employee MUST have dateOfBirth — if missing, PUT /employee/{id} with dateOfBirth="1990-01-15"
2. Division MUST exist — GET /division, if empty: GET /municipality/query?query=Oslo, then POST /division
3. Employment MUST exist — POST /employee/employment with division, startDate="2026-01-01"
4. Employment details MUST exist — POST /employee/employment/details with occupationCode, annualSalary

THEN create the salary transaction:
POST /salary/transaction with body:
{date: "YYYY-MM-DD", year: YYYY, month: M, payslips: [{employee: {id: emp_id}, specifications: [{salaryType: {id: type_id}, rate: amount, count: 1}]}]}

CRITICAL: Use "rate" and "count" in specifications, NOT "amount".
GET /salary/type to find "Fastlønn" (base salary) and "Bonus" type IDs.
"""
            focused_prompt = focused_prompt + salary_cheat_sheet
        execution_brief = self._build_execution_brief(
            planner=planner,
            request_prompt=request.prompt,
        )
        try:
            runs_dir = Path(__file__).parent.parent.parent / "runs"
            palace = MemoryPalace(runs_dir)
            recipe = palace.find_recipe(planner.task_type, request.prompt)
            if recipe:
                execution_brief["memory_palace_recipe"] = palace.format_as_example(
                    recipe
                )
                trace.write(
                    "memory_palace_hit",
                    {
                        "task_type": recipe.task_type,
                        "source": recipe.source_file,
                        "calls": recipe.total_calls,
                    },
                )
        except Exception as e:
            logger.debug("Memory palace lookup failed: %s", e)
        if execution_state.pre_resolved_accounts:
            execution_brief["pre_resolved_accounts"] = {
                str(k): v for k, v in execution_state.pre_resolved_accounts.items()
            }
        if is_salary_task:
            execution_brief["planned_endpoints"] = [
                ep
                for ep in execution_brief.get("planned_endpoints", [])
                if "/ledger/voucher" not in ep.get("path", "")
            ]
            execution_brief["planned_endpoints"].extend(
                [
                    {
                        "method": "GET",
                        "path": "/salary/type",
                        "summary": "List salary/wage types (find Fastlønn and Bonus IDs)",
                    },
                    {
                        "method": "GET",
                        "path": "/division",
                        "summary": "Check if division exists (required for employment)",
                    },
                    {
                        "method": "POST",
                        "path": "/division",
                        "summary": "Create division (prerequisite for employment)",
                    },
                    {
                        "method": "POST",
                        "path": "/employee/employment",
                        "summary": "Create employment record (prerequisite for salary)",
                    },
                    {
                        "method": "POST",
                        "path": "/employee/employment/details",
                        "summary": "Create employment details (prerequisite for salary)",
                    },
                    {
                        "method": "POST",
                        "path": "/salary/transaction",
                        "summary": "Create salary transaction with payslips and specifications",
                    },
                ]
            )
        messages: list[dict[str, Any]] = [{"role": "system", "content": focused_prompt}]
        middleware = ExecutionMiddleware()

        for account_number, account_id in execution_state.pre_resolved_accounts.items():
            if account_id is not None:
                middleware.registry.register(f"account_{account_number}", account_id)

        sandbox_context = await middleware.prefetch_sandbox_context(
            tripletex, task_type=planner.task_type
        )
        if sandbox_context:
            execution_brief["sandbox_context"] = middleware.scrub_ids_for_llm(
                sandbox_context
            )
        bank_info = sandbox_context.get("bank_account")
        execution_brief["bank_account_ready"] = (
            isinstance(bank_info, dict) and bank_info.get("isBankAccount") is True
        )

        if execution_state.sandbox_discovery:
            for entity_type, entity_data in execution_state.sandbox_discovery.items():
                if isinstance(entity_data, dict) and entity_data.get("id"):
                    name = (
                        entity_data.get("key_fields", {}).get("name")
                        or entity_data.get("key_fields", {}).get("firstName")
                        or entity_type
                    )
                    import re as _re2

                    safe_name = _re2.sub(r"[^a-zA-Z0-9æøåÆØÅ_-]", "_", str(name))[
                        :30
                    ].strip("_")
                    middleware.registry.register(
                        f"{entity_type}_{safe_name}", entity_data["id"]
                    )
                    if entity_data.get("key_fields", {}).get("version") is not None:
                        middleware.registry.register(
                            f"{entity_type}_{safe_name}_version",
                            entity_data["key_fields"]["version"],
                        )
            execution_brief["sandbox_discovery"] = middleware.scrub_ids_for_llm(
                execution_state.sandbox_discovery
            )

        execution_brief["vat_types"] = {
            "vat_25_id": 3,
            "vat_15_id": 5,
            "vat_0_id": 6,
            "note": "These are standard Norwegian vatType IDs. Use id=3 for 25%, id=5 for 15%, id=6 for 0% exempt.",
        }
        execution_state.middleware = middleware
        trace.write(
            "execution_brief",
            {
                "planned_endpoints": execution_brief.get("planned_endpoints", []),
                "prefetched_schema_names": list(
                    execution_brief.get("prefetched_schemas", {}).keys()
                ),
                "field_rules_count": len(execution_brief.get("field_rules", [])),
                "field_rules": execution_brief.get("field_rules", []),
                "has_trace_example": execution_brief.get("successful_trace_example")
                is not None,
                "trace_example_task": (
                    execution_brief.get("successful_trace_example") or {}
                ).get("description"),
                "computed_dates": execution_brief.get("computed_dates", {}),
            },
        )
        computed_dates = execution_brief.get("computed_dates") or {}
        today_iso = computed_dates.get("today_iso")
        due_date_iso = computed_dates.get("due_date_iso")
        user_parts = [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "task_prompt": request.prompt,
                        "planner": planner.model_dump(),
                        "execution_brief": execution_brief,
                        "instructions": [
                            (
                                "SCHEMA-FIRST: execution_brief contains prefetched_schemas with "
                                "FULL field definitions for all planned endpoints. Read these "
                                "BEFORE calling any endpoint. Do NOT use inspect_tripletex_endpoint "
                                "or search_tripletex_api unless the schema is genuinely missing."
                            ),
                            (
                                "FIELD RULES: execution_brief.field_rules contains historical "
                                "guidance for each endpoint. Follow these on your FIRST attempt, "
                                "but if a rule leads to a 422 error, IGNORE that rule and use the "
                                "prefetched_schemas as the authoritative reference instead. "
                                "The API schema always takes priority over field rules."
                            ),
                            (
                                "TRACE TEMPLATE: If execution_brief.successful_trace_example "
                                "is present, follow that exact call sequence as your template. "
                                "It shows the proven minimal-call path for this task type."
                            ),
                            "Use the fewest Tripletex API calls possible — every call counts against your efficiency score.",
                            "ZERO 4xx errors is the target. Get it right on the FIRST attempt.",
                            (
                                "TRACK SUCCESSES: After each successful POST, note the resource_id. "
                                "Never re-create an entity you already created successfully."
                            ),
                            (
                                "If a validation error suggests a correct field name, "
                                "USE THAT EXACT SUGGESTION on your next attempt."
                            ),
                            "Do not retry identical failed mutations unchanged.",
                            "Return completion JSON only when the task is actually done.",
                            (
                                "DETERMINISTIC DATES: "
                                f"today={today_iso}, due_date={due_date_iso}. "
                                "Use these for any date fields not explicitly specified "
                                "in the prompt."
                            ),
                            (
                                "STRUCTURED DATA: The execution_brief contains pre-extracted "
                                "entities, line_items, and actions from the prompt. Use "
                                "these instead of re-interpreting the multilingual prompt."
                            ),
                            (
                                "PRODUCT RULE: If line_items have product_number set, you MUST "
                                "create Product entities with POST /product BEFORE creating "
                                'orderlines. Each orderline must reference product={"id": '
                                "product_id}."
                            ),
                            (
                                "EMAIL RULE: When creating customer or supplier with an email, "
                                "ALWAYS set BOTH email AND invoiceEmail to the same value."
                            ),
                            (
                                "PAYMENT RULE: Payment registration is PUT /invoice/{id}/:payment "
                                "(NOT POST). Use query params: paymentDate, paymentTypeId, "
                                "paidAmount."
                            ),
                            (
                                "VOUCHER RULE: For POST /ledger/voucher, ALWAYS set BOTH "
                                "amountGross AND amountGrossCurrency to the same value on "
                                "every posting. Never use row=0. ALWAYS set row=1,2,3... explicitly."
                            ),
                            (
                                "ADMIN RULE: If the prompt mentions admin, administrator, "
                                "rolle, role, tilgang, accès, Zugang, privilegios, or similar "
                                "— you MUST call grant_employee_entitlements with template='ALL_PRIVILEGES' "
                                "after creating/finding the employee. This is worth 5 points."
                            ),
                            (
                                "UPDATE RULE: For update tasks (update_employee, update_customer, etc.), "
                                "GET the entity first with fields=* to get its id and version. "
                                "Then PUT with id, version, and ONLY the changed fields."
                            ),
                        ],
                    },
                    ensure_ascii=False,
                ),
            }
        ]
        user_parts.extend(attachment_parts)
        messages.append({"role": "user", "content": user_parts})

        executor_model_chain = model_chain or [self._settings.openrouter_model]
        max_steps = max(self._settings.agent_max_steps, 28)
        task_tier = _classify_task_tier(planner.task_type)
        if task_tier == 3:
            max_steps = max(max_steps, 50)

        time_budget_seconds = 280.0
        for step_idx in range(max_steps):
            elapsed = time.monotonic() - execution_state.start_time
            remaining = time_budget_seconds - elapsed
            if remaining <= 15:
                trace.write(
                    "time_budget_exceeded", {"elapsed": elapsed, "remaining": remaining}
                )
                logger.warning(
                    "Time budget exhausted (%.0fs elapsed). Forcing completion.",
                    elapsed,
                )
                return
            steps_remaining = max_steps - step_idx
            if execution_state.consecutive_proxy_errors >= 2:
                trace.write(
                    "circuit_breaker_tripped",
                    {
                        "consecutive_proxy_errors": execution_state.consecutive_proxy_errors,
                        "step": step_idx,
                    },
                )
                logger.warning(
                    "Circuit breaker: %d consecutive proxy token errors. Stopping.",
                    execution_state.consecutive_proxy_errors,
                )
                return
            if step_idx >= 4:
                _compress_old_tool_results(messages, keep_recent=6)
            budget_warning = _build_budget_warning(
                steps_remaining=steps_remaining,
                time_remaining=remaining,
            )
            if budget_warning:
                messages.append({"role": "user", "content": budget_warning})
            num_steps = len(getattr(planner, "ordered_steps", None) or [])
            use_thinking = num_steps >= 5 or task_tier >= 2
            llm_retries = 2 if step_idx == 0 else 1
            for llm_attempt in range(llm_retries):
                try:
                    message = await self._chat_completion_with_fallback(
                        openrouter,
                        messages=messages,
                        tools=tools,
                        max_tokens=64000 if use_thinking else 16000,
                        model_chain=executor_model_chain,
                        enable_thinking=use_thinking,
                    )
                    break
                except Exception as llm_exc:
                    if llm_attempt < llm_retries - 1:
                        logger.warning(
                            "Executor LLM call failed (attempt %d/%d), retrying in 2s: %s",
                            llm_attempt + 1,
                            llm_retries,
                            llm_exc,
                        )
                        trace.write(
                            "executor_llm_retry",
                            {
                                "attempt": llm_attempt + 1,
                                "error": str(llm_exc)[:300],
                                "step": step_idx,
                            },
                        )
                        await asyncio.sleep(2)
                    else:
                        raise
            tool_calls = message.get("tool_calls") or []
            assistant_content = message.get("content") or ""
            thinking_content = message.get("thinking") or ""

            if thinking_content:
                trace.write("thinking", {"text": thinking_content[:3000]})
            if assistant_content and assistant_content.strip():
                trace.write("assistant_reasoning", {"text": assistant_content[:2000]})

            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": tool_calls,
                    }
                )
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    arguments = json.loads(
                        tool_call["function"].get("arguments") or "{}"
                    )
                    tool_call_key = _build_tool_call_key(tool_name, arguments)

                    if tool_name == "ask_api_advisor":
                        trace.write(
                            "api_advisor_query",
                            {"question": arguments.get("question", "")},
                        )
                        advisor_result = await self._ask_api_advisor(
                            openrouter,
                            question=arguments.get("question", ""),
                            planner=planner,
                            request_prompt=request.prompt,
                        )
                        trace.write(
                            "api_advisor_response",
                            {
                                "endpoints_found": advisor_result.get(
                                    "endpoints_found", 0
                                ),
                                "response_preview": (
                                    advisor_result.get("advisor_response") or ""
                                )[:200],
                            },
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": json.dumps(
                                    advisor_result, ensure_ascii=False
                                ),
                            }
                        )
                        continue

                    if tool_name == "override_enforcer":
                        execution_state.enforcer_override_count += 1
                        override_reason = arguments.get("reason", "")
                        trace.write(
                            "enforcer_override",
                            {
                                "reason": override_reason,
                                "count": execution_state.enforcer_override_count,
                            },
                        )
                        if execution_state.enforcer_override_count > 3:
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "content": json.dumps(
                                        {
                                            "ok": False,
                                            "message": (
                                                f"Override DENIED — you have used {execution_state.enforcer_override_count} overrides. "
                                                "The enforcer is protecting you from known API failures. "
                                                f"Follow the suggestion: {execution_state.last_enforcer_suggestion}"
                                            ),
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            )
                            continue
                        execution_state.enforcer_override_active = True
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": json.dumps(
                                    {
                                        "ok": True,
                                        "message": f"Override accepted ({execution_state.enforcer_override_count}/3). Your next tool call will bypass the enforcer.",
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                        )
                        continue

                    preflight = _preflight_enforce(
                        tool_name, arguments, execution_state
                    )
                    if (
                        preflight is not None
                        and not execution_state.enforcer_override_active
                    ):
                        execution_state.last_enforcer_suggestion = preflight[
                            "suggestion"
                        ]
                        trace.write(
                            "enforcer_rejected",
                            {
                                "tool_name": tool_name,
                                "arguments": arguments,
                                "reason": preflight["reason"],
                                "suggestion": preflight["suggestion"],
                            },
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": json.dumps(
                                    {
                                        "ok": False,
                                        "enforcer_rejected": True,
                                        "reason": preflight["reason"],
                                        "suggestion": preflight["suggestion"],
                                        "hint": "Fix the issue and retry, or call override_enforcer if you believe this check is wrong.",
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                        )
                        continue
                    should_run_semantic = not execution_state.enforcer_override_active
                    semantic: dict[str, Any] | None = None
                    if should_run_semantic:
                        semantic = await self._semantic_enforce(
                            openrouter,
                            tool_name=tool_name,
                            arguments=arguments,
                            planner=planner,
                            request_prompt=request.prompt,
                            execution_history=execution_state.recent_tool_call_keys[
                                -8:
                            ],
                        )
                        if semantic is not None:
                            trace.write(
                                "semantic_enforcer_rejected",
                                {
                                    "tool_name": tool_name,
                                    "arguments": arguments,
                                    "reason": semantic["reason"],
                                    "suggestion": semantic["suggestion"],
                                },
                            )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "content": json.dumps(
                                        {
                                            "ok": False,
                                            "semantic_enforcer_rejected": True,
                                            "reason": semantic["reason"],
                                            "suggestion": semantic["suggestion"],
                                            "hint": "Fix the values and retry, or call override_enforcer if you believe this check is wrong.",
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            )
                            continue
                    execution_state.enforcer_override_active = False

                    if tool_name == "tripletex_request":
                        method_str = (arguments.get("method") or "?").upper()
                        path_str = arguments.get("path", "?")
                        trace.write(
                            "enforcer_passed",
                            {
                                "tool_name": tool_name,
                                "call": f"{method_str} {path_str}",
                            },
                        )

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
                        trace,
                    )
                    trace.write(
                        "tool_result",
                        {
                            "tool_name": tool_name,
                            "result": {
                                "ok": tool_result.get("ok", True),
                                "status_code": tool_result.get("status_code"),
                                "summary": tool_result.get("summary", "")[:300]
                                if isinstance(tool_result.get("summary"), str)
                                else "",
                                "error": tool_result.get("error", "")[:500]
                                if isinstance(tool_result.get("error"), str)
                                else str(tool_result.get("error", ""))[:500],
                                "resource_id": tool_result.get("resource_id"),
                                "validation_summary": tool_result.get(
                                    "validation_summary"
                                ),
                            },
                        },
                    )
                    if (
                        not tool_result.get("ok")
                        and tool_result.get("status_code") in (409, 422)
                        and execution_state.last_enforcer_suggestion
                    ):
                        tool_result["enforcer_reminder"] = (
                            f"The enforcer previously warned about this. "
                            f"Follow its suggestion: {execution_state.last_enforcer_suggestion}"
                        )
                    if (
                        not tool_result.get("ok")
                        and tool_result.get("status_code") == 422
                    ):
                        tool_result["schema_priority_hint"] = (
                            "If a field_rule guided this call, that rule may be stale. "
                            "Check the prefetched_schemas for the correct field names, "
                            "types, and required fields. The API schema is authoritative."
                        )

                    advisor_endpoint_key = f"{(arguments.get('method') or 'GET').upper()} {arguments.get('path', '')}"
                    if (
                        tool_name == "tripletex_request"
                        and not tool_result.get("ok")
                        and tool_result.get("status_code") == 422
                        and advisor_endpoint_key
                        not in execution_state.advisor_422_endpoints
                    ):
                        execution_state.advisor_422_endpoints.add(advisor_endpoint_key)
                        validation_summary = tool_result.get("validation_summary") or []
                        validation_text = "; ".join(
                            f"{item.get('field', '')}: {item.get('message', '')}"
                            for item in validation_summary
                            if isinstance(item, dict)
                        )
                        if not validation_text:
                            validation_text = str(tool_result.get("error") or "")[:800]
                        advisor_question = (
                            f"I attempted {arguments.get('method', 'GET')} {arguments.get('path', '')} "
                            f"for Tripletex task_type={planner.task_type}. Prompt: {request.prompt}. "
                            f"The request failed with 422 validation errors: {validation_text}. "
                            f"Arguments were: {json.dumps(arguments, ensure_ascii=False)}. "
                            "Recommend the exact next Tripletex call or payload correction to make, "
                            "using the minimum-change fix."
                        )
                        trace.write(
                            "api_advisor_query",
                            {"question": advisor_question[:2000], "automatic": True},
                        )
                        advisor_result = await self._ask_api_advisor(
                            openrouter,
                            question=advisor_question,
                            planner=planner,
                            request_prompt=request.prompt,
                        )
                        trace.write(
                            "api_advisor_response",
                            {
                                "endpoints_found": advisor_result.get(
                                    "endpoints_found", 0
                                ),
                                "response_preview": (
                                    advisor_result.get("advisor_response")
                                    or advisor_result.get("error")
                                    or ""
                                )[:200],
                                "automatic": True,
                            },
                        )
                        if advisor_result.get("ok"):
                            tool_result["advisor_guidance"] = (
                                advisor_result.get("advisor_response") or ""
                            )[:2000]
                            tool_result["advisor_endpoints_found"] = advisor_result.get(
                                "endpoints_found", 0
                            )
                            existing_hint = str(tool_result.get("hint") or "").strip()
                            advisor_hint = (
                                "Review advisor_guidance before retrying. Make the minimum correction "
                                "it recommends instead of guessing."
                            )
                            tool_result["hint"] = (
                                f"{existing_hint} {advisor_hint}".strip()
                                if existing_hint
                                else advisor_hint
                            )

                    blocking_issue = _detect_blocking_issue(
                        planner=planner,
                        request_prompt=request.prompt,
                        tool_name=tool_name,
                        arguments=arguments,
                        tool_result=tool_result,
                    )
                    if blocking_issue is not None:
                        trace.write("blocked_warning", blocking_issue)
                        tool_result["warning"] = blocking_issue.get("message")
                        tool_result["hint"] = (
                            "This may indicate an external prerequisite. "
                            "Try completing what you can or find an alternative approach."
                        )
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
                            "error": drift_issue.get("message"),
                            "hint": drift_issue.get("hint"),
                            "grounded_primary_resource": drift_issue.get(
                                "grounded_primary_resource"
                            ),
                            "grounded_linked_resources": drift_issue.get(
                                "grounded_linked_resources"
                            ),
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
                    if (
                        tool_name == "tripletex_request"
                        and not execution_state.reclassification_done
                        and (arguments.get("method") or "").upper() in ("POST", "PUT")
                        and tool_result.get("ok")
                    ):
                        call_path = (arguments.get("path") or "").rstrip("/")
                        planned_paths = {
                            ep.get("path", "").rstrip("/")
                            for ep in execution_brief.get("planned_endpoints", [])
                        }
                        if call_path and call_path not in planned_paths:
                            execution_state.off_plan_api_calls += 1
                        if execution_state.off_plan_api_calls >= 2:
                            alt_type = getattr(planner, "alternative_task_type", None)
                            if alt_type:
                                alt_rules = select_field_rules(
                                    [f"POST {p}" for p in get_endpoint_chain(alt_type)]
                                )
                                alt_playbook = EXECUTOR_PLAYBOOKS.get(alt_type, "")
                                reclass_msg = (
                                    f"[RECLASSIFICATION] Your API calls don't match the planned "
                                    f"task_type '{planner.task_type}'. The alternative classification "
                                    f"'{alt_type}' may be more appropriate. "
                                )
                                if alt_playbook:
                                    reclass_msg += f"Playbook for {alt_type}: {alt_playbook[:500]} "
                                if alt_rules:
                                    reclass_msg += f"Field rules: {json.dumps(alt_rules[:5], ensure_ascii=False)}"
                                tool_result["reclassification_hint"] = reclass_msg
                                trace.write(
                                    "reclassification",
                                    {
                                        "original_type": planner.task_type,
                                        "suggested_type": alt_type,
                                        "off_plan_calls": execution_state.off_plan_api_calls,
                                    },
                                )
                                execution_state.reclassification_done = True

                    compacted = _compact_tool_result_for_model(tool_result)
                    if execution_state.middleware:
                        compacted = execution_state.middleware.scrub_ids_for_llm(
                            compacted
                        )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": json.dumps(
                                compacted,
                                ensure_ascii=False,
                            ),
                        }
                    )
                continue

            content = message.get("content") or ""
            if isinstance(content, list):
                content = "\n".join(str(part) for part in content)
            final_payload = _extract_completion_json(content)
            if final_payload is not None and final_payload.get("status") == "completed":
                trace.write("final_payload", final_payload)
                return
            # Model returned text without completion JSON — treat as reasoning
            # and continue the loop so it can make more tool calls
            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You returned text instead of a tool call or completion JSON. "
                        "If the task is complete, respond with ONLY: "
                        '{"status": "completed", "summary": "..."} '
                        "If more work is needed, use the available tools."
                    ),
                }
            )

        raise OpenRouterError("Agent reached max steps before completing the task")

    async def _run_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tripletex: TripletexClient,
        execution_state: ExecutionState,
        trace: RunTrace | None = None,
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
            if tool_name == "aggregate_endpoint":
                if trace:
                    trace.write(
                        "tool_start",
                        {
                            "tool_name": "aggregate_endpoint",
                            "arguments": arguments,
                        },
                    )

                endpoint = arguments["endpoint"]
                params = arguments.get("params", {})
                group_by = arguments["group_by"]
                sum_field = arguments["sum_field"]
                sort_dir = arguments.get("sort", "desc")
                limit = arguments.get("limit", 10)

                if not isinstance(params, dict):
                    return {"ok": False, "error": "params must be an object"}
                if sort_dir not in ("asc", "desc"):
                    sort_dir = "desc"
                if not isinstance(limit, int) or limit < 1:
                    limit = 10

                all_values: list[Any] = []
                page_from = 0
                page_size = 1000
                while True:
                    page_params = {**params, "count": page_size, "from": page_from}
                    try:
                        response = await tripletex.request(
                            method="GET", path=endpoint, params=page_params
                        )
                    except Exception as exc:
                        return {"ok": False, "error": str(exc)[:300]}

                    if isinstance(response, dict):
                        values = response.get("values", [])
                        full_size = response.get("fullResultSize", len(values))
                    elif isinstance(response, list):
                        values = response
                        full_size = len(values)
                    else:
                        break

                    if not isinstance(values, list):
                        break

                    all_values.extend(values)
                    page_from += len(values)
                    if page_from >= full_size or not values:
                        break

                groups: defaultdict[str, float] = defaultdict(float)
                group_meta: dict[str, dict[str, Any]] = {}

                def _get_nested(obj: Any, path: str) -> Any:
                    for key in path.split("."):
                        if isinstance(obj, dict):
                            obj = obj.get(key)
                        else:
                            return None
                    return obj

                for item in all_values:
                    if not isinstance(item, dict):
                        continue
                    key = _get_nested(item, group_by)
                    if key is None:
                        continue
                    val = _get_nested(item, sum_field)
                    if val is None:
                        continue
                    try:
                        groups[str(key)] += float(val)
                    except (TypeError, ValueError):
                        continue

                    if str(key) not in group_meta and "account" in group_by:
                        account = _get_nested(item, "account")
                        if isinstance(account, dict):
                            group_meta[str(key)] = {
                                "number": account.get("number"),
                                "name": account.get("name"),
                                "id": account.get("id"),
                            }

                sorted_groups = sorted(
                    groups.items(), key=lambda x: x[1], reverse=(sort_dir == "desc")
                )
                top_results = sorted_groups[:limit]

                result_list: list[dict[str, Any]] = []
                for key, total in top_results:
                    entry: dict[str, Any] = {
                        "group_key": key,
                        "total": round(total, 2),
                    }
                    if key in group_meta:
                        entry["meta"] = group_meta[key]
                    result_list.append(entry)

                result = {
                    "ok": True,
                    "total_records_scanned": len(all_values),
                    "groups_found": len(groups),
                    "top_results": result_list,
                    "summary": (
                        f"Aggregated {len(all_values)} records into {len(groups)} groups, "
                        f"showing top {len(result_list)} by {sum_field} ({sort_dir})"
                    ),
                }

                if trace:
                    trace.write(
                        "tool_result",
                        {
                            "tool_name": "aggregate_endpoint",
                            "result": {
                                "ok": True,
                                "summary": result["summary"],
                            },
                        },
                    )

                return result
            if tool_name == "tripletex_request":
                raw_path = arguments.get("path", "")
                if raw_path.startswith("/v2/"):
                    raw_path = raw_path[3:]
                elif raw_path.startswith("v2/"):
                    raw_path = raw_path[2:]

                if "?" in raw_path:
                    from urllib.parse import urlparse, parse_qs

                    parsed = urlparse(raw_path)
                    raw_path = parsed.path
                    extracted_params = {
                        k: v[0] for k, v in parse_qs(parsed.query).items()
                    }
                    existing_params = arguments.get("params") or {}
                    arguments["params"] = {**extracted_params, **existing_params}

                arguments["path"] = raw_path

                if arguments["method"] == "GET" and arguments.get("json_body"):
                    body = arguments.pop("json_body")
                    if isinstance(body, dict):
                        existing_params = arguments.get("params") or {}
                        arguments["params"] = {
                            **{k: v for k, v in body.items() if v is not None},
                            **existing_params,
                        }

                if arguments["method"] == "PUT" and "/:payment" in raw_path:
                    body = arguments.get("json_body")
                    if isinstance(body, dict) and body:
                        params = arguments.get("params") or {}
                        for key in (
                            "paymentDate",
                            "paymentTypeId",
                            "paidAmount",
                            "paidAmountCurrency",
                        ):
                            if key in body and key not in params:
                                params[key] = body[key]
                        arguments["params"] = params
                        arguments["json_body"] = None

                if execution_state.middleware:
                    (
                        arguments["method"],
                        arguments["path"],
                        arguments["params"],
                        arguments["json_body"],
                    ) = execution_state.middleware.intercept_tool_call(
                        arguments["method"],
                        arguments["path"],
                        arguments.get("params"),
                        arguments.get("json_body"),
                    )
                    dedup_hit = execution_state.middleware.check_dedup(
                        arguments["method"],
                        arguments["path"],
                        arguments.get("json_body"),
                    )
                    if dedup_hit is not None:
                        return {
                            "ok": True,
                            "result": dedup_hit,
                            "dedup": True,
                            "resource_id": _extract_primary_resource_id(dedup_hit),
                        }

                preflight_error, endpoint = self._validate_tripletex_request(
                    arguments,
                    execution_state,
                )
                if preflight_error is not None:
                    return preflight_error

                try:
                    json_body = arguments.get("json_body")
                    if (
                        json_body
                        and isinstance(json_body, dict)
                        and arguments["method"] in ("POST", "PUT")
                    ):
                        fixed_body, corrections = validate_and_fix_payload(
                            arguments["method"],
                            arguments["path"],
                            json_body,
                            self._spec_index,
                        )
                        if corrections:
                            arguments["json_body"] = fixed_body
                            if trace:
                                trace.write(
                                    "schema_corrections",
                                    {
                                        "corrections": corrections,
                                        "path": arguments["path"],
                                    },
                                )
                except Exception as e:
                    logger.debug("Schema validator failed: %s", e)

                try:
                    response = await tripletex.request(
                        method=arguments["method"],
                        path=arguments["path"],
                        params=arguments.get("params"),
                        json_body=arguments.get("json_body"),
                    )
                except TripletexApiError as api_err:
                    if (
                        api_err.status_code == 403
                        and "supplierInvoice" in arguments.get("path", "")
                        and arguments["method"] == "POST"
                    ):
                        logger.info(
                            "Cascading fallback: /supplierInvoice 403 -> /ledger/voucher"
                        )
                        arguments["path"] = "/ledger/voucher"
                        response = await tripletex.request(
                            method=arguments["method"],
                            path=arguments["path"],
                            params=arguments.get("params"),
                            json_body=arguments.get("json_body"),
                        )
                    else:
                        raise

                if execution_state.middleware:
                    execution_state.middleware.process_response(
                        arguments["method"],
                        arguments["path"],
                        response,
                        request_body=arguments.get("json_body"),
                    )
                    execution_state.middleware.record_for_dedup(
                        arguments["method"],
                        arguments["path"],
                        arguments.get("json_body"),
                        response,
                    )
                result = {
                    "ok": True,
                    "endpoint": endpoint,
                    "result": response,
                }
                resource_id = _extract_primary_resource_id(response)
                if resource_id is not None:
                    result["resource_id"] = resource_id
                if execution_state.middleware:
                    refs = execution_state.middleware.get_entity_registry_brief()
                    if refs:
                        result["available_refs"] = {
                            f"$REF:{k}": v for k, v in list(refs.items())[-10:]
                        }
                method_upper = (arguments.get("method") or "").upper()
                if method_upper in ("POST", "PUT"):
                    key_fields = _extract_key_fields(response)
                    execution_state.created_resources.append(
                        CreatedResource(
                            path=arguments.get("path", ""),
                            resource_id=resource_id,
                            method=method_upper,
                            key_fields=key_fields,
                        )
                    )
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
            if tool_name == "tripletex_request" and trace is not None:
                method = str(arguments.get("method") or "").upper()
                path = str(arguments.get("path") or "")
                params_raw = arguments.get("params")
                params = params_raw if isinstance(params_raw, dict) else None
                json_body_raw = arguments.get("json_body")
                json_body = (
                    json_body_raw if isinstance(json_body_raw, (dict, list)) else None
                )

                # Try auto-fix for known 422 patterns
                auto_fix_result = await self._try_auto_fix_422(
                    exc,
                    tripletex,
                    method,
                    path,
                    params,
                    json_body,
                    execution_state,
                    trace,
                )
                if auto_fix_result is not None:
                    return {
                        "ok": True,
                        "status_code": None,
                        "summary": f"Auto-fixed 422 and retried {method} {path}",
                        "error": "",
                        "resource_id": _extract_primary_resource_id(auto_fix_result),
                        "validation_summary": None,
                    }

            body_str = str(exc.body).lower() if exc.body else ""
            is_proxy_token_error = "invalid or expired proxy token" in body_str or (
                "expired" in body_str and "proxy" in body_str
            )
            if is_proxy_token_error:
                execution_state.consecutive_proxy_errors += 1
            else:
                execution_state.consecutive_proxy_errors = 0
                execution_state.total_api_errors += 1
            result = {
                "ok": False,
                "status_code": exc.status_code,
                "error": exc.body,
            }
            if is_proxy_token_error:
                result["proxy_token_expired"] = True
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

    async def _try_auto_fix_422(
        self,
        exc: TripletexApiError,
        tripletex: TripletexClient,
        method: str,
        path: str,
        params: dict[str, Any] | None,
        json_body: dict | list | None,
        execution_state: ExecutionState,
        trace: RunTrace,
    ) -> dict[str, Any] | None:
        """Attempt deterministic auto-fix for known 422 patterns. Returns response if fixed, None if not."""
        _ = tripletex
        _ = method
        _ = path
        _ = params
        _ = json_body
        _ = execution_state

        if exc.status_code != 422 or not isinstance(exc.body, dict):
            return None

        messages = exc.body.get("validationMessages", [])
        if not messages:
            return None

        first_msg = messages[0] if messages else {}
        if not isinstance(first_msg, dict):
            return None

        field = str(first_msg.get("field", ""))
        message = str(first_msg.get("message", ""))
        field_lower = field.lower()
        message_lower = message.lower()

        # Pattern 1: Account ID wrong type (unresolved $REF or non-existent account)
        if "account" in field_lower and "korrekt type" in message_lower:
            trace.write("auto_fix_422", {"pattern": "account_type", "field": field})
            return None

        # Pattern 2: Missing account — create it automatically (future)
        if "account" in field_lower and (
            "finnes ikke" in message_lower or "does not exist" in message_lower
        ):
            trace.write("auto_fix_422", {"pattern": "missing_account", "field": field})
            return None

        # Pattern 3: Postings don't sum to zero
        if "sum" in message_lower and "0" in message:
            trace.write(
                "auto_fix_422_hint",
                {
                    "pattern": "postings_sum",
                    "hint": "Try vatType={id:3} with single debit posting for auto-split, or add explicit credit posting with vatType={id:0}",
                },
            )
            return None

        # Pattern 4: Supplier/employee ref wrong type (unresolved $REF)
        if (
            "supplier" in field_lower or "employee" in field_lower
        ) and "korrekt type" in message_lower:
            if isinstance(json_body, dict):
                postings = json_body.get("postings", [])
                if isinstance(postings, list):
                    for posting in postings:
                        if not isinstance(posting, dict):
                            continue
                        supplier = posting.get("supplier", {})
                        if isinstance(supplier, dict):
                            supplier_id = supplier.get("id")
                            if isinstance(supplier_id, str) and supplier_id.startswith(
                                "$REF:"
                            ):
                                ref_key = supplier_id[5:]
                                trace.write(
                                    "auto_fix_422",
                                    {
                                        "pattern": "unresolved_supplier_ref",
                                        "ref": ref_key,
                                    },
                                )
            return None

        trace.write(
            "auto_fix_422_unhandled",
            {
                "field": field,
                "message": message,
                "status_code": exc.status_code,
            },
        )
        return None

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
            if request_schema:
                # Auto-register schema as inspected without blocking
                execution_state.inspected_schemas.add(request_schema)

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

        strippable: list[dict[str, Any]] = []
        structural: list[dict[str, Any]] = []
        for p in problems:
            if p.get("auto_strip"):
                strippable.append(p)
            else:
                structural.append(p)

        if strippable and isinstance(payload, dict):
            for p in strippable:
                field = p.get("field_name")
                if field and field in payload:
                    del payload[field]
                    logger.warning(
                        "Auto-stripped field '%s' from %s payload: %s",
                        field,
                        schema_name,
                        p.get("message"),
                    )

        if structural:
            primary_problem = structural[0]
            return {
                "ok": False,
                "error": primary_problem.get("message"),
                "schema_validation_errors": structural[:5],
                "stripped_fields": [
                    p.get("field_name") for p in strippable if p.get("field_name")
                ],
                "hint": (
                    "Adjust the JSON body to match the inspected request schema before retrying this write."
                ),
            }

        if strippable:
            logger.info(
                "Auto-stripped %d field(s) from %s payload: %s",
                len(strippable),
                schema_name,
                ", ".join(p.get("field_name", "?") for p in strippable),
            )

        return None

    async def _discover_sandbox_entities(
        self,
        tripletex: TripletexClient,
        planner: PlannerOutput,
        execution_state: ExecutionState,
        trace: RunTrace,
    ) -> dict[str, Any]:
        _ROLE_QUERY_MAP: dict[str, tuple[str, str, str]] = {
            "customer": ("/customer", "organizationNumber", "organization_number"),
            "client": ("/customer", "organizationNumber", "organization_number"),
            "supplier": ("/supplier", "organizationNumber", "organization_number"),
            "vendor": ("/supplier", "organizationNumber", "organization_number"),
            "employee": ("/employee", "email", "email"),
        }
        discovery: dict[str, dict[str, Any]] = {}
        entities = getattr(planner, "entities", [])

        for entity in entities:
            role_lower = entity.role.lower()
            mapping = _ROLE_QUERY_MAP.get(role_lower)
            if mapping is None:
                continue
            endpoint, query_param, slot_field = mapping
            identifier = getattr(entity, slot_field, None)
            if not identifier and entity.name:
                query_param = "name"
                identifier = entity.name
            if not identifier:
                continue

            try:
                result = await tripletex.request(
                    method="GET",
                    path=endpoint,
                    params={query_param: identifier, "count": 5},
                )
            except Exception:
                continue

            values = []
            if isinstance(result, dict):
                values = result.get("values") or []
            if not isinstance(values, list):
                values = []

            track_key = f"GET:{endpoint}?{query_param}={identifier}"
            execution_state.created_entity_keys.add(track_key)

            if not values:
                discovery[role_lower] = {
                    "exists": False,
                    "identifier": identifier,
                    "endpoint": endpoint,
                }
                continue

            existing = values[0]
            existing_id = existing.get("id")
            mismatched_fields: dict[str, dict[str, Any]] = {}

            if entity.name and existing.get("name"):
                if entity.name.strip().lower() != existing["name"].strip().lower():
                    mismatched_fields["name"] = {
                        "expected": entity.name,
                        "actual": existing["name"],
                    }
            if entity.email:
                for f in ("email", "invoiceEmail"):
                    actual = existing.get(f)
                    if (
                        actual
                        and entity.email.strip().lower() != actual.strip().lower()
                    ):
                        mismatched_fields[f] = {
                            "expected": entity.email,
                            "actual": actual,
                        }
            if entity.phone and existing.get("phoneNumber"):
                if entity.phone.strip() != existing["phoneNumber"].strip():
                    mismatched_fields["phoneNumber"] = {
                        "expected": entity.phone,
                        "actual": existing["phoneNumber"],
                    }

            discovery[role_lower] = {
                "exists": True,
                "id": existing_id,
                "identifier": identifier,
                "endpoint": endpoint,
                "matches_task": len(mismatched_fields) == 0,
                "mismatched_fields": mismatched_fields if mismatched_fields else None,
                "key_fields": {
                    k: v
                    for k, v in existing.items()
                    if k in _KEY_FIELD_NAMES and v is not None
                },
            }

        line_items = getattr(planner, "line_items", [])
        for li in line_items:
            if not li.product_number:
                continue
            prod_key = f"product_{li.product_number}"
            if prod_key in discovery:
                continue
            try:
                result = await tripletex.request(
                    method="GET",
                    path="/product",
                    params={"number": li.product_number, "count": 5},
                )
            except Exception:
                continue

            values = []
            if isinstance(result, dict):
                values = result.get("values") or []
            if not isinstance(values, list):
                values = []

            execution_state.created_entity_keys.add(
                f"GET:/product?number={li.product_number}"
            )

            if not values:
                discovery[prod_key] = {
                    "exists": False,
                    "identifier": li.product_number,
                    "endpoint": "/product",
                }
                continue

            existing = values[0]
            mismatched_fields = {}
            if li.unit_price_excluding_vat is not None:
                actual_price = existing.get("priceExcludingVatCurrency")
                if (
                    actual_price is not None
                    and abs(float(actual_price) - li.unit_price_excluding_vat) > 0.01
                ):
                    mismatched_fields["priceExcludingVatCurrency"] = {
                        "expected": li.unit_price_excluding_vat,
                        "actual": actual_price,
                    }

            discovery[prod_key] = {
                "exists": True,
                "id": existing.get("id"),
                "identifier": li.product_number,
                "endpoint": "/product",
                "matches_task": len(mismatched_fields) == 0,
                "mismatched_fields": mismatched_fields if mismatched_fields else None,
                "key_fields": {
                    k: v
                    for k, v in existing.items()
                    if k in _KEY_FIELD_NAMES and v is not None
                },
            }

        if discovery:
            trace.write("sandbox_discovery", discovery)
        return discovery

    async def _batch_resolve_accounts(
        self,
        tripletex: TripletexClient,
        prompt: str,
        execution_state: ExecutionState,
        trace: RunTrace,
        planner: PlannerOutput | None = None,
    ) -> dict[int, int | None]:
        import asyncio

        account_pattern = re.compile(r"\b([1-9]\d{3})\b")
        candidates = set(int(m) for m in account_pattern.findall(prompt))
        accounts_to_resolve = {n for n in candidates if 1000 <= n <= 9999}
        current_year = date.today().year
        accounts_to_resolve -= {current_year, current_year - 1, current_year + 1}

        if planner:
            non_account_numbers: set[int] = set()
            for li in getattr(planner, "line_items", []) or []:
                if li.product_number:
                    try:
                        non_account_numbers.add(int(li.product_number))
                    except (ValueError, TypeError):
                        pass
                if li.unit_price_excluding_vat is not None:
                    price = li.unit_price_excluding_vat
                    for val in (abs(price), abs(price) * 1.25):
                        int_val = int(val)
                        if 1000 <= int_val <= 9999:
                            non_account_numbers.add(int_val)
                if li.quantity is not None:
                    q = int(abs(li.quantity))
                    if 1000 <= q <= 9999:
                        non_account_numbers.add(q)
            if non_account_numbers:
                accounts_to_resolve -= non_account_numbers

        if not accounts_to_resolve:
            return {}

        resolved: dict[int, int | None] = {}

        async def _resolve_one(account_number: int) -> tuple[int, int | None]:
            try:
                result = await tripletex.request(
                    method="GET",
                    path="/ledger/account",
                    params={"number": account_number},
                )
                if isinstance(result, dict):
                    values = result.get("values", [])
                    if values and isinstance(values[0], dict):
                        account_id = values[0].get("id")
                        if isinstance(account_id, int):
                            return (account_number, account_id)
                return (account_number, None)
            except Exception:
                return (account_number, None)

        tasks = [_resolve_one(n) for n in sorted(accounts_to_resolve)]
        results = await asyncio.gather(*tasks)
        for account_number, account_id in results:
            resolved[account_number] = account_id

        if resolved:
            trace.write(
                "batch_account_resolution",
                {
                    "resolved": {str(k): v for k, v in resolved.items()},
                    "found": sum(1 for v in resolved.values() if v is not None),
                    "missing": sum(1 for v in resolved.values() if v is None),
                },
            )

        return resolved

    async def _setup_entities_deterministic(
        self,
        tripletex: TripletexClient,
        planner: PlannerOutput,
        execution_state: ExecutionState,
        trace: RunTrace,
    ) -> None:
        if planner.task_type == "create_department":
            return

        for entity in getattr(planner, "entities", []):
            role = entity.role.lower()
            extra = entity.extra_fields or {}

            if role in execution_state.sandbox_discovery:
                existing = execution_state.sandbox_discovery[role]
                if isinstance(existing, dict) and existing.get("exists"):
                    continue

            try:
                if role == "department" or extra.get("department"):
                    dept_name = entity.name or extra.get("department")
                    if dept_name and role == "department":
                        await tripletex.request(
                            method="POST",
                            path="/department",
                            json_body={"name": dept_name},
                        )
                        trace.write(
                            "entity_setup",
                            {"role": role, "name": dept_name, "action": "created"},
                        )

                elif role == "employee":
                    pass

            except Exception as exc:
                trace.write(
                    "entity_setup_error",
                    {"role": role, "error": str(exc)[:200]},
                )

    async def _try_deterministic_execution(
        self,
        tripletex: TripletexClient,
        planner: PlannerOutput,
        execution_state: ExecutionState,
        trace: RunTrace,
    ) -> dict[str, Any] | None:
        if planner.task_type not in _DETERMINISTIC_TASK_TYPES:
            return None

        entities = getattr(planner, "entities", [])
        if not entities:
            logger.info("Deterministic skipped: no entities extracted by planner")
            return None

        entity = entities[0]
        extra = entity.extra_fields or {}
        payload: dict[str, Any] = {}

        if planner.task_type == "create_supplier":
            if not entity.name:
                return None
            payload = {"name": entity.name}
            if entity.organization_number:
                payload["organizationNumber"] = entity.organization_number
            if entity.email:
                payload["email"] = entity.email
                payload["invoiceEmail"] = entity.email
            if entity.phone:
                payload["phoneNumber"] = entity.phone
            endpoint = "/supplier"

        elif planner.task_type == "create_customer":
            if not entity.name:
                return None
            payload = {"name": entity.name}
            if entity.organization_number:
                payload["organizationNumber"] = entity.organization_number
            if entity.email:
                payload["email"] = entity.email
                payload["invoiceEmail"] = entity.email
            if entity.phone:
                payload["phoneNumber"] = entity.phone
            address = extra.get("address") or extra.get("postalAddress")
            if isinstance(address, dict):
                payload["postalAddress"] = address
            elif isinstance(address, str):
                payload["postalAddress"] = {
                    "addressLine1": address,
                    "city": extra.get("city", ""),
                    "zipCode": extra.get("postal_code", extra.get("postalCode", "")),
                    "country": {"id": 161},
                }
            elif extra.get("addressLine1") or extra.get("street"):
                payload["postalAddress"] = {
                    "addressLine1": extra.get("addressLine1", extra.get("street", "")),
                    "city": extra.get("city", ""),
                    "zipCode": extra.get(
                        "postal_code", extra.get("postalCode", extra.get("zipCode", ""))
                    ),
                    "country": {"id": 161},
                }
            _CUSTOMER_EXTRA_FORWARD = {
                "description",
                "description_request",
                "category",
                "invoicesDueIn",
                "invoicesDueInType",
            }
            _DESCRIPTION_ALIASES = {"description_request"}
            for key in _CUSTOMER_EXTRA_FORWARD:
                val = extra.get(key)
                if val is not None:
                    api_key = "description" if key in _DESCRIPTION_ALIASES else key
                    if isinstance(val, str) and len(val) > 5000:
                        val = val[:5000]
                    payload[api_key] = val
            if "description" not in payload:
                for key, val in extra.items():
                    if (
                        "description" in key.lower() or "handover" in key.lower()
                    ) and isinstance(val, str):
                        payload["description"] = val[:5000] if len(val) > 5000 else val
                        break
            endpoint = "/customer"

        elif planner.task_type == "create_department":
            name = entity.name or extra.get("name")
            if not name:
                return None
            payload = {"name": name}
            endpoint = "/department"

        else:
            return None

        trace.write(
            "deterministic_execution",
            {"task_type": planner.task_type, "endpoint": endpoint, "payload": payload},
        )

        try:
            response = await tripletex.request(
                method="POST",
                path=endpoint,
                json_body=payload,
            )
            resource_id = _extract_primary_resource_id(response)
            if resource_id is not None:
                execution_state.created_resources.append(
                    CreatedResource(
                        path=endpoint,
                        resource_id=resource_id,
                        method="POST",
                        key_fields=_extract_key_fields(response),
                    )
                )
            trace.write(
                "deterministic_result",
                {"ok": True, "resource_id": resource_id, "endpoint": endpoint},
            )
            trace.write(
                "final_payload",
                {
                    "status": "completed",
                    "summary": f"{planner.task_type} completed deterministically via POST {endpoint} (id={resource_id})",
                },
            )
            return {"ok": True, "resource_id": resource_id}
        except TripletexApiError as exc:
            trace.write(
                "deterministic_fallback",
                {"error": str(exc.body)[:300], "status_code": exc.status_code},
            )
            logger.warning(
                "Deterministic execution failed for %s, falling back to executor: %s",
                planner.task_type,
                exc,
            )
            return None

    async def _verify_and_repair(
        self,
        openrouter: OpenRouterClient,
        tripletex: TripletexClient,
        planner: PlannerOutput,
        execution_state: ExecutionState,
        trace: RunTrace,
    ) -> None:
        remaining = 280.0 - (time.monotonic() - execution_state.start_time)
        if remaining < 30:
            return
        if not execution_state.created_resources:
            return

        post_resources = [
            r
            for r in execution_state.created_resources
            if r.method == "POST" and r.resource_id is not None
        ]
        if not post_resources:
            return

        entity_lookup: dict[str, dict[str, Any]] = {}
        for entity in getattr(planner, "entities", []):
            role = entity.role.lower()
            entry: dict[str, Any] = {}
            if entity.name:
                entry["name"] = entity.name
            if entity.email:
                entry["email"] = entity.email
                entry["invoiceEmail"] = entity.email
            if entity.organization_number:
                entry["organizationNumber"] = entity.organization_number
            if entity.phone:
                entry["phoneNumber"] = entity.phone
            entity_lookup[role] = entry

        if not entity_lookup:
            return

        _ENTITY_PATH_ROLES = {
            "/employee": ("employee",),
            "/customer": ("customer", "client"),
            "/supplier": ("supplier", "vendor"),
            "/project": ("project",),
            "/product": ("product",),
            "/department": ("department",),
        }

        _SKIP_VERIFY_PATHS = frozenset(
            {
                "/employee/employment",
                "/employee/employment/details",
                "/employee/standardTime",
                "/employee/entitlement",
                "/salary/transaction",
                "/salary/payslip",
                "/division",
                "/travelExpense",
                "/travelExpense/cost",
                "/travelExpense/perDiemCompensation",
                "/ledger/voucher",
                "/ledger/accountingDimensionName",
                "/ledger/accountingDimensionValue",
                "/order/orderline",
                "/order/orderline/list",
                "/timesheet/entry",
                "/timesheet/entry/list",
                "/project/projectActivity",
                "/activity",
            }
        )

        mismatches: list[dict[str, Any]] = []
        for resource in post_resources:
            remaining = 280.0 - (time.monotonic() - execution_state.start_time)
            if remaining < 20:
                break

            path_base = resource.path.rstrip("/").split("?")[0]

            if path_base in _SKIP_VERIFY_PATHS:
                continue

            matched_roles: tuple[str, ...] = ()
            for endpoint_prefix, roles in _ENTITY_PATH_ROLES.items():
                if path_base == endpoint_prefix:
                    matched_roles = roles
                    break
            if not matched_roles:
                continue

            expected: dict[str, Any] | None = None
            matched_role = ""
            for role in matched_roles:
                if role in entity_lookup:
                    expected = entity_lookup[role]
                    matched_role = role
                    break
            if expected is None:
                continue

            try:
                actual = await tripletex.request(
                    method="GET",
                    path=f"{path_base}/{resource.resource_id}",
                    params={"fields": "*"},
                )
            except Exception:
                continue

            actual_value = (
                actual.get("value", actual) if isinstance(actual, dict) else actual
            )
            if not isinstance(actual_value, dict):
                continue

            field_mismatches: dict[str, dict[str, Any]] = {}
            for field_name, expected_val in expected.items():
                actual_val = actual_value.get(field_name)
                if actual_val is None and expected_val is not None:
                    field_mismatches[field_name] = {
                        "expected": expected_val,
                        "actual": None,
                    }
                elif (
                    isinstance(expected_val, str)
                    and isinstance(actual_val, str)
                    and expected_val.strip().lower() != actual_val.strip().lower()
                ):
                    field_mismatches[field_name] = {
                        "expected": expected_val,
                        "actual": actual_val,
                    }

            if field_mismatches:
                mismatches.append(
                    {
                        "role": matched_role,
                        "path": path_base,
                        "resource_id": resource.resource_id,
                        "mismatches": field_mismatches,
                        "version": actual_value.get("version"),
                    }
                )

        if not mismatches:
            trace.write(
                "verification", {"status": "pass", "checked": len(post_resources)}
            )
            return

        trace.write(
            "verification",
            {
                "status": "mismatches_found",
                "checked": len(post_resources),
                "mismatches": mismatches,
            },
        )

        for mismatch in mismatches:
            remaining = 280.0 - (time.monotonic() - execution_state.start_time)
            if remaining < 15:
                break

            try:
                full_obj = await tripletex.request(
                    method="GET",
                    path=f"{mismatch['path']}/{mismatch['resource_id']}",
                    params={"fields": "*"},
                )
                repair_body = (
                    full_obj.get("value", full_obj)
                    if isinstance(full_obj, dict)
                    else {}
                )
                if not isinstance(repair_body, dict):
                    continue
                repair_body = dict(repair_body)
                for field_name, vals in mismatch["mismatches"].items():
                    repair_body[field_name] = vals["expected"]

                await tripletex.request(
                    method="PUT",
                    path=f"{mismatch['path']}/{mismatch['resource_id']}",
                    json_body=repair_body,
                )
                trace.write(
                    "verification_repair",
                    {
                        "role": mismatch["role"],
                        "resource_id": mismatch["resource_id"],
                        "repaired_fields": list(mismatch["mismatches"].keys()),
                        "status": "success",
                    },
                )
            except Exception as exc:
                trace.write(
                    "verification_repair",
                    {
                        "role": mismatch["role"],
                        "resource_id": mismatch["resource_id"],
                        "status": "failed",
                        "error": str(exc)[:200],
                    },
                )


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
                "name": "aggregate_endpoint",
                "description": (
                    "Fetch ALL data from a paginated Tripletex endpoint and perform "
                    "server-side aggregation. Use this for analysis tasks where you need "
                    "to find top accounts, sum amounts, or compare periods. Returns "
                    "aggregated results — no need to paginate yourself."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "endpoint": {
                            "type": "string",
                            "description": "The GET endpoint to fetch from, e.g. '/ledger/posting'",
                        },
                        "params": {
                            "type": "object",
                            "description": 'Query params for the endpoint, e.g. {"dateFrom": "2026-01-01", "dateTo": "2026-01-31"}',
                            "additionalProperties": True,
                        },
                        "group_by": {
                            "type": "string",
                            "description": "Field path to group by, e.g. 'account.number' or 'account.id'",
                        },
                        "sum_field": {
                            "type": "string",
                            "description": "Field to sum within each group, e.g. 'amount' or 'amountGross'",
                        },
                        "sort": {
                            "type": "string",
                            "enum": ["asc", "desc"],
                            "description": "Sort direction for the aggregated results",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of top results to return (default 10)",
                        },
                    },
                    "required": ["endpoint", "params", "group_by", "sum_field"],
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

    definitions.append(
        {
            "type": "function",
            "function": {
                "name": "ask_api_advisor",
                "description": (
                    "Consult an API advisor before making a Tripletex API call you're "
                    "unsure about. The advisor will research the API documentation, "
                    "inspect relevant schemas, and recommend the EXACT call to make "
                    "including method, path, and body/params. Use this whenever you're "
                    "uncertain about: which endpoint to use, required fields, correct "
                    "field names, or proper request format. This does NOT count as an "
                    "API call — it only reads documentation."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": (
                                "Your specific question about the Tripletex API. "
                                "Include what you're trying to accomplish and any "
                                "entity IDs you've already created."
                            ),
                        },
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
        }
    )

    definitions.append(
        {
            "type": "function",
            "function": {
                "name": "override_enforcer",
                "description": (
                    "Override the enforcer's rejection of your previous tool call. "
                    "Use this ONLY when you are confident the rejected call is correct "
                    "despite the enforcer's concerns. After calling this, re-submit "
                    "the exact same tool call and it will be allowed through."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Why you believe the enforcer is wrong",
                        },
                    },
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            },
        }
    )

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
            "mitarbeiter",
            "employé",
            "empleado",
            "funcionário",
            "entitlement",
            "privilege",
            "administrator",
            "admin",
            "brukertilgang",
            "tilgang",
            "rolle",
            "role",
            "rôle",
            "rol",
            "zugang",
            "accès",
            "acceso",
            "acesso",
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


def _extract_completion_json(content: str) -> dict[str, Any] | None:
    """Extract completion JSON from executor response, even if mixed with text."""
    stripped = content.strip()
    if not stripped:
        return None

    try:
        result = json.loads(stripped)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", stripped, re.DOTALL)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1).strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    for match in re.finditer(r'\{[^{}]*"status"\s*:\s*"completed"[^{}]*\}', stripped):
        try:
            result = json.loads(match.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    for match in re.finditer(r"\{[^{}]+\}", stripped):
        try:
            result = json.loads(match.group(0))
            if isinstance(result, dict) and result.get("status") == "completed":
                return result
        except json.JSONDecodeError:
            continue

    return None


def _build_budget_warning(*, steps_remaining: int, time_remaining: float) -> str | None:
    if steps_remaining <= 3 or time_remaining <= 45:
        return (
            f"[BUDGET] {steps_remaining} steps and {int(time_remaining)}s remaining. "
            "Complete the task NOW. Return completion JSON immediately after your current action."
        )
    if steps_remaining <= 8 or time_remaining <= 90:
        return (
            f"[BUDGET] {steps_remaining} steps and {int(time_remaining)}s remaining. "
            "Focus on the core task only. Skip optional steps."
        )
    return None


def _compress_old_tool_results(
    messages: list[dict[str, Any]], keep_recent: int = 6
) -> None:
    tool_msg_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_msg_indices) <= keep_recent:
        return
    indices_to_compress = tool_msg_indices[:-keep_recent]
    for idx in indices_to_compress:
        msg = messages[idx]
        content_str = msg.get("content", "")
        try:
            parsed = (
                json.loads(content_str) if isinstance(content_str, str) else content_str
            )
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(parsed, dict):
            continue
        if parsed.get("ok") and "summary" in parsed:
            compressed = {"ok": True, "summary": parsed["summary"]}
            resource_id = parsed.get("resource_id")
            if resource_id is not None:
                compressed["resource_id"] = resource_id
            msg["content"] = json.dumps(compressed, ensure_ascii=False)
        elif not parsed.get("ok") and "error" in parsed:
            compressed = {"ok": False, "error_summary": str(parsed["error"])[:200]}
            msg["content"] = json.dumps(compressed, ensure_ascii=False)


def _compact_tool_result_for_model(tool_result: dict[str, Any]) -> dict[str, Any]:
    compacted = compact_response(
        tool_result,
        max_depth=6,
        max_items=20,
        max_string=800,
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

    fields_to_rename: list[tuple[str, str]] = []
    for field_name, value in payload.items():
        field_path = f"{path}.{field_name}"
        definition = properties.get(field_name)
        if not isinstance(definition, dict):
            case_match = _case_insensitive_field_match(field_name, writable_fields)
            if case_match is not None and isinstance(payload, dict) and depth == 0:
                fields_to_rename.append((field_name, case_match))
                logger.warning(
                    "Auto-correcting field casing '%s' -> '%s' in %s payload",
                    field_name,
                    case_match,
                    schema_name,
                )
                continue
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
                    "auto_strip": depth == 0,
                    "field_name": field_name,
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
                    "auto_strip": depth == 0,
                    "field_name": field_name,
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

    if fields_to_rename and isinstance(payload, dict):
        for old_name, new_name in fields_to_rename:
            if old_name in payload:
                payload[new_name] = payload.pop(old_name)

    return problems


def _case_insensitive_field_match(
    field_name: str, schema_fields: list[str]
) -> str | None:
    lower = field_name.lower()
    for sf in schema_fields:
        if sf.lower() == lower and sf != field_name:
            return sf
    return None


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


_KEY_FIELD_NAMES = frozenset(
    {
        "id",
        "version",
        "name",
        "firstName",
        "lastName",
        "email",
        "invoiceEmail",
        "organizationNumber",
        "phoneNumber",
        "number",
        "invoiceNumber",
        "amount",
        "amountCurrency",
        "unitPriceExcludingVatCurrency",
        "fixedPrice",
        "budget",
        "startDate",
        "endDate",
        "invoiceDate",
        "invoiceDueDate",
        "paymentDate",
        "paidAmount",
    }
)


def _extract_key_fields(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    value = result.get("value")
    if not isinstance(value, dict):
        value = result
    return {k: v for k, v in value.items() if k in _KEY_FIELD_NAMES and v is not None}


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

    # Aliases for all 7 competition languages:
    # English, Norwegian (Bokmål), Nynorsk, Spanish, Portuguese, German, French
    resource_aliases = {
        "project": [
            "project",
            "prosjekt",  # en, nb/nn
            "proyecto",
            "projeto",
            "projekt",
            "projet",  # es, pt, de, fr
        ],
        "customer": [
            "customer",
            "kunde",
            "klient",  # en, nb/nn
            "cliente",
            "client",  # es/pt, fr
        ],
        "invoice": [
            "invoice",
            "faktura",
            "regning",  # en, nb/nn
            "factura",
            "fatura",
            "rechnung",
            "facture",  # es, pt, de, fr
        ],
        "order": [
            "order",
            "ordre",  # en, nb/nn/fr
            "orden",
            "pedido",
            "ordem",  # es, pt
            "bestellung",
            "auftrag",  # de
            "commande",  # fr
        ],
        "product": [
            "product",
            "produkt",  # en, nb/nn/de
            "producto",
            "produto",
            "produit",  # es, pt, fr
        ],
        "employee": [
            "employee",
            "ansatt",
            "medarbeider",  # en, nb
            "tilsett",
            "tilsatt",  # nn
            "empleado",
            "empregado",  # es, pt
            "mitarbeiter",
            "angestellter",  # de
            "employé",
            "salarié",  # fr
        ],
        "department": [
            "department",
            "avdeling",  # en, nb/nn
            "departamento",  # es/pt
            "abteilung",  # de
            "département",
            "service",  # fr
        ],
        "payment": [
            "payment",
            "betaling",  # en, nb/nn
            "pago",
            "pagamento",  # es, pt
            "zahlung",  # de
            "paiement",  # fr
        ],
        "travel_expense": [
            "travel expense",
            "reiseregning",  # en, nb
            "reiserekning",  # nn
            "gasto de viaje",
            "despesa de viagem",  # es, pt
            "reisekosten",
            "reisekostenabrechnung",  # de
            "note de frais",  # fr
        ],
        "voucher": [
            "voucher",
            "bilag",
            "kupong",  # en, nb/nn
            "comprobante",
            "comprovante",  # es, pt
            "beleg",
            "gutschein",  # de
            "bon",
            "pièce comptable",  # fr
        ],
        "ledger": [
            "ledger",
            "regnskap",
            "postering",  # en, nb
            "rekneskap",  # nn
            "libro mayor",
            "razão",  # es, pt
            "hauptbuch",  # de
            "grand livre",  # fr
        ],
        "account": [
            "account",
            "konto",  # en, nb/nn/de
            "cuenta",
            "conta",
            "compte",  # es, pt, fr
        ],
        "supplier": [
            "supplier",
            "leverandør",
            "leverandor",  # en, nb/nn
            "proveedor",
            "fornecedor",  # es, pt
            "lieferant",  # de
            "fournisseur",  # fr
        ],
        "credit_note": [
            "credit note",
            "kreditnota",
            "kredittnota",  # en, nb/nn
            "nota de crédito",
            "nota de credito",  # es/pt
            "gutschrift",  # de
            "avoir",  # fr
        ],
        "dimension": [
            "dimension",
            "dimensjon",  # en/de/fr, nb/nn
            "dimensión",
            "dimensão",  # es, pt
        ],
    }

    # Use word-boundary matching to prevent partial matches
    # (e.g. "produkt" should NOT match inside "produktlinje")
    scored_resources: list[tuple[int, str]] = []
    for resource_name, aliases in resource_aliases.items():
        score = 0
        for alias in aliases:
            # Word-boundary regex to avoid substring false positives
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, request_prompt.lower()):
                score += 6
            if re.search(pattern, planner.goal.lower()):
                score += 5
            if re.search(pattern, planner.task_type.lower()):
                score += 4
            if re.search(pattern, text):
                score += 1
        if score > 0:
            scored_resources.append((score, resource_name))

    # Boost the resource that matches the task_type directly
    # e.g. create_voucher -> voucher gets a big boost
    _, inferred_target, _ = _infer_task_operation(planner.task_type)
    if inferred_target:
        for i, (score, resource_name) in enumerate(scored_resources):
            if resource_name == inferred_target:
                scored_resources[i] = (score + 20, resource_name)
                break
        else:
            # Task type target not in scored list — add it with high priority
            scored_resources.append((20, inferred_target))

    scored_resources.sort(key=lambda item: (-item[0], item[1]))
    if not scored_resources:
        if inferred_target:
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

    # Relation role aliases for all 7 competition languages
    role_aliases = {
        "customer": [
            "customer",
            "kunde",
            "klient",
            "client",  # en, nb/nn, fr
            "cliente",  # es/pt
        ],
        "employee": [
            "employee",
            "ansatt",
            "medarbeider",  # en, nb
            "tilsett",
            "tilsatt",  # nn
            "empleado",
            "empregado",  # es, pt
            "mitarbeiter",
            "angestellter",  # de
            "employé",
            "salarié",  # fr
        ],
        "projectManager": [
            "project manager",
            "prosjektleder",  # en, nb/nn
            "projectmanager",  # en variant
            "jefe de proyecto",
            "gerente de projeto",  # es, pt
            "projektleiter",  # de
            "chef de projet",  # fr
        ],
        "contact": [
            "contact",
            "kontakt",  # en, nb/nn/de
            "contact person",
            "kontaktperson",  # en, nb/nn
            "contacto",
            "contato",  # es, pt
            "ansprechpartner",  # de
        ],
        "department": [
            "department",
            "avdeling",  # en, nb/nn
            "departamento",  # es/pt
            "abteilung",  # de
            "département",
            "service",  # fr
        ],
        "supplier": [
            "supplier",
            "leverandør",
            "leverandor",  # en, nb/nn
            "proveedor",
            "fornecedor",  # es, pt
            "lieferant",  # de
            "fournisseur",  # fr
        ],
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
) -> dict[str, Any] | None:
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

    # Task-type-aware family expansions: certain task types legitimately
    # use endpoint families outside their primary resource name.
    _TASK_TYPE_FAMILY_EXPANSIONS: dict[str, set[str]] = {
        "create_voucher": {"ledger", "voucher", "account", "dimension"},
        "reverse_voucher": {"ledger", "voucher", "account"},
        "create_invoice": {
            "invoice",
            "order",
            "customer",
            "product",
            "orderline",
            "ledger",
            "bank",
        },
        "register_payment": {
            "invoice",
            "payment",
            "ledger",
            "voucher",
            "bank",
            "account",
        },
        "register_supplier_invoice": {
            "supplier",
            "ledger",
            "voucher",
            "account",
            "incomingInvoice",
        },
        "create_credit_note": {"invoice", "credit", "ledger"},
        "delete_invoice": {"invoice", "ledger", "order"},
        "create_order": {"order", "customer", "product", "orderline"},
        "create_project": {
            "project",
            "customer",
            "employee",
            "department",
            "activity",
            "timesheet",
            "order",
            "invoice",
            "ledger",
            "hourlyRates",
            "token",
        },
        "create_employee": {"employee", "department", "token"},
        "update_employee": {"employee", "token"},
        "create_travel_expense": {
            "travelExpense",
            "employee",
            "currency",
            "token",
            "department",
        },
        "delete_travel_expense": {"travelExpense", "employee"},
        "bank_reconciliation": {
            "bank",
            "reconciliation",
            "ledger",
            "statement",
            "account",
        },
        "ledger_error_correction": {"ledger", "voucher", "account", "posting"},
        "year_end_closing": {
            "yearEnd",
            "ledger",
            "salary",
            "account",
            "balance",
            "reconciliation",
            "accountingOffice",
        },
    }
    task_type_lower = planner.task_type.lower().strip()
    extra_families = _TASK_TYPE_FAMILY_EXPANSIONS.get(task_type_lower, set())
    allowed_families.update(extra_families)

    if endpoint_family in allowed_families:
        return None

    trailing_families = execution_state.recent_endpoint_families[-6:]
    if len(trailing_families) < 6 or any(
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


def _preflight_enforce(
    tool_name: str,
    arguments: dict[str, Any],
    execution_state: ExecutionState,
) -> dict[str, Any] | None:
    """Pre-flight enforcer: intercepts tool calls before API execution.

    Returns None if the call is allowed, or a rejection dict with
    {rejected: True, reason: str, suggestion: str} if blocked.
    """
    if tool_name != "tripletex_request":
        return None

    method = (arguments.get("method") or "").upper()
    path = (arguments.get("path") or "").lower()
    body = arguments.get("json_body") or {}

    # --- POST /product or /product/list: check if products exist first ---
    if method == "POST" and "/product" in path:
        products_to_check: list[str] = []
        if path.endswith("/list"):
            items = arguments.get("json_body") or []
            if isinstance(items, list):
                for item in items:
                    num = item.get("number") or item.get("productNumber")
                    if num:
                        products_to_check.append(str(num))
        else:
            num = body.get("number") or body.get("productNumber")
            if num:
                products_to_check.append(str(num))

        unchecked = [
            n
            for n in products_to_check
            if f"GET:/product?number={n}" not in execution_state.created_entity_keys
        ]
        if unchecked:
            nums = ", ".join(unchecked)
            return {
                "rejected": True,
                "reason": (
                    f"Product number(s) {nums} may already exist in the sandbox. "
                    "The sandbox pre-seeds products — creating duplicates causes a 422."
                ),
                "suggestion": (
                    f"First call GET /product for each number ({nums}) to check. "
                    "Reuse existing products. Only create ones that don't exist."
                ),
            }

    # --- POST /employee with email: check if employee exists first ---
    if method == "POST" and path.rstrip("/") == "/employee":
        email = body.get("email")
        if email:
            get_key = f"GET:/employee?email={email}"
            if get_key not in execution_state.created_entity_keys:
                return {
                    "rejected": True,
                    "reason": (
                        f"Employee with email {email} may already exist in the sandbox. "
                        "The sandbox pre-seeds employees — creating a duplicate causes a 422."
                    ),
                    "suggestion": (
                        f'First call GET /employee with params {{"email": "{email}"}}. '
                        "If found, reuse the existing employee ID. Only POST if GET returns 0 results."
                    ),
                }

    # --- POST /project without startDate ---
    if (
        method == "POST"
        and "/project" in path
        and path.rstrip("/").endswith("/project")
    ):
        if "startDate" not in body:
            return {
                "rejected": True,
                "reason": "POST /project requires startDate — the API will return a 422 without it.",
                "suggestion": "Add startDate to the body. Use today_iso from the execution brief.",
            }

    # --- POST /ledger/voucher: check amountGross/amountGrossCurrency ---
    if (
        method == "POST"
        and "/ledger/voucher" in path
        and path.rstrip("/").endswith("/voucher")
    ):
        postings = body.get("postings") or []
        for i, posting in enumerate(postings):
            if "amountGross" in posting and "amountGrossCurrency" not in posting:
                return {
                    "rejected": True,
                    "reason": (
                        f"Posting {i}: amountGross is set but amountGrossCurrency is missing. "
                        "Tripletex requires BOTH to be set to the same value."
                    ),
                    "suggestion": (
                        f"Add amountGrossCurrency: {posting['amountGross']} to posting {i}."
                    ),
                }
            row = posting.get("row")
            if row is None or row == 0:
                return {
                    "rejected": True,
                    "reason": (
                        f"Posting {i}: row={'MISSING (omitted)' if row is None else '0'}. "
                        "You MUST set row explicitly. Omitting row is the SAME as row=0. "
                        "Row 0 is system-generated and Tripletex returns 422 for it."
                    ),
                    "suggestion": f'Add "row": {i + 1} to posting {i}. Do NOT omit the row field — omitting it causes the exact same 422 as row=0.',
                }

    # --- POST /customer or /supplier without invoiceEmail when email is set ---
    if method == "POST" and any(p in path for p in ("/customer", "/supplier")):
        if body.get("email") and not body.get("invoiceEmail"):
            return {
                "rejected": True,
                "reason": "email is set but invoiceEmail is missing. Scoring checks BOTH fields.",
                "suggestion": f'Add invoiceEmail: "{body["email"]}" (same as email).',
            }

    # --- Wrong method for payment/reverse/send/creditNote ---
    if method == "POST":
        if "/:payment" in path:
            return {
                "rejected": True,
                "reason": "Payment registration uses PUT, not POST.",
                "suggestion": "Change method to PUT. Use query params: paymentDate, paymentTypeId, paidAmount.",
            }
        if "/:reverse" in path:
            return {
                "rejected": True,
                "reason": "Voucher reversal uses PUT, not POST.",
                "suggestion": "Change method to PUT. Use date as query parameter.",
            }
        if "/:send" in path:
            return {
                "rejected": True,
                "reason": "Invoice sending uses PUT, not POST.",
                "suggestion": "Change method to PUT. Use sendType=EMAIL as query parameter.",
            }
        if "/:createcreditnote" in path:
            return {
                "rejected": True,
                "reason": "Credit note creation uses PUT, not POST.",
                "suggestion": "Change method to PUT. Use date as query parameter.",
            }

    if method == "POST" and "/project/hourlyrates" in path:
        return {
            "rejected": True,
            "reason": (
                "POST /project/hourlyRates will 409 if a default rate already exists. "
                "GET /project/hourlyRates?projectId=X first, then PUT to update."
            ),
            "suggestion": "Call GET /project/hourlyRates?projectId=<project_id> first. If results exist, use PUT to update.",
        }

    if (
        method == "POST"
        and "/activity" in path
        and path.rstrip("/").endswith("/activity")
    ):
        name = body.get("name")
        if name:
            get_key = f"GET:/activity?name={name}"
            if get_key not in execution_state.created_entity_keys:
                return {
                    "rejected": True,
                    "reason": f"Activity '{name}' may already exist — creating a duplicate causes 422.",
                    "suggestion": f"GET /activity?name={name} first. Reuse if found.",
                }

    if method == "POST" and path.rstrip("/") == "/customer":
        org_num = body.get("organizationNumber")
        if org_num:
            get_key = f"GET:/customer?organizationNumber={org_num}"
            if get_key not in execution_state.created_entity_keys:
                return {
                    "rejected": True,
                    "reason": (
                        f"Customer with org number {org_num} may already exist. "
                        "Check before creating to avoid duplicates."
                    ),
                    "suggestion": (
                        f'GET /customer with params {{"organizationNumber": "{org_num}"}}. '
                        "Reuse if found, only POST if not."
                    ),
                }

    if method == "POST" and path.rstrip("/") == "/supplier":
        org_num = body.get("organizationNumber")
        if org_num:
            get_key = f"GET:/supplier?organizationNumber={org_num}"
            if get_key not in execution_state.created_entity_keys:
                return {
                    "rejected": True,
                    "reason": (
                        f"Supplier with org number {org_num} may already exist. "
                        "Check before creating to avoid duplicates."
                    ),
                    "suggestion": (
                        f'GET /supplier with params {{"organizationNumber": "{org_num}"}}. '
                        "Reuse if found, only POST if not."
                    ),
                }

    if method == "GET":
        params = arguments.get("params") or {}
        if "/product" in path and "number" in params:
            execution_state.created_entity_keys.add(
                f"GET:/product?number={params['number']}"
            )
        if "/employee" in path and "email" in params:
            execution_state.created_entity_keys.add(
                f"GET:/employee?email={params['email']}"
            )
        if "/activity" in path and "name" in params:
            execution_state.created_entity_keys.add(
                f"GET:/activity?name={params['name']}"
            )
        if "/customer" in path and "organizationNumber" in params:
            execution_state.created_entity_keys.add(
                f"GET:/customer?organizationNumber={params['organizationNumber']}"
            )
        if "/supplier" in path and "organizationNumber" in params:
            execution_state.created_entity_keys.add(
                f"GET:/supplier?organizationNumber={params['organizationNumber']}"
            )

    return None


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


# ---------------------------------------------------------------------------
# Multi-model routing
# ---------------------------------------------------------------------------

_TASK_TYPE_EXPECTED_ROLES: dict[str, frozenset[str]] = {
    "create_invoice": frozenset({"customer", "client"}),
    "create_order": frozenset({"customer", "client"}),
    "register_supplier_invoice": frozenset({"supplier", "vendor"}),
    "create_customer": frozenset({"customer", "client"}),
    "create_supplier": frozenset({"supplier", "vendor"}),
    "create_employee": frozenset({"employee"}),
    "register_payment": frozenset({"customer", "client", "supplier", "vendor"}),
}


def _maybe_correct_task_type(
    planner: PlannerOutput,
    trace: RunTrace,
) -> PlannerOutput | None:
    expected_roles = _TASK_TYPE_EXPECTED_ROLES.get(planner.task_type)
    if expected_roles is None:
        return None

    actual_roles = frozenset(e.role.lower() for e in getattr(planner, "entities", []))
    if not actual_roles:
        return None

    if actual_roles & expected_roles:
        return None

    alt = getattr(planner, "alternative_task_type", None)
    if not alt:
        return None

    alt_expected = _TASK_TYPE_EXPECTED_ROLES.get(alt)
    if alt_expected and (actual_roles & alt_expected):
        trace.write(
            "task_type_correction",
            {
                "original": planner.task_type,
                "corrected": alt,
                "reason": f"Entity roles {sorted(actual_roles)} match alternative '{alt}' better than primary '{planner.task_type}'",
            },
        )
        data = planner.model_dump()
        data["task_type"] = alt
        data["alternative_task_type"] = planner.task_type
        return PlannerOutput.model_validate(data)

    return None


_TIER_1_TASK_TYPES: frozenset[str] = frozenset(
    {
        "create_employee",
        "create_customer",
        "create_product",
        "create_department",
        "enable_module",
        "update_employee",
        "update_customer",
        "create_supplier",
        "update_contact",
        "delete_travel_expense",
    }
)

_TIER_2_TASK_TYPES: frozenset[str] = frozenset(
    {
        "create_invoice",
        "register_payment",
        "create_project",
        "create_credit_note",
        "create_order",
        "create_travel_expense",
        "delete_invoice",
        "reverse_voucher",
        "create_voucher",
        "register_supplier_invoice",
        "update_supplier",
        "update_product",
        "update_order",
        "update_invoice",
    }
)

_TIER_3_TASK_TYPES: frozenset[str] = frozenset(
    {
        "bank_reconciliation",
        "ledger_error_correction",
        "year_end_closing",
    }
)


def _classify_task_tier(task_type: str) -> int:
    """Classify a planner task_type into scoring tier (1, 2, or 3).

    Handles the ``other_<description>`` prefix the planner sometimes emits
    by stripping it and re-matching against the known tier sets.
    """
    normalized = task_type.lower().strip()

    # Strip 'other_' prefix if present, then try to match the remainder
    candidates = [normalized]
    if normalized.startswith("other_"):
        candidates.append(normalized[len("other_") :])

    for candidate in candidates:
        if candidate in _TIER_1_TASK_TYPES:
            return 1
        if candidate in _TIER_2_TASK_TYPES:
            return 2
        if candidate in _TIER_3_TASK_TYPES:
            return 3

    # Unknown task type → default to Tier 2 (better model, not wasted on Tier 3 chain)
    # True Tier 3 tasks are explicitly listed above.
    return 2


def _build_planner_model_chain(settings: Settings, *, fast: bool = False) -> list[str]:
    primary = settings.planner_fast_model if fast else settings.planner_model
    return _dedupe_models(
        [
            primary,
            settings.planner_model,
            settings.openrouter_model,
        ]
    )


def _build_executor_model_chain(settings: Settings, task_tier: int) -> list[str]:
    """Build ordered model chain for the executor phase based on task tier."""
    if task_tier == 1:
        return _dedupe_models(
            [
                settings.tier1_executor_model,
                settings.openrouter_model,
            ]
        )
    if task_tier == 2:
        return _dedupe_models(
            [
                settings.tier2_executor_model,
                settings.tier1_executor_model,
                settings.openrouter_model,
            ]
        )
    # Tier 3 (default): full escalation chain
    return _dedupe_models(
        [
            settings.tier3_executor_model,
            settings.tier2_executor_model,
            settings.tier1_executor_model,
            settings.openrouter_model,
        ]
    )


def _dedupe_models(models: list[str]) -> list[str]:
    """Deduplicate model list while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for model in models:
        if model not in seen:
            seen.add(model)
            result.append(model)
    return result


def _should_fallback_on_error(exc: Exception) -> bool:
    """Determine if an OpenRouter error is retryable with a different model."""
    if isinstance(exc, OpenRouterError):
        sc = exc.status_code
        if sc is not None:
            # 429 (rate limit), 5xx (server error) → retryable
            if sc == 429 or sc >= 500:
                return True
            # 4xx client errors (400, 401, 403) → not retryable
            return False
        # No status code (empty choices, parse failure) → retryable
        return True
    # httpx transport errors → retryable
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    return False


def _refresh_overview_json() -> None:
    """Regenerate runs/overview.json after a run completes."""
    try:
        from tripletex_agent.dashboard import _generate_overview

        _generate_overview()
    except Exception:
        logger.debug("overview.json refresh failed (non-fatal)", exc_info=True)
