import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _s(val: Any) -> str:
    if isinstance(val, str):
        return val
    if val is None:
        return ""
    return (
        json.dumps(val, ensure_ascii=False)
        if isinstance(val, (dict, list))
        else str(val)
    )


def build_analysis_payload(events: list[dict[str, Any]]) -> str:
    parts: list[str] = []

    for event in events:
        event_type = event.get("event_type", "")
        payload = event.get("payload", {})
        timestamp = event.get("timestamp", "")

        if event_type == "init":
            parts.append(
                f"[{timestamp}] INIT: prompt={_s(payload.get('prompt', ''))[:500]}"
            )
            parts.append(
                f"  files={payload.get('file_count', 0)}, source={_s(payload.get('metadata', {}).get('source', ''))}"
            )

        elif event_type == "planner":
            parts.append(
                f"[{timestamp}] PLANNER: task_type={_s(payload.get('task_type', ''))}"
            )
            parts.append(f"  goal={_s(payload.get('goal', ''))[:300]}")
            steps = payload.get("ordered_steps", [])
            if steps:
                parts.append(
                    f"  steps={json.dumps(steps[:10], ensure_ascii=False)[:500]}"
                )

        elif event_type == "thinking":
            text = _s(payload.get("text", ""))[:800]
            parts.append(f"[{timestamp}] THINKING: {text}")

        elif event_type == "assistant_reasoning":
            text = _s(payload.get("text", ""))[:500]
            parts.append(f"[{timestamp}] REASONING: {text}")

        elif event_type == "tool_start":
            tool_name = _s(payload.get("tool_name", ""))
            args = payload.get("arguments", {})
            if tool_name == "tripletex_request":
                parts.append(
                    f"[{timestamp}] CALL: {_s(args.get('method', '?'))} {_s(args.get('path', '?'))}"
                )
                body = args.get("json_body")
                if body:
                    parts.append(f"  body={json.dumps(body, ensure_ascii=False)[:400]}")
            else:
                parts.append(
                    f"[{timestamp}] TOOL: {tool_name}({json.dumps(args, ensure_ascii=False)[:300]})"
                )

        elif event_type == "tool_result":
            result = payload.get("result", {})
            if not isinstance(result, dict):
                result = {"raw": result}
            tool_name = _s(payload.get("tool_name", ""))
            ok = result.get("ok", True)
            if ok:
                summary = _s(result.get("summary", ""))
                resource_id = result.get("resource_id")
                parts.append(
                    f"[{timestamp}] RESULT OK: {tool_name} summary={summary[:200]} resource_id={resource_id}"
                )
            else:
                error = _s(result.get("error", ""))[:300]
                status_code = result.get("status_code", "?")
                validation = result.get("validation_summary", [])
                parts.append(
                    f"[{timestamp}] RESULT FAIL: {tool_name} status={status_code} error={error}"
                )
                if validation:
                    parts.append(
                        f"  validation={json.dumps(validation, ensure_ascii=False)[:300]}"
                    )

        elif event_type == "enforcer_rejected":
            parts.append(
                f"[{timestamp}] ENFORCER BLOCKED: {_s(payload.get('reason', ''))[:200]}"
            )
            parts.append(f"  suggestion={_s(payload.get('suggestion', ''))[:200]}")

        elif event_type == "semantic_enforcer_rejected":
            parts.append(
                f"[{timestamp}] SEMANTIC BLOCKED: {_s(payload.get('reason', ''))[:200]}"
            )

        elif event_type == "enforcer_override":
            parts.append(
                f"[{timestamp}] ENFORCER OVERRIDE: {_s(payload.get('reason', ''))[:200]}"
            )

        elif event_type == "done":
            parts.append(
                f"[{timestamp}] DONE: calls={payload.get('tripletex_call_count', 0)} errors={payload.get('tripletex_error_count', 0)}"
            )

        elif event_type == "error":
            parts.append(f"[{timestamp}] ERROR: {_s(payload.get('message', ''))[:500]}")

        elif event_type == "competition_scoring":
            parts.append(
                f"[{timestamp}] SCORING: score_raw={payload.get('score_raw', 0)}/{payload.get('score_max', 0)}"
            )
            parts.append(f"  normalized={payload.get('normalized_score', 0)}")
            parts.append(
                f"  checks_passed={payload.get('checks_passed', 0)}/{payload.get('checks_total', 0)}"
            )
            checks = payload.get("checks", [])
            if checks:
                parts.append(f"  checks={json.dumps(checks, ensure_ascii=False)[:600]}")
            comment = _s(payload.get("comment", ""))
            if comment:
                parts.append(f"  comment={comment[:300]}")

        elif event_type == "error_summary":
            errors = payload.get("errors", [])
            parts.append(f"[{timestamp}] ERROR_SUMMARY: {len(errors)} error(s)")
            for err in errors[:5]:
                if isinstance(err, dict):
                    parts.append(
                        f"  - {_s(err.get('tool_name', '?'))}: {_s(err.get('error', ''))[:200]}"
                    )
                else:
                    parts.append(f"  - {_s(err)[:200]}")

        elif event_type == "execution_brief":
            parts.append(
                f"[{timestamp}] BRIEF: {len(payload.get('planned_endpoints', []))} planned endpoints, {payload.get('field_rules_count', 0)} field rules"
            )

        elif event_type == "final_payload":
            parts.append(f"[{timestamp}] FINAL: {_s(payload.get('summary', ''))[:300]}")

    analysis_text = "\n".join(parts)

    if len(analysis_text) > 30000:
        analysis_text = analysis_text[:30000] + "\n... (truncated)"

    return analysis_text


_SCORING_CONTEXT = """
SCORING SYSTEM (critical for analysis):
- Field-by-field verification: each task has specific checks worth different points
- Raw score normalized to 0-1: correctness = points_earned / max_points
- Tier multiplier: Tier 1 (x1), Tier 2 (x2), Tier 3 (x3)
- EFFICIENCY BONUS (only for PERFECT 1.0 correctness runs):
  * Call efficiency: fewer API calls vs best known solution = higher bonus
  * Error cleanliness: fewer 4xx errors = higher bonus
  * A perfect, maximally efficient run scores UP TO 2x the tier score (e.g. Tier 2 perfect efficient = 4.0 pts)
  * A perfect but wasteful run scores only slightly above tier (e.g. Tier 2 perfect wasteful = ~2.1 pts)
- Non-perfect runs get NO efficiency bonus: just correctness x tier
- Best score per task is kept forever — bad runs never lower score
- Efficiency benchmarks recalculate every 12h against all teams

KEY EFFICIENCY INSIGHTS:
- Reducing unnecessary API calls is HIGH VALUE for perfect runs
- Eliminating 4xx errors (400, 404, 422) is HIGH VALUE
- Getting to perfect correctness FIRST, then optimizing calls is the best strategy
- Each failed check is a multiplied loss (tier x lost_fraction)
"""


def build_postmortem_prompt(analysis_payload: str) -> list[dict[str, Any]]:
    system = (
        "You are an expert AI agent debugger analyzing completed Tripletex accounting agent runs. "
        "You receive a chronological trace of the agent's execution including its planning, "
        "API calls, errors, thinking, and competition scoring results.\n\n"
        f"{_SCORING_CONTEXT}\n\n"
        "Analyze the run and provide a structured post-mortem. Be specific and actionable. "
        "Reference actual API calls, error messages, and scoring results. "
        "Pay special attention to efficiency: count total API calls, count 4xx errors, "
        "and suggest specific calls that could be eliminated.\n\n"
        "Return a JSON object with these fields:\n"
        "- headline: One sentence summary (max 100 chars)\n"
        "- what_went_wrong: Array of 1-4 specific issues found (each max 200 chars)\n"
        "- what_worked: Array of 0-2 things that went well (each max 100 chars)\n"
        "- what_to_try_next: Array of 1-3 actionable recommendations (each max 200 chars)\n"
        "- confidence: 'high', 'medium', or 'low' — how confident you are in the analysis\n"
        "- root_cause_category: One of: 'api_error', 'wrong_endpoint', 'wrong_payload', "
        "'missing_entity', 'auth_issue', 'timeout', 'logic_error', 'enforcer_conflict', "
        "'perfect_run', 'unknown'\n"
        "- api_call_count: Total number of Tripletex API calls made\n"
        "- error_call_count: Number of 4xx error responses\n"
        "- unnecessary_calls: Array of 0-3 specific API calls that seem unnecessary (each max 150 chars)\n\n"
        "If the run was PERFECT (all checks passed), focus on what_worked and efficiency improvements — "
        "identify which API calls could be eliminated to improve the efficiency bonus.\n"
        "If checks failed, focus on what_went_wrong with specific API call evidence."
    )

    user = (
        f"Analyze this agent execution trace:\n\n{analysis_payload}\n\n"
        "Return ONLY a JSON object with the fields specified in the system prompt."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


async def generate_postmortem(
    events: list[dict[str, Any]],
    openrouter_client: Any,
    model: str = "claude-sonnet-4-6",
) -> dict[str, Any] | None:
    try:
        analysis_payload = build_analysis_payload(events)
        messages = build_postmortem_prompt(analysis_payload)

        result = await openrouter_client.complete_json(
            messages=messages,
            max_tokens=1500,
            model_override=model,
        )

        expected_fields = {
            "headline",
            "what_went_wrong",
            "confidence",
            "root_cause_category",
        }
        if not expected_fields.issubset(set(result.keys())):
            logger.warning(
                "Post-mortem response missing expected fields: %s", result.keys()
            )
            return None

        return result

    except Exception as exc:
        logger.warning("Post-mortem generation failed: %s", exc)
        return None


def append_postmortem_event(path: Path, analysis: dict[str, Any]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    event = {
        "timestamp": now,
        "event_type": "post_mortem",
        "payload": analysis,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
