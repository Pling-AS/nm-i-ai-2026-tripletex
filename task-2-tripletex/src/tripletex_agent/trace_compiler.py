import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CompiledStep:
    method: str
    path: str
    json_body: dict | None
    params: dict | None
    creates_ref: str | None
    uses_refs: list[str]


@dataclass
class CompiledTrace:
    task_type: str
    prompt_signature: str
    source_trace: str
    source_score: float
    steps: list[CompiledStep]
    total_calls: int


def _read_jsonl_events(trace_path: Path) -> list[dict]:
    events: list[dict] = []
    try:
        with trace_path.open("r", encoding="utf-8") as file:
            for raw in file:
                line = raw.strip()
                if not line:
                    continue
                if not line.startswith("{"):
                    continue
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        events.append(row)
                except Exception:
                    continue
    except OSError as exc:
        logger.warning("Failed reading trace %s: %s", trace_path, exc)
    return events


def _normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    cleaned_chars: list[str] = []
    prev_space = False
    for ch in lowered:
        is_alnum = ch.isalnum() or ch in ("_", "-", ":")
        if is_alnum:
            cleaned_chars.append(ch)
            prev_space = False
        else:
            if not prev_space:
                cleaned_chars.append(" ")
                prev_space = True
    return " ".join("".join(cleaned_chars).split())


def _tokenize_prompt(prompt: str) -> list[str]:
    normalized = _normalize_text(prompt)
    tokens = [t for t in normalized.split(" ") if len(t) >= 3]
    seen: dict[str, bool] = {}
    ordered: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen[token] = True
        ordered.append(token)
        if len(ordered) >= 24:
            break
    return ordered


def _make_prompt_signature(prompt: str) -> str:
    normalized = _normalize_text(prompt)
    tokens = _tokenize_prompt(prompt)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{digest}|{' '.join(tokens)}"


def _extract_score(events: list[dict]) -> tuple[bool, float]:
    best_score = 0.0
    perfect = False
    for event in events:
        if event.get("event_type") != "competition_scoring":
            continue
        payload = event.get("payload") or {}
        passed = payload.get("checks_passed")
        total = payload.get("checks_total")
        if isinstance(passed, int) and isinstance(total, int) and total > 0:
            if passed == total:
                perfect = True
        normalized_score = payload.get("normalized_score")
        raw_score = payload.get("score_raw")
        candidate = None
        if isinstance(normalized_score, (int, float)):
            candidate = float(normalized_score)
        elif isinstance(raw_score, (int, float)):
            candidate = float(raw_score)
        if candidate is not None and candidate > best_score:
            best_score = candidate
    return perfect, best_score


def _is_success(result_payload: dict | None) -> bool:
    if not isinstance(result_payload, dict):
        return False
    ok = result_payload.get("ok")
    if isinstance(ok, bool):
        return ok
    status = result_payload.get("status_code")
    return isinstance(status, int) and 200 <= status < 300


def _first_segment(path: str) -> str:
    parts = [p for p in path.split("/") if p]
    return parts[0] if parts else "resource"


def _sanitize_ref_piece(value: str) -> str:
    out: list[str] = []
    for ch in value:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned[:40] if cleaned else "item"


def _guess_create_ref(
    resource: str, body: dict | None, counters: dict[str, int]
) -> str:
    if isinstance(body, dict):
        for key in ("name", "firstName", "description", "title", "number"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                piece = _sanitize_ref_piece(value)
                base = f"{resource}_{piece}"
                counters[base] = counters.get(base, 0) + 1
                return base if counters[base] == 1 else f"{base}_{counters[base]}"
    base = f"{resource}_id"
    counters[base] = counters.get(base, 0) + 1
    return base if counters[base] == 1 else f"{base}_{counters[base]}"


def _extract_result_id(result_payload: dict | None) -> int | None:
    if not isinstance(result_payload, dict):
        return None
    rid = result_payload.get("resource_id")
    return rid if isinstance(rid, int) else None


def _deep_copy_dict(value: dict | None) -> dict | None:
    if value is None:
        return None
    try:
        copied = json.loads(json.dumps(value))
        return copied if isinstance(copied, dict) else None
    except Exception:
        return None


def _collect_ints(value: object, out: dict[int, bool]) -> None:
    if isinstance(value, int):
        out[value] = True
        return
    if isinstance(value, dict):
        for v in value.values():
            _collect_ints(v, out)
        return
    if isinstance(value, list):
        for item in value:
            _collect_ints(item, out)


def _collect_refs(value: object, out: dict[str, bool]) -> None:
    if isinstance(value, str):
        marker = "$REF:"
        start = value.find(marker)
        while start >= 0:
            ref = value[start + len(marker) :].split(" ")[0].split("/")[0].strip('"')
            if ref:
                out[ref] = True
            nxt = start + len(marker)
            start = value.find(marker, nxt)
        return
    if isinstance(value, dict):
        for v in value.values():
            _collect_refs(v, out)
        return
    if isinstance(value, list):
        for item in value:
            _collect_refs(item, out)


def _replace_ids(
    value: object, id_to_ref: dict[int, str], uses: dict[str, bool]
) -> object:
    if isinstance(value, int):
        ref = id_to_ref.get(value)
        if ref:
            uses[ref] = True
            return f"$REF:{ref}"
        return value
    if isinstance(value, str):
        if value.startswith("$REF:"):
            uses[value[5:]] = True
        return value
    if isinstance(value, dict):
        replaced: dict = {}
        for k, v in value.items():
            replaced[k] = _replace_ids(v, id_to_ref, uses)
        return replaced
    if isinstance(value, list):
        return [_replace_ids(item, id_to_ref, uses) for item in value]
    return value


def _replace_ids_in_path(
    path: str, id_to_ref: dict[int, str], uses: dict[str, bool]
) -> str:
    segments = path.split("/")
    updated: list[str] = []
    for seg in segments:
        if seg.isdigit():
            rid = int(seg)
            ref = id_to_ref.get(rid)
            if ref:
                uses[ref] = True
                updated.append(f"$REF:{ref}")
                continue
        if seg.startswith("$REF:"):
            uses[seg[5:]] = True
        updated.append(seg)
    return "/".join(updated)


def _step_fingerprint(
    method: str, path: str, body: dict | None, params: dict | None
) -> str:
    return json.dumps(
        {
            "method": method.upper(),
            "path": path,
            "json_body": body,
            "params": params,
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _trace_to_compiled_path(trace_path: Path) -> Path:
    compiled_dir = trace_path.parent / "compiled_traces"
    compiled_dir.mkdir(parents=True, exist_ok=True)
    return compiled_dir / f"{trace_path.stem}.json"


def _to_dict(compiled: CompiledTrace) -> dict:
    return {
        "task_type": compiled.task_type,
        "prompt_signature": compiled.prompt_signature,
        "source_trace": compiled.source_trace,
        "source_score": compiled.source_score,
        "total_calls": compiled.total_calls,
        "steps": [
            {
                "method": step.method,
                "path": step.path,
                "json_body": step.json_body,
                "params": step.params,
                "creates_ref": step.creates_ref,
                "uses_refs": step.uses_refs,
            }
            for step in compiled.steps
        ],
    }


def _from_dict(data: dict) -> CompiledTrace | None:
    try:
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list):
            return None
        steps: list[CompiledStep] = []
        for item in raw_steps:
            if not isinstance(item, dict):
                continue
            uses = item.get("uses_refs")
            uses_refs = (
                [u for u in uses if isinstance(u, str)]
                if isinstance(uses, list)
                else []
            )
            steps.append(
                CompiledStep(
                    method=str(item.get("method", "")).upper(),
                    path=str(item.get("path", "")),
                    json_body=item.get("json_body")
                    if isinstance(item.get("json_body"), dict)
                    else None,
                    params=item.get("params")
                    if isinstance(item.get("params"), dict)
                    else None,
                    creates_ref=item.get("creates_ref")
                    if isinstance(item.get("creates_ref"), str)
                    else None,
                    uses_refs=uses_refs,
                )
            )
        return CompiledTrace(
            task_type=str(data.get("task_type", "")),
            prompt_signature=str(data.get("prompt_signature", "")),
            source_trace=str(data.get("source_trace", "")),
            source_score=float(data.get("source_score", 0.0)),
            steps=steps,
            total_calls=int(data.get("total_calls", len(steps))),
        )
    except Exception:
        return None


def compile_trace(trace_path: Path) -> CompiledTrace | None:
    """Read a JSONL trace file and compile it into a minimal execution plan.

    Returns None if the run did not pass all checks.
    Saves a compiled JSON to sibling compiled_traces/ directory when successful.
    """
    events = _read_jsonl_events(trace_path)
    if not events:
        return None

    perfect, score = _extract_score(events)
    if not perfect:
        logger.info("Skipping non-perfect trace: %s", trace_path.name)
        return None

    init_payload = {}
    planner_payload = {}
    done_payload = {}
    for event in events:
        event_type = event.get("event_type")
        payload = event.get("payload") or {}
        if event_type == "init" and isinstance(payload, dict):
            init_payload = payload
        elif event_type == "planner" and isinstance(payload, dict):
            planner_payload = payload
        elif event_type == "done" and isinstance(payload, dict):
            done_payload = payload

    task_type = str(planner_payload.get("task_type") or "")
    prompt = str(init_payload.get("prompt") or "")
    prompt_signature = _make_prompt_signature(prompt)

    pending: list[dict] = []
    calls: list[dict] = []
    for event in events:
        event_type = event.get("event_type")
        payload = event.get("payload") or {}
        if event_type == "tool_start":
            if payload.get("tool_name") != "tripletex_request":
                continue
            arguments = payload.get("arguments") or {}
            if not isinstance(arguments, dict):
                continue
            method = str(arguments.get("method") or "").upper()
            path = str(arguments.get("path") or "")
            body = _deep_copy_dict(arguments.get("json_body"))
            params = _deep_copy_dict(arguments.get("params"))
            call = {
                "method": method,
                "path": path,
                "json_body": body,
                "params": params,
                "result": None,
            }
            pending.append(call)
            calls.append(call)
            continue
        if event_type == "tool_result":
            if payload.get("tool_name") != "tripletex_request":
                continue
            if not pending:
                continue
            result = payload.get("result")
            if not isinstance(result, dict):
                result = {}
            current = pending.pop(0)
            current["result"] = result

    call_log = done_payload.get("tripletex_call_log")
    if isinstance(call_log, list):
        log_idx = 0
        for call in calls:
            if isinstance(call.get("result"), dict):
                continue
            if log_idx >= len(call_log):
                break
            entry = call_log[log_idx]
            log_idx += 1
            if not isinstance(entry, dict):
                continue
            status_code = entry.get("status_code")
            if isinstance(status_code, int):
                call["result"] = {
                    "ok": 200 <= status_code < 300,
                    "status_code": status_code,
                    "resource_id": None,
                }

    successful: list[dict] = []
    for call in calls:
        result = call.get("result")
        if not _is_success(result):
            continue
        method = call.get("method")
        if method not in ("GET", "POST", "PUT"):
            continue
        successful.append(call)

    mutation_calls = [c for c in successful if c.get("method") in ("POST", "PUT")]
    if not mutation_calls:
        logger.info(
            "No successful mutation calls in perfect trace: %s", trace_path.name
        )
        return None

    used_ids: dict[int, bool] = {}
    for call in mutation_calls:
        _collect_ints(call.get("json_body"), used_ids)
        _collect_ints(call.get("params"), used_ids)
        for seg in str(call.get("path") or "").split("/"):
            if seg.isdigit():
                used_ids[int(seg)] = True

    selected: list[dict] = []
    seen_discovery: dict[str, bool] = {}
    for call in successful:
        method = call.get("method")
        if method == "GET":
            rid = _extract_result_id(call.get("result"))
            if rid is None or rid not in used_ids:
                continue
            fp = _step_fingerprint(
                method,
                str(call.get("path") or ""),
                _deep_copy_dict(call.get("json_body")),
                _deep_copy_dict(call.get("params")),
            )
            if fp in seen_discovery:
                continue
            seen_discovery[fp] = True
            selected.append(call)
        elif method in ("POST", "PUT"):
            selected.append(call)

    compact: list[dict] = []
    seen_mut: dict[str, bool] = {}
    for call in selected:
        method = str(call.get("method") or "")
        fp = _step_fingerprint(
            method,
            str(call.get("path") or ""),
            _deep_copy_dict(call.get("json_body")),
            _deep_copy_dict(call.get("params")),
        )
        if method in ("POST", "PUT") and fp in seen_mut:
            continue
        if method in ("POST", "PUT"):
            seen_mut[fp] = True
        compact.append(call)

    id_to_ref: dict[int, str] = {}
    counters: dict[str, int] = {}
    steps: list[CompiledStep] = []

    for call in compact:
        method = str(call.get("method") or "").upper()
        path = str(call.get("path") or "")
        json_body = _deep_copy_dict(call.get("json_body"))
        params = _deep_copy_dict(call.get("params"))
        result = call.get("result") if isinstance(call.get("result"), dict) else {}

        uses: dict[str, bool] = {}
        _collect_refs(path, uses)
        _collect_refs(json_body, uses)
        _collect_refs(params, uses)

        replaced_path = _replace_ids_in_path(path, id_to_ref, uses)
        replaced_body_obj = _replace_ids(json_body, id_to_ref, uses)
        replaced_params_obj = _replace_ids(params, id_to_ref, uses)

        replaced_body = (
            replaced_body_obj if isinstance(replaced_body_obj, dict) else None
        )
        replaced_params = (
            replaced_params_obj if isinstance(replaced_params_obj, dict) else None
        )

        creates_ref: str | None = None
        created_id = _extract_result_id(result)
        if isinstance(created_id, int):
            if created_id in id_to_ref:
                creates_ref = id_to_ref[created_id]
            else:
                resource = _first_segment(path)
                creates_ref = _guess_create_ref(resource, json_body, counters)
                id_to_ref[created_id] = creates_ref

        uses_refs = sorted([r for r in uses.keys() if r])
        steps.append(
            CompiledStep(
                method=method,
                path=replaced_path,
                json_body=replaced_body,
                params=replaced_params,
                creates_ref=creates_ref,
                uses_refs=uses_refs,
            )
        )

    compiled = CompiledTrace(
        task_type=task_type,
        prompt_signature=prompt_signature,
        source_trace=trace_path.name,
        source_score=score,
        steps=steps,
        total_calls=len(steps),
    )

    out_path = _trace_to_compiled_path(trace_path)
    try:
        out_path.write_text(
            json.dumps(_to_dict(compiled), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Failed to save compiled trace %s: %s", out_path, exc)

    return compiled


def _signature_tokens(signature: str) -> list[str]:
    if "|" not in signature:
        return []
    tail = signature.split("|", 1)[1]
    return [t for t in tail.split(" ") if t]


def _prompt_similarity(prompt: str, prompt_signature: str) -> float:
    wanted = _tokenize_prompt(prompt)
    have = _signature_tokens(prompt_signature)
    if not wanted or not have:
        return 0.0
    wanted_set = {w: True for w in wanted}
    have_set = {h: True for h in have}
    overlap = 0
    for tok in wanted_set:
        if tok in have_set:
            overlap += 1
    union = len(wanted_set) + len(have_set) - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def match_compiled_trace(
    prompt: str, task_type: str, compiled_dir: Path
) -> CompiledTrace | None:
    """Find best compiled trace match by task type and prompt similarity."""
    if not compiled_dir.exists() or not compiled_dir.is_dir():
        return None

    best: CompiledTrace | None = None
    best_score = 0.0

    for path in compiled_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        compiled = _from_dict(data)
        if compiled is None:
            continue
        if compiled.task_type != task_type:
            continue
        sim = _prompt_similarity(prompt, compiled.prompt_signature)
        score = sim + (0.001 * (1.0 / max(1, compiled.total_calls)))
        if score > best_score:
            best_score = score
            best = compiled

    if best is None:
        return None
    if _prompt_similarity(prompt, best.prompt_signature) < 0.25:
        return None
    return best


def _resolve_refs_for_replay(
    value: object, refs: dict[str, int], entity_registry: object
) -> object:
    if isinstance(value, str):
        if value.startswith("$REF:"):
            ref_name = value[5:]
            if ref_name in refs:
                return refs[ref_name]
            resolver = getattr(entity_registry, "resolve", None)
            if callable(resolver):
                try:
                    resolved = resolver(ref_name)
                    if isinstance(resolved, int):
                        return resolved
                except Exception:
                    pass
            return value
        return value
    if isinstance(value, dict):
        return {
            k: _resolve_refs_for_replay(v, refs, entity_registry)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_resolve_refs_for_replay(v, refs, entity_registry) for v in value]
    return value


def _resolve_path_for_replay(
    path: str, refs: dict[str, int], entity_registry: object
) -> str:
    segments = path.split("/")
    out: list[str] = []
    for seg in segments:
        if seg.startswith("$REF:"):
            ref_name = seg[5:]
            value = refs.get(ref_name)
            resolver = getattr(entity_registry, "resolve", None)
            if value is None and callable(resolver):
                try:
                    maybe = resolver(ref_name)
                    if isinstance(maybe, int):
                        value = maybe
                except Exception:
                    value = None
            if isinstance(value, int):
                out.append(str(value))
            else:
                out.append(seg)
        else:
            out.append(seg)
    return "/".join(out)


def _extract_response_id(response: object) -> int | None:
    if not isinstance(response, dict):
        return None
    rid = response.get("id")
    if isinstance(rid, int):
        return rid
    wrapped = response.get("value")
    if isinstance(wrapped, dict):
        rid2 = wrapped.get("id")
        if isinstance(rid2, int):
            return rid2
    return None


def replay_trace(
    compiled: CompiledTrace, tripletex_client: object, entity_registry: object
) -> list[dict]:
    """Replay a compiled trace with runtime ID substitution."""
    results: list[dict] = []
    runtime_refs: dict[str, int] = {}

    for idx, step in enumerate(compiled.steps):
        path = _resolve_path_for_replay(step.path, runtime_refs, entity_registry)
        body_obj = _resolve_refs_for_replay(
            step.json_body, runtime_refs, entity_registry
        )
        params_obj = _resolve_refs_for_replay(
            step.params, runtime_refs, entity_registry
        )
        body = body_obj if isinstance(body_obj, dict) else None
        params = params_obj if isinstance(params_obj, dict) else None

        ok = False
        error = ""
        response: object = None
        status_code: int | None = None

        try:
            request_fn = getattr(tripletex_client, "request")
            response = request_fn(
                method=step.method,
                path=path,
                params=params,
                json_body=body,
            )
            if isinstance(response, dict):
                status = response.get("status") or response.get("status_code")
                if isinstance(status, int):
                    status_code = status
                    ok = 200 <= status < 300
                else:
                    ok_field = response.get("ok")
                    if isinstance(ok_field, bool):
                        ok = ok_field
                    else:
                        ok = True
            else:
                ok = True
        except Exception as exc:
            error = str(exc)

        created_id = _extract_response_id(response)
        if ok and step.creates_ref and isinstance(created_id, int):
            runtime_refs[step.creates_ref] = created_id
            registrar = getattr(entity_registry, "register", None)
            if callable(registrar):
                try:
                    registrar(step.creates_ref, created_id)
                except Exception:
                    pass

        results.append(
            {
                "step_index": idx,
                "method": step.method,
                "path": path,
                "ok": ok,
                "status_code": status_code,
                "creates_ref": step.creates_ref,
                "created_id": created_id,
                "error": error,
            }
        )

    return results
