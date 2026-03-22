import copy
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MutationCandidate:
    description: str
    modified_calls: list[dict]
    mutation_type: str


class MutationFuzzer:
    VAT_ROTATION_IDS = (3, 5, 6)
    REMOVE_FIELDS = ("departmentNumber", "isCustomer", "isSupplier")

    def __init__(self, max_candidates: int = 8):
        self.max_candidates = max(1, min(max_candidates, 8))

    def analyze_near_miss(self, trace_path: Path) -> list[MutationCandidate]:
        events = self._load_jsonl_events(trace_path)
        if not events:
            return []

        scoring = self._latest_payload(events, "competition_scoring")
        if not self._is_near_miss(scoring):
            return []

        post_mortem = self._latest_payload(events, "post_mortem")
        task_type = self._task_type(events)
        original_calls = self._extract_calls(events)

        if not original_calls:
            return []

        candidates = self.generate_mutations(original_calls, post_mortem, task_type)
        return candidates[:5]

    def generate_mutations(
        self,
        original_calls: list[dict],
        post_mortem: dict,
        task_type: str,
    ) -> list[MutationCandidate]:
        calls = copy.deepcopy(original_calls)
        suggestions_text = self._flatten_post_mortem(post_mortem)
        relevant_indexes = self._infer_relevant_call_indexes(
            calls, suggestions_text, task_type
        )

        if not relevant_indexes:
            return []

        candidates: list[MutationCandidate] = []
        seen: set[str] = set()

        def add_candidate(
            description: str, mutation_type: str, mutated_calls: list[dict]
        ) -> None:
            if len(candidates) >= self.max_candidates:
                return
            key = (
                mutation_type
                + "::"
                + json.dumps(mutated_calls, sort_keys=True, ensure_ascii=False)
            )
            if key in seen:
                return
            seen.add(key)
            candidates.append(
                MutationCandidate(
                    description=description,
                    modified_calls=mutated_calls,
                    mutation_type=mutation_type,
                )
            )

        for vat_id in self.VAT_ROTATION_IDS:
            mutated = copy.deepcopy(calls)
            changed = False
            for idx in relevant_indexes:
                body = mutated[idx].get("json_body")
                if isinstance(body, dict):
                    changed = self._set_vat_type(body, vat_id) or changed
            if changed:
                add_candidate(
                    f"Rotate vatType.id to {vat_id} on VAT-related calls",
                    "vat_fix",
                    mutated,
                )

        renamed = copy.deepcopy(calls)
        changed_rename = False
        for idx in relevant_indexes:
            body = renamed[idx].get("json_body")
            if isinstance(body, dict):
                changed_rename = (
                    self._rename_key_recursive(body, "fixedprice", "fixedPrice")
                    or changed_rename
                )
                changed_rename = (
                    self._rename_key_recursive(body, "isfixedprice", "isFixedPrice")
                    or changed_rename
                )
        if changed_rename:
            add_candidate(
                "Fix field casing (fixedprice→fixedPrice, isfixedprice→isFixedPrice)",
                "field_fix",
                renamed,
            )

        add_price_type = copy.deepcopy(calls)
        changed_price_type = False
        for idx in relevant_indexes:
            body = add_price_type[idx].get("json_body")
            if isinstance(body, dict):
                changed_price_type = self._ensure_price_type(body) or changed_price_type
        if changed_price_type:
            add_candidate(
                "Add priceType='FIXED_PRICE' where fixed-price fields are used",
                "field_fix",
                add_price_type,
            )

        add_fixed_flag = copy.deepcopy(calls)
        changed_fixed_flag = False
        for idx in relevant_indexes:
            body = add_fixed_flag[idx].get("json_body")
            if isinstance(body, dict):
                changed_fixed_flag = (
                    self._ensure_is_fixed_price(body) or changed_fixed_flag
                )
        if changed_fixed_flag:
            add_candidate(
                "Ensure isFixedPrice=true on fixed-price project payloads",
                "field_fix",
                add_fixed_flag,
            )

        add_refs = copy.deepcopy(calls)
        ref_changed = False
        inferred_customer = self._infer_party_ref(calls, "customer")
        inferred_supplier = self._infer_party_ref(calls, "supplier")
        for idx in relevant_indexes:
            body = add_refs[idx].get("json_body")
            if isinstance(body, dict):
                ref_changed = (
                    self._add_party_refs_to_postings(
                        body, inferred_customer, inferred_supplier
                    )
                    or ref_changed
                )
        if ref_changed:
            add_candidate(
                "Add missing customer/supplier refs for 1500/2400 postings",
                "add_ref",
                add_refs,
            )

        for field_name in self.REMOVE_FIELDS:
            removed = copy.deepcopy(calls)
            changed = False
            for idx in relevant_indexes:
                body = removed[idx].get("json_body")
                if isinstance(body, dict):
                    changed = self._remove_key_recursive(body, field_name) or changed
            if changed:
                add_candidate(
                    f"Remove suspicious field '{field_name}' from request payloads",
                    "remove_field",
                    removed,
                )

        lower_suggestions = suggestions_text.lower()

        if "fixedprice" in lower_suggestions or "fixed price" in lower_suggestions:
            p = copy.deepcopy(calls)
            changed = False
            for idx in relevant_indexes:
                body = p[idx].get("json_body")
                if isinstance(body, dict):
                    changed = (
                        self._rename_key_recursive(body, "fixedprice", "fixedPrice")
                        or changed
                    )
                    changed = self._ensure_price_type(body) or changed
            if changed:
                add_candidate(
                    "Post-mortem: enforce fixedPrice camelCase + priceType",
                    "field_fix",
                    p,
                )

        if "departmentnumber" in lower_suggestions:
            p = copy.deepcopy(calls)
            changed = False
            for idx in relevant_indexes:
                body = p[idx].get("json_body")
                if isinstance(body, dict):
                    changed = (
                        self._remove_key_recursive(body, "departmentNumber") or changed
                    )
            if changed:
                add_candidate(
                    "Post-mortem: remove departmentNumber from department payloads",
                    "remove_field",
                    p,
                )

        if (
            "customer" in lower_suggestions
            or "supplier" in lower_suggestions
            or "1500" in lower_suggestions
            or "2400" in lower_suggestions
        ):
            p = copy.deepcopy(calls)
            changed = False
            for idx in relevant_indexes:
                body = p[idx].get("json_body")
                if isinstance(body, dict):
                    changed = (
                        self._add_party_refs_to_postings(
                            body, inferred_customer, inferred_supplier
                        )
                        or changed
                    )
            if changed:
                add_candidate(
                    "Post-mortem: add missing AR/AP party references",
                    "add_ref",
                    p,
                )

        for vat_id_str in re.findall(r"vattype[^\d]*(\d+)", lower_suggestions):
            vat_id = int(vat_id_str)
            if vat_id <= 0:
                continue
            p = copy.deepcopy(calls)
            changed = False
            for idx in relevant_indexes:
                body = p[idx].get("json_body")
                if isinstance(body, dict):
                    changed = self._set_vat_type(body, vat_id) or changed
            if changed:
                add_candidate(
                    f"Post-mortem: set vatType.id={vat_id}",
                    "vat_fix",
                    p,
                )

        return candidates[: self.max_candidates]

    def apply_mutation(
        self,
        original_calls: list[dict],
        mutation: MutationCandidate,
    ) -> list[dict]:
        if mutation.modified_calls:
            return copy.deepcopy(mutation.modified_calls)
        return copy.deepcopy(original_calls)

    def _load_jsonl_events(self, trace_path: Path) -> list[dict]:
        events: list[dict] = []
        try:
            with trace_path.open("r", encoding="utf-8") as handle:
                for line_number, raw in enumerate(handle, start=1):
                    row = raw.strip()
                    if not row:
                        continue
                    try:
                        obj = json.loads(row)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Skipping corrupt JSONL line %s in %s",
                            line_number,
                            trace_path,
                        )
                        continue
                    if isinstance(obj, dict):
                        events.append(obj)
        except OSError as exc:
            logger.warning("Could not read trace %s: %s", trace_path, exc)
            return []
        return events

    def _latest_payload(self, events: list[dict], event_type: str) -> dict:
        for event in reversed(events):
            if event.get("event_type") == event_type:
                payload = event.get("payload")
                if isinstance(payload, dict):
                    return payload
        return {}

    def _task_type(self, events: list[dict]) -> str:
        planner = self._latest_payload(events, "planner")
        task_type = planner.get("task_type")
        return task_type if isinstance(task_type, str) else ""

    def _extract_calls(self, events: list[dict]) -> list[dict]:
        calls: list[dict] = []
        for event in events:
            if event.get("event_type") != "tool_start":
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            if payload.get("tool_name") != "tripletex_request":
                continue
            arguments = payload.get("arguments")
            if not isinstance(arguments, dict):
                continue
            method = arguments.get("method")
            path = arguments.get("path")
            body = arguments.get("json_body")
            if not isinstance(method, str) or not isinstance(path, str):
                continue
            calls.append(
                {
                    "method": method,
                    "path": path,
                    "json_body": copy.deepcopy(body)
                    if isinstance(body, dict)
                    else body,
                }
            )
        return calls

    def _is_near_miss(self, scoring: dict) -> bool:
        if not scoring:
            return False
        passed = scoring.get("checks_passed")
        total = scoring.get("checks_total")
        if not isinstance(passed, int) or not isinstance(total, int):
            return False
        if total <= 0:
            return False
        if passed <= 0 or passed >= total:
            return False
        return (passed / total) >= 0.5

    def _flatten_post_mortem(self, post_mortem: dict) -> str:
        if not isinstance(post_mortem, dict):
            return ""
        chunks: list[str] = []
        for key in (
            "headline",
            "what_went_wrong",
            "what_to_try_next",
            "root_cause_category",
        ):
            value = post_mortem.get(key)
            if isinstance(value, str):
                chunks.append(value)
            elif isinstance(value, list):
                chunks.extend([v for v in value if isinstance(v, str)])
        return "\n".join(chunks)

    def _infer_relevant_call_indexes(
        self,
        calls: list[dict],
        suggestions_text: str,
        task_type: str,
    ) -> list[int]:
        text = (suggestions_text or "").lower()
        task = (task_type or "").lower()

        relevant: set[int] = set()
        for idx, call in enumerate(calls):
            method = str(call.get("method", "")).upper()
            path = str(call.get("path", "")).lower()
            body = call.get("json_body")

            if method not in {"POST", "PUT", "PATCH"}:
                continue

            if (
                "vat" in text
                and isinstance(body, dict)
                and self._contains_key(body, "vatType")
            ):
                relevant.add(idx)
            if (
                "fixedprice" in text or "priceType" in text or "fixed price" in text
            ) and "/project" in path:
                relevant.add(idx)
            if (
                "departmentnumber" in text or "department" in task
            ) and "/department" in path:
                relevant.add(idx)
            if (
                "customer" in text
                or "supplier" in text
                or "1500" in text
                or "2400" in text
                or "voucher" in text
            ) and isinstance(body, dict):
                if self._contains_key(body, "postings") or "/voucher" in path:
                    relevant.add(idx)

            if isinstance(body, dict):
                if self._contains_key(body, "fixedprice") or self._contains_key(
                    body, "fixedPrice"
                ):
                    relevant.add(idx)
                if self._contains_key(body, "vatType"):
                    relevant.add(idx)
                if any(self._contains_key(body, k) for k in self.REMOVE_FIELDS):
                    relevant.add(idx)

        if relevant:
            return sorted(relevant)

        for idx in range(len(calls) - 1, -1, -1):
            method = str(calls[idx].get("method", "")).upper()
            if method in {"POST", "PUT", "PATCH"} and isinstance(
                calls[idx].get("json_body"), dict
            ):
                return [idx]
        return []

    def _contains_key(self, obj: object, key: str) -> bool:
        if isinstance(obj, dict):
            if key in obj:
                return True
            for value in obj.values():
                if self._contains_key(value, key):
                    return True
        elif isinstance(obj, list):
            for item in obj:
                if self._contains_key(item, key):
                    return True
        return False

    def _rename_key_recursive(self, obj: object, old_key: str, new_key: str) -> bool:
        changed = False
        if isinstance(obj, dict):
            if old_key in obj:
                old_val = obj.pop(old_key)
                if new_key not in obj:
                    obj[new_key] = old_val
                changed = True
            for value in obj.values():
                changed = self._rename_key_recursive(value, old_key, new_key) or changed
        elif isinstance(obj, list):
            for item in obj:
                changed = self._rename_key_recursive(item, old_key, new_key) or changed
        return changed

    def _remove_key_recursive(self, obj: object, target_key: str) -> bool:
        changed = False
        if isinstance(obj, dict):
            if target_key in obj:
                del obj[target_key]
                changed = True
            for value in obj.values():
                changed = self._remove_key_recursive(value, target_key) or changed
        elif isinstance(obj, list):
            for item in obj:
                changed = self._remove_key_recursive(item, target_key) or changed
        return changed

    def _set_vat_type(self, obj: object, vat_id: int) -> bool:
        changed = False
        if isinstance(obj, dict):
            if "vatType" in obj:
                current = obj.get("vatType")
                new_value = {"id": vat_id}
                if current != new_value:
                    obj["vatType"] = new_value
                    changed = True
            for value in obj.values():
                changed = self._set_vat_type(value, vat_id) or changed
        elif isinstance(obj, list):
            for item in obj:
                changed = self._set_vat_type(item, vat_id) or changed
        return changed

    def _ensure_price_type(self, obj: object) -> bool:
        changed = False
        if isinstance(obj, dict):
            has_fixed = "fixedPrice" in obj or "fixedprice" in obj
            has_flag = "isFixedPrice" in obj
            if (has_fixed or has_flag) and "priceType" not in obj:
                obj["priceType"] = "FIXED_PRICE"
                changed = True
            for value in obj.values():
                changed = self._ensure_price_type(value) or changed
        elif isinstance(obj, list):
            for item in obj:
                changed = self._ensure_price_type(item) or changed
        return changed

    def _ensure_is_fixed_price(self, obj: object) -> bool:
        changed = False
        if isinstance(obj, dict):
            has_fixed = "fixedPrice" in obj or "fixedprice" in obj
            if has_fixed and "isFixedPrice" not in obj:
                obj["isFixedPrice"] = True
                changed = True
            for value in obj.values():
                changed = self._ensure_is_fixed_price(value) or changed
        elif isinstance(obj, list):
            for item in obj:
                changed = self._ensure_is_fixed_price(item) or changed
        return changed

    def _infer_party_ref(self, calls: list[dict], party_key: str) -> dict:
        for call in calls:
            body = call.get("json_body")
            if not isinstance(body, dict):
                continue
            ref = self._find_first_ref(body, party_key)
            if isinstance(ref, dict) and "id" in ref:
                return {"id": ref.get("id")}

        placeholder_id = "$REF:customer" if party_key == "customer" else "$REF:supplier"
        return {"id": placeholder_id}

    def _find_first_ref(self, obj: object, key: str):
        if isinstance(obj, dict):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
            for value in obj.values():
                found = self._find_first_ref(value, key)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = self._find_first_ref(item, key)
                if found is not None:
                    return found
        return None

    def _add_party_refs_to_postings(
        self,
        body: dict,
        customer_ref: dict,
        supplier_ref: dict,
    ) -> bool:
        postings = body.get("postings")
        if not isinstance(postings, list):
            return False

        changed = False
        for posting in postings:
            if not isinstance(posting, dict):
                continue
            account_number = self._extract_account_number(posting)
            if account_number == 1500 and "customer" not in posting:
                posting["customer"] = copy.deepcopy(customer_ref)
                changed = True
            if account_number == 2400 and "supplier" not in posting:
                posting["supplier"] = copy.deepcopy(supplier_ref)
                changed = True
        return changed

    def _extract_account_number(self, posting: dict) -> int | None:
        direct_number = posting.get("accountNumber")
        if isinstance(direct_number, int):
            return direct_number
        if isinstance(direct_number, str) and direct_number.isdigit():
            return int(direct_number)

        account = posting.get("account")
        if isinstance(account, dict):
            number = account.get("number")
            if isinstance(number, int):
                return number
            if isinstance(number, str) and number.isdigit():
                return int(number)
        return None
