import json
import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REF_PATTERN = re.compile(r'"\$REF:([^"]+)"')

_TIME_TAGS: dict[str, str] = {}


def _resolve_time_tag(tag: str) -> str:
    today = date.today()
    mapping = {
        "today": today.isoformat(),
        "safe_today": (today - timedelta(days=1)).isoformat(),
        "yesterday": (today - timedelta(days=1)).isoformat(),
        "tomorrow": (today + timedelta(days=1)).isoformat(),
        "start_of_month": today.replace(day=1).isoformat(),
        "end_of_month": (
            (today.replace(day=28) + timedelta(days=4)).replace(day=1)
            - timedelta(days=1)
        ).isoformat(),
        "start_of_year": date(today.year, 1, 1).isoformat(),
        "end_of_year": date(today.year, 12, 31).isoformat(),
        "start_of_last_year": date(today.year - 1, 1, 1).isoformat(),
        "end_of_last_year": date(today.year - 1, 12, 31).isoformat(),
        "due_30": (today + timedelta(days=30)).isoformat(),
        "safe_project_start": (today - timedelta(days=30)).isoformat(),
    }
    return mapping.get(tag, tag)


_STYRK_DATA: dict[str, dict[str, Any]] | None = None


def _load_styrk_data() -> dict[str, dict[str, Any]]:
    global _STYRK_DATA
    if _STYRK_DATA is not None:
        return _STYRK_DATA
    styrk_path = Path(__file__).parent / "styrk_codes.json"
    if styrk_path.exists():
        try:
            raw = json.loads(styrk_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                _STYRK_DATA = raw
            else:
                _STYRK_DATA = {}
        except Exception:
            _STYRK_DATA = {}
    else:
        _STYRK_DATA = {}
    return _STYRK_DATA


def _fuzzy_styrk_match(query: str) -> dict[str, Any] | None:
    data = _load_styrk_data()
    query_lower = query.lower().strip()

    if query_lower in data:
        return {"code": query_lower, "name": data[query_lower]}

    for code, name in data.items():
        name_str = str(name)
        if query_lower in code or query_lower in name_str.lower():
            return {"code": code, "name": name_str}

    for code, name in data.items():
        name_str = str(name)
        words = query_lower.split()
        if all(w in name_str.lower() for w in words):
            return {"code": code, "name": name_str}

    return None


class EntityRegistry:
    def __init__(self) -> None:
        self._refs: dict[str, int] = {}
        self._by_id: dict[int, str] = {}

    def register(self, ref_name: str, entity_id: int) -> None:
        self._refs[ref_name] = entity_id
        self._by_id[entity_id] = ref_name
        logger.info("Registry: %s -> %d", ref_name, entity_id)

    def resolve(self, ref_name: str) -> int | None:
        return self._refs.get(ref_name)

    def all_refs(self) -> dict[str, int]:
        return dict(self._refs)

    def auto_register_from_request_response(
        self, method: str, path: str, body: dict | None, response: Any
    ) -> None:
        if method not in ("POST", "PUT") or not isinstance(response, dict):
            return

        entity_data = response
        value = response.get("value")
        if isinstance(value, dict) and value.get("id"):
            entity_data = value

        entity_id = entity_data.get("id")
        if not entity_id or not isinstance(entity_id, int):
            return

        resource = path.strip("/").split("/")[0] if "/" in path else path.strip("/")

        name = None
        if isinstance(body, dict):
            name = (
                body.get("name")
                or body.get("firstName")
                or body.get("description")
                or body.get("title")
                or body.get("number")
            )
            if isinstance(name, str):
                name = re.sub(r"[^a-zA-Z0-9æøåÆØÅ_-]", "_", name)[:30].strip("_")

        if name:
            ref_key = f"{resource}_{name}"
        else:
            ref_key = f"{resource}_{entity_id}"

        self.register(ref_key, entity_id)

        version = entity_data.get("version")
        if version is not None:
            self.register(f"{ref_key}_version", version)


def _resolve_refs_in_value(value: Any, registry: EntityRegistry) -> Any:
    if isinstance(value, str):
        if value.startswith("$REF:"):
            ref_name = value[5:]
            resolved = registry.resolve(ref_name)
            if resolved is not None:
                return resolved
            for key, val in registry.all_refs().items():
                if ref_name.lower() in key.lower() or key.lower() in ref_name.lower():
                    logger.info("Fuzzy-resolved $REF:%s -> %s (%d)", ref_name, key, val)
                    return val
            logger.warning("Unresolved $REF:%s — passing through", ref_name)
            return value
        if value.startswith("$TIME:"):
            tag = value[6:]
            resolved = _resolve_time_tag(tag)
            logger.info("Resolved $TIME:%s -> %s", tag, resolved)
            return resolved
        if value.startswith("$STYRK:"):
            query = value[7:]
            match = _fuzzy_styrk_match(query)
            if match:
                logger.info(
                    "Resolved $STYRK:%s -> code %s (%s)",
                    query,
                    match["code"],
                    match["name"],
                )
                return match["code"]
            logger.warning("Unresolved $STYRK:%s", query)
            return value

    if isinstance(value, dict):
        return {k: _resolve_refs_in_value(v, registry) for k, v in value.items()}

    if isinstance(value, list):
        return [_resolve_refs_in_value(item, registry) for item in value]

    return value


class ExecutionMiddleware:
    def __init__(self) -> None:
        self._vat_types: list[dict[str, Any]] | None = None
        self._account_cache: dict[int, dict[str, Any]] = {}
        self._dedup_cache: dict[str, Any] = {}
        self.registry = EntityRegistry()

    async def prefetch_vat_types(self, tripletex: Any) -> list[dict[str, Any]]:
        if self._vat_types is not None:
            return self._vat_types
        try:
            result = await tripletex.request(
                method="GET",
                path="/ledger/vatType",
                params={"count": 1000},
            )
            if isinstance(result, dict):
                values = result.get("values", [])
            elif isinstance(result, list):
                values = result
            else:
                values = []

            vat_map = []
            for v in values:
                if isinstance(v, dict) and v.get("id") is not None:
                    vat_map.append(
                        {
                            "id": v["id"],
                            "number": v.get("number"),
                            "name": v.get("name", ""),
                            "percentage": v.get("percentage"),
                        }
                    )
            self._vat_types = vat_map
            logger.info("Pre-fetched %d VAT types", len(vat_map))
            return vat_map
        except Exception as exc:
            logger.warning("VAT type pre-fetch failed: %s", exc)
            self._vat_types = []
            return []

    _PREFETCH_MAP: dict[str, list[str]] = {
        "create_supplier": [],
        "create_customer": [],
        "create_product": [],
        "create_department": [],
        "create_voucher": [],
        "create_employee": ["employees", "departments"],
        "create_order": ["customers", "products"],
        "create_invoice": [
            "customers",
            "products",
            "employees",
            "payment_types",
            "bank_account",
        ],
        "register_payment": ["customers", "payment_types", "bank_account"],
        "create_credit_note": ["customers", "bank_account"],
        "create_project": ["customers", "employees", "departments", "activities"],
        "create_travel_expense": ["employees", "departments"],
        "delete_travel_expense": ["employees"],
        "register_supplier_invoice": ["suppliers"],
        "reverse_voucher": ["customers", "payment_types", "bank_account"],
        "update_employee": ["employees", "departments"],
        "update_customer": ["customers"],
        "update_supplier": ["suppliers"],
        "update_product": ["products"],
        "update_order": ["customers", "products"],
        "update_invoice": ["customers"],
        "update_contact": ["customers"],
        "delete_invoice": ["customers"],
        "enable_module": [],
        "year_end_closing": [],
        "ledger_error_correction": [],
        "bank_reconciliation": [
            "customers",
            "suppliers",
            "payment_types",
            "bank_account",
        ],
    }

    _ALL_CATEGORIES = [
        "customers",
        "products",
        "employees",
        "suppliers",
        "departments",
        "activities",
        "payment_types",
        "bank_account",
    ]

    async def prefetch_sandbox_context(
        self, tripletex: Any, task_type: str | None = None
    ) -> dict[str, Any]:
        import asyncio

        categories = self._PREFETCH_MAP.get(task_type or "", self._ALL_CATEGORIES)
        if not categories:
            logger.info("Prefetch skipped for task_type=%s", task_type)
            return {}

        call_count_before = tripletex.call_count

        async def _safe_get(path: str, params: dict | None = None) -> list[dict]:
            try:
                result = await tripletex.request(
                    method="GET", path=path, params=params or {"count": 20}
                )
                if isinstance(result, dict):
                    return result.get("values", [])
                return result if isinstance(result, list) else []
            except Exception:
                return []

        _CATEGORY_QUERIES: dict[str, tuple[str, dict[str, int]]] = {
            "customers": ("/customer", {"count": 15}),
            "products": ("/product", {"count": 15}),
            "employees": ("/employee", {"count": 15}),
            "suppliers": ("/supplier", {"count": 15}),
            "departments": ("/department", {"count": 10}),
            "activities": ("/activity", {"count": 20}),
            "payment_types": ("/invoice/paymentType", {"count": 10}),
        }

        tasks: dict[str, asyncio.Task[list[dict]]] = {}
        for cat in categories:
            if cat == "bank_account":
                continue
            query = _CATEGORY_QUERIES.get(cat)
            if query:
                tasks[cat] = asyncio.create_task(_safe_get(query[0], query[1]))

        results: dict[str, list[dict]] = {}
        if tasks:
            gathered = await asyncio.gather(*tasks.values())
            for cat_name, result in zip(tasks.keys(), gathered):
                results[cat_name] = result

        customers = results.get("customers", [])
        products = results.get("products", [])
        employees = results.get("employees", [])
        suppliers = results.get("suppliers", [])
        departments = results.get("departments", [])
        activities = results.get("activities", [])
        payment_types = results.get("payment_types", [])

        def _compact_and_register(
            entities: list[dict], resource_type: str
        ) -> list[dict[str, Any]]:
            compact: list[dict[str, Any]] = []
            for e in entities:
                if not isinstance(e, dict) or not e.get("id"):
                    continue
                name = (
                    e.get("name")
                    or e.get("firstName")
                    or e.get("number")
                    or str(e["id"])
                )
                safe_name = re.sub(r"[^a-zA-Z0-9æøåÆØÅ_-]", "_", str(name))[:30].strip(
                    "_"
                )
                ref_key = f"{resource_type}_{safe_name}"
                self.registry.register(ref_key, e["id"])
                if e.get("version") is not None:
                    self.registry.register(f"{ref_key}_version", e["version"])

                entry: dict[str, Any] = {"ref": f"$REF:{ref_key}", "name": str(name)}
                if e.get("organizationNumber"):
                    entry["orgNr"] = e["organizationNumber"]
                if e.get("email"):
                    entry["email"] = e["email"]
                if e.get("number") and resource_type != "account":
                    entry["number"] = e["number"]
                compact.append(entry)
            return compact

        context: dict[str, Any] = {
            "customers": _compact_and_register(customers, "customer"),
            "products": _compact_and_register(products, "product"),
            "employees": _compact_and_register(employees, "employee"),
            "suppliers": _compact_and_register(suppliers, "supplier"),
            "departments": _compact_and_register(departments, "department"),
            "activities": _compact_and_register(activities, "activity"),
            "payment_types": _compact_and_register(payment_types, "paymentType"),
        }

        if "bank_account" in categories:
            try:
                bank = await tripletex.request(
                    method="GET",
                    path="/ledger/account",
                    params={"number": 1920},
                )
                if isinstance(bank, dict):
                    values = bank.get("values", [])
                    if values and isinstance(values[0], dict):
                        acc = values[0]
                        self.registry.register("account_1920", acc["id"])
                        if acc.get("version") is not None:
                            self.registry.register(
                                "account_1920_version", acc["version"]
                            )
                        context["bank_account"] = {
                            "ref": "$REF:account_1920",
                            "number": 1920,
                            "isBankAccount": acc.get("isBankAccount", False),
                            "version": acc.get("version"),
                        }
            except Exception:
                pass

        total = sum(len(v) for v in context.values() if isinstance(v, list))
        logger.info(
            "Omni-context pre-fetched: %d entities across %d categories",
            total,
            len(context),
        )
        return context

    def intercept_tool_call(
        self, method: str, path: str, params: dict | None, body: dict | list | None
    ) -> tuple[str, str, dict | None, dict | list | None]:
        # Resolve $REF: tokens in URL path segments (prevents 422 on path IDs)
        if "$REF:" in path:
            for ref_key, ref_id in self.registry.all_refs().items():
                token = f"$REF:{ref_key}"
                if token in path:
                    path = path.replace(token, str(ref_id))
                    logger.info("Resolved %s in path -> %d", token, ref_id)

        # Resolve $REF/$TIME/$STYRK in body (handles both dict and list bodies)
        if isinstance(body, (dict, list)):
            body = _resolve_refs_in_value(body, self.registry)

        if isinstance(params, dict):
            params = _resolve_refs_in_value(params, self.registry)

        if method == "POST" and "/ledger/voucher" in path and isinstance(body, dict):
            body = self._resolve_account_ids_in_postings(body)

        if method == "POST" and path == "/project" and isinstance(body, dict):
            start = body.get("startDate")
            if not start:
                safe_start = (date.today() - timedelta(days=30)).isoformat()
                body["startDate"] = safe_start

        return method, path, params, body

    def check_dedup(
        self, method: str, path: str, body: dict | list | None
    ) -> Any | None:
        if method != "POST" or body is None:
            return None
        key = f"{method}:{path}:{json.dumps(body, sort_keys=True, ensure_ascii=False)}"
        cached = self._dedup_cache.get(key)
        if cached is not None:
            logger.info("Dedup hit: %s %s — returning cached response", method, path)
        return cached

    def record_for_dedup(
        self, method: str, path: str, body: dict | list | None, response: Any
    ) -> None:
        if method != "POST" or body is None:
            return
        key = f"{method}:{path}:{json.dumps(body, sort_keys=True, ensure_ascii=False)}"
        self._dedup_cache[key] = response

    def process_response(
        self, method: str, path: str, response: Any, request_body: dict | None = None
    ) -> None:
        if method == "GET" and "/ledger/account" in path:
            self._cache_accounts(response)

        if method in ("POST", "PUT"):
            self.registry.auto_register_from_request_response(
                method, path, request_body, response
            )

        if method == "GET":
            self._register_get_results(path, response)

    def get_entity_registry_brief(self) -> dict[str, int]:
        return self.registry.all_refs()

    def scrub_ids_for_llm(self, value: Any) -> Any:
        reverse_map = {
            v: k
            for k, v in self.registry.all_refs().items()
            if not k.endswith("_version")
        }
        return self._scrub_recursive(value, reverse_map)

    def _scrub_recursive(self, value: Any, reverse_map: dict[int, str]) -> Any:
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                if (
                    k in ("id", "resource_id")
                    and isinstance(v, int)
                    and v in reverse_map
                ):
                    result[k] = f"$REF:{reverse_map[v]}"
                elif k in (
                    "version",
                    "count",
                    "fullResultSize",
                    "from",
                    "status_code",
                    "number",
                    "row",
                    "checks_passed",
                    "checks_total",
                ):
                    result[k] = v
                else:
                    result[k] = self._scrub_recursive(v, reverse_map)
            return result

        if isinstance(value, list):
            return [self._scrub_recursive(item, reverse_map) for item in value]

        if isinstance(value, str):
            scrubbed = value
            for raw_id, ref_name in reverse_map.items():
                scrubbed = scrubbed.replace(str(raw_id), f"$REF:{ref_name}")
            return scrubbed

        return value

    def _cache_accounts(self, response: Any) -> None:
        if not isinstance(response, dict):
            return
        values = response.get("values", [response] if "id" in response else [])
        for v in values:
            if isinstance(v, dict) and v.get("id") and v.get("number"):
                self._account_cache[v["number"]] = {
                    "id": v["id"],
                    "number": v["number"],
                    "name": v.get("name", ""),
                    "version": v.get("version"),
                }

    def _register_get_results(self, path: str, response: Any) -> None:
        if not isinstance(response, dict):
            return
        values = response.get("values", [])
        if not isinstance(values, list):
            return
        resource = path.strip("/").split("/")[0] if "/" in path else path.strip("/")
        for v in values:
            if not isinstance(v, dict) or not v.get("id"):
                continue
            name = v.get("name") or v.get("firstName") or v.get("number")
            if name is not None:
                import re as _re3

                safe = _re3.sub(r"[^a-zA-Z0-9æøåÆØÅ_-]", "_", str(name))[:30].strip("_")
                ref_key = f"{resource}_{safe}"
            else:
                ref_key = f"{resource}_{v['id']}"
            if ref_key not in self.registry.all_refs():
                self.registry.register(ref_key, v["id"])
                if v.get("version") is not None:
                    self.registry.register(f"{ref_key}_version", v["version"])
            number = v.get("number")
            if number is not None:
                num_key = f"account_{number}"
                if num_key not in self.registry.all_refs():
                    self.registry.register(num_key, v["id"])

    def _resolve_account_ids_in_postings(self, body: dict[str, Any]) -> dict[str, Any]:
        postings = body.get("postings")
        if not isinstance(postings, list):
            return body

        for posting in postings:
            if not isinstance(posting, dict):
                continue
            account = posting.get("account")
            if not isinstance(account, dict):
                continue
            if "id" in account and isinstance(account["id"], int) and account["id"] > 0:
                continue
            number = account.get("number")
            if number and isinstance(number, int) and number in self._account_cache:
                cached = self._account_cache[number]
                posting["account"] = {"id": cached["id"]}

        return body

    def get_vat_brief(self) -> dict[str, Any]:
        if not self._vat_types:
            return {}
        vat_25 = next((v for v in self._vat_types if v.get("percentage") == 25.0), None)
        vat_15 = next((v for v in self._vat_types if v.get("percentage") == 15.0), None)
        vat_0 = next(
            (
                v
                for v in self._vat_types
                if v.get("percentage") == 0.0 and "fri" in (v.get("name", "")).lower()
            ),
            None,
        )
        return {
            "vat_types_fetched": True,
            "vat_type_count": len(self._vat_types),
            "vat_25_id": vat_25["id"] if vat_25 else 3,
            "vat_15_id": vat_15["id"] if vat_15 else 5,
            "vat_0_id": vat_0["id"] if vat_0 else 6,
            "all_vat_types": self._vat_types[:20],
        }
