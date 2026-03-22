import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger(__name__)


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.strip().lower().split())


def _normalize_org_number(value: str | None) -> str:
    if not value:
        return ""
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return digits or _normalize_text(str(value))


def _text_match(query: str | None, candidate: str | None) -> bool:
    q = _normalize_text(query)
    c = _normalize_text(candidate)
    if not q or not c:
        return False
    return q in c or c in q


def _extract_values(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, dict):
        values = response.get("values")
        if isinstance(values, list):
            return [item for item in values if isinstance(item, dict)]
        return []
    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]
    return []


def _first_str(entity: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = entity.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _entity_name(entity: dict[str, Any]) -> str:
    direct_name = _first_str(entity, "name", "fullName", "displayName")
    if direct_name:
        return direct_name
    first_name = _first_str(entity, "firstName")
    last_name = _first_str(entity, "lastName")
    composed = " ".join(part for part in [first_name, last_name] if part)
    return composed.strip()


@dataclass
class SandboxState:
    employees: list[dict[str, Any]] = field(default_factory=list)
    customers: list[dict[str, Any]] = field(default_factory=list)
    suppliers: list[dict[str, Any]] = field(default_factory=list)
    products: list[dict[str, Any]] = field(default_factory=list)
    departments: list[dict[str, Any]] = field(default_factory=list)
    projects: list[dict[str, Any]] = field(default_factory=list)
    accounts: dict[int, dict[str, Any]] = field(default_factory=dict)
    activities: list[dict[str, Any]] = field(default_factory=list)
    salary_types: list[dict[str, Any]] = field(default_factory=list)
    payment_types: list[dict[str, Any]] = field(default_factory=list)

    def find_employee(
        self,
        email: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any] | None:
        email_q = _normalize_text(email)
        if email_q:
            for employee in self.employees:
                for key in ("email", "invoiceEmail"):
                    value = employee.get(key)
                    if isinstance(value, str) and _normalize_text(value) == email_q:
                        return employee
        if name:
            for employee in self.employees:
                if _text_match(name, _entity_name(employee)):
                    return employee
        return None

    def find_customer(
        self,
        org_number: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any] | None:
        org_q = _normalize_org_number(org_number)
        if org_q:
            for customer in self.customers:
                raw = customer.get("organizationNumber")
                if (
                    _normalize_org_number(str(raw) if raw is not None else None)
                    == org_q
                ):
                    return customer
        if name:
            for customer in self.customers:
                if _text_match(name, _entity_name(customer)):
                    return customer
        return None

    def find_supplier(
        self,
        org_number: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any] | None:
        org_q = _normalize_org_number(org_number)
        if org_q:
            for supplier in self.suppliers:
                raw = supplier.get("organizationNumber")
                if (
                    _normalize_org_number(str(raw) if raw is not None else None)
                    == org_q
                ):
                    return supplier
        if name:
            for supplier in self.suppliers:
                if _text_match(name, _entity_name(supplier)):
                    return supplier
        return None

    def find_account(self, number: int) -> dict[str, Any] | None:
        return self.accounts.get(number)

    def find_activity(self, name: str) -> dict[str, Any] | None:
        for activity in self.activities:
            if _text_match(name, _entity_name(activity)):
                return activity
        return None

    def find_department(self, name: str) -> dict[str, Any] | None:
        for department in self.departments:
            if _text_match(name, _entity_name(department)):
                return department
        return None

    def find_project(self, name: str) -> dict[str, Any] | None:
        for project in self.projects:
            if _text_match(name, _entity_name(project)):
                return project
        return None

    def to_context_string(self) -> str:
        def _preview(
            entities: list[dict[str, Any]],
            formatter: Any,
            max_items: int = 3,
        ) -> str:
            if not entities:
                return "none"
            parts: list[str] = []
            for entity in entities[:max_items]:
                try:
                    rendered = formatter(entity)
                except Exception:
                    rendered = "?"
                if rendered:
                    parts.append(str(rendered))
            remaining = len(entities) - len(parts)
            if remaining > 0:
                parts.append(f"+{remaining} more")
            return ", ".join(parts) if parts else "none"

        employee_preview = _preview(
            self.employees,
            lambda item: (
                f"{_entity_name(item)}"
                + (
                    f" (email: {_first_str(item, 'email', 'invoiceEmail')})"
                    if _first_str(item, "email", "invoiceEmail")
                    else ""
                )
            ).strip(),
        )
        customer_preview = _preview(
            self.customers,
            lambda item: (
                f"{_entity_name(item)}"
                + (
                    f" (org: {_first_str(item, 'organizationNumber')})"
                    if _first_str(item, "organizationNumber")
                    else ""
                )
            ).strip(),
        )
        supplier_preview = _preview(
            self.suppliers,
            lambda item: (
                f"{_entity_name(item)}"
                + (
                    f" (org: {_first_str(item, 'organizationNumber')})"
                    if _first_str(item, "organizationNumber")
                    else ""
                )
            ).strip(),
        )
        account_numbers = sorted(self.accounts.keys())
        account_preview_parts: list[str] = []
        for number in account_numbers[:3]:
            account = self.accounts[number]
            name = _entity_name(account)
            account_preview_parts.append(f"{number} ({name})" if name else str(number))
        if len(account_numbers) > 3:
            account_preview_parts.append(f"+{len(account_numbers) - 3} more")
        account_preview = (
            ", ".join(account_preview_parts) if account_preview_parts else "none"
        )

        return (
            "Sandbox: "
            f"employees={len(self.employees)} [{employee_preview}]; "
            f"customers={len(self.customers)} [{customer_preview}]; "
            f"suppliers={len(self.suppliers)} [{supplier_preview}]; "
            f"products={len(self.products)}; "
            f"departments={len(self.departments)}; "
            f"projects={len(self.projects)}; "
            f"accounts={len(self.accounts)} [{account_preview}]; "
            f"activities={len(self.activities)}; "
            f"salary_types={len(self.salary_types)}; "
            f"payment_types={len(self.payment_types)}"
        )


async def _fetch_endpoint(
    tripletex_client: Any,
    *,
    label: str,
    path: str,
    count: int,
) -> list[dict[str, Any]]:
    try:
        response = await tripletex_client.request(
            method="GET",
            path=path,
            params={"count": count},
        )
        values = _extract_values(response)
        logger.info(
            "Shadow discovery fetched %s: %d entities",
            label,
            len(values),
        )
        return values
    except Exception as exc:
        logger.warning("Shadow discovery failed for %s (%s): %s", label, path, exc)
        return []


async def discover_sandbox_parallel(tripletex_client: Any) -> SandboxState:
    logger.info("Starting parallel shadow discovery burst (10 GET calls)")
    (
        employees,
        customers,
        suppliers,
        products,
        departments,
        projects,
        accounts_raw,
        activities,
        salary_types,
        payment_types,
    ) = await asyncio.gather(
        _fetch_endpoint(
            tripletex_client,
            label="employees",
            path="/employee",
            count=100,
        ),
        _fetch_endpoint(
            tripletex_client,
            label="customers",
            path="/customer",
            count=100,
        ),
        _fetch_endpoint(
            tripletex_client,
            label="suppliers",
            path="/supplier",
            count=100,
        ),
        _fetch_endpoint(
            tripletex_client,
            label="products",
            path="/product",
            count=100,
        ),
        _fetch_endpoint(
            tripletex_client,
            label="departments",
            path="/department",
            count=100,
        ),
        _fetch_endpoint(
            tripletex_client,
            label="projects",
            path="/project",
            count=100,
        ),
        _fetch_endpoint(
            tripletex_client,
            label="accounts",
            path="/ledger/account",
            count=1000,
        ),
        _fetch_endpoint(
            tripletex_client,
            label="activities",
            path="/activity",
            count=100,
        ),
        _fetch_endpoint(
            tripletex_client,
            label="salary_types",
            path="/salary/type",
            count=100,
        ),
        _fetch_endpoint(
            tripletex_client,
            label="payment_types",
            path="/invoice/paymentType",
            count=100,
        ),
    )

    account_map: dict[int, dict[str, Any]] = {}
    for account in accounts_raw:
        number_raw = account.get("number")
        if number_raw is None:
            continue
        try:
            number = int(number_raw)
        except (TypeError, ValueError):
            continue
        account_map[number] = account

    state = SandboxState(
        employees=employees,
        customers=customers,
        suppliers=suppliers,
        products=products,
        departments=departments,
        projects=projects,
        accounts=account_map,
        activities=activities,
        salary_types=salary_types,
        payment_types=payment_types,
    )

    logger.info(
        "Shadow discovery complete: employees=%d customers=%d suppliers=%d products=%d departments=%d projects=%d accounts=%d activities=%d salary_types=%d payment_types=%d",
        len(state.employees),
        len(state.customers),
        len(state.suppliers),
        len(state.products),
        len(state.departments),
        len(state.projects),
        len(state.accounts),
        len(state.activities),
        len(state.salary_types),
        len(state.payment_types),
    )
    return state
