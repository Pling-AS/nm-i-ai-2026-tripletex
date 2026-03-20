from tripletex_agent.agent import (
    ExecutionState,
    TripletexAccountingAgent,
    _build_deterministic_completion_payload,
    _detect_off_target_drift,
    _extract_explicit_prompt_values,
    _find_requested_field_paths,
    _infer_relation_roles,
    _infer_requested_schema_fields,
    _infer_task_resources,
    _should_block_repeated_exploration,
    _detect_blocking_issue,
    _build_validation_hint,
    build_tool_definitions,
)
from tripletex_agent.config import Settings
from tripletex_agent.schemas import PlannerOutput


def test_generic_create_completion_payload() -> None:
    planner = PlannerOutput(
        task_type="create_customer",
        goal="Create customer",
        likely_resources=[],
        success_checks=[],
        risk_notes=[],
        suggested_first_action="Inspect customer endpoint",
    )

    payload = _build_deterministic_completion_payload(
        planner=planner,
        request_prompt="Create customer Acme AS with email post@acme.no",
        tool_name="tripletex_request",
        arguments={"method": "POST", "path": "/customer"},
        tool_result={"ok": True, "resource_id": 42},
    )

    assert payload == {
        "status": "completed",
        "summary": "Customer 42 was created successfully.",
    }


def test_generic_update_completion_payload() -> None:
    planner = PlannerOutput(
        task_type="update_customer",
        goal="Update customer",
        likely_resources=[],
        success_checks=[],
        risk_notes=[],
        suggested_first_action="Find matching customer",
    )

    payload = _build_deterministic_completion_payload(
        planner=planner,
        request_prompt="Update customer Acme AS with new email",
        tool_name="tripletex_request",
        arguments={"method": "PUT", "path": "/customer/42"},
        tool_result={"ok": True, "resource_id": 42},
    )

    assert payload == {
        "status": "completed",
        "summary": "Customer 42 was updated successfully.",
    }


def test_duplicate_create_validation_hint_is_generic() -> None:
    hint = _build_validation_hint(
        tool_name="tripletex_request",
        arguments={"method": "POST", "path": "/customer"},
        validation_summary=[
            {
                "field": "email",
                "message": "Det finnes allerede en kunde med denne e-postadressen.",
            }
        ],
    )

    assert hint is not None
    assert "existing resource" in hint
    assert "update or reuse" in hint


def test_non_matching_path_does_not_complete() -> None:
    planner = PlannerOutput(
        task_type="create_invoice",
        goal="Create invoice",
        likely_resources=[],
        success_checks=[],
        risk_notes=[],
        suggested_first_action="Find customer",
    )

    payload = _build_deterministic_completion_payload(
        planner=planner,
        request_prompt="Create invoice for customer Acme AS",
        tool_name="tripletex_request",
        arguments={"method": "POST", "path": "/order"},
        tool_result={"ok": True, "resource_id": 88},
    )

    assert payload is None


def test_missing_linked_reference_validation_hint_is_generic() -> None:
    hint = _build_validation_hint(
        tool_name="tripletex_request",
        arguments={"method": "POST", "path": "/invoice"},
        validation_summary=[
            {
                "field": "customer.id",
                "message": "Feltet må fylles ut.",
            }
        ],
    )

    assert hint is not None
    assert "linked `customer` reference" in hint
    assert '{"customer": {"id": ...}}' in hint


def test_external_blocker_validation_hint_is_generic() -> None:
    hint = _build_validation_hint(
        tool_name="tripletex_request",
        arguments={"method": "POST", "path": "/invoice"},
        validation_summary=[
            {
                "field": "",
                "message": "Faktura kan ikke opprettes før selskapet har registrert et bankkontonummer.",
            }
        ],
    )

    assert hint is not None
    assert "external or company-level setup" in hint
    assert "requested task" in hint


def test_execution_brief_collects_candidate_endpoints() -> None:
    agent = TripletexAccountingAgent(Settings())
    planner = PlannerOutput(
        task_type="create_invoice",
        goal="Create invoice",
        likely_resources=["Tripletex API", "Order data"],
        success_checks=["Invoice exists"],
        risk_notes=["Linked entities may be required"],
        suggested_first_action="Find customer",
    )

    brief = agent._build_execution_brief(
        planner=planner,
        request_prompt="Create invoice for customer Acme AS",
    )

    assert brief["task_type"] == "create_invoice"
    assert brief["candidate_queries"]
    assert brief["candidate_endpoints"]
    assert all("path" in item for item in brief["candidate_endpoints"])
    assert all("method" in item for item in brief["candidate_endpoints"])
    assert any("external or company-level" in rule for rule in brief["working_rules"])


def test_infer_task_resources_prefers_prompt_target_over_planner_drift() -> None:
    planner = PlannerOutput(
        task_type="project_registration",
        goal="Register a project linked to Kari Trå",
        likely_resources=["Kari Trå's user ID", "Project details"],
        success_checks=[],
        risk_notes=[],
        suggested_first_action="Inspect Kari Trå's user ID in the API",
    )

    primary_resource, linked_resources = _infer_task_resources(
        planner=planner,
        request_prompt="Kan du registrere et prosjekt linket til Kari Trå?",
    )

    assert primary_resource == "project"
    assert "employee" not in linked_resources


def test_off_target_drift_returns_recovery_signal_for_unrelated_family() -> None:
    planner = PlannerOutput(
        task_type="project_registration",
        goal="Register a project linked to Kari Trå",
        likely_resources=["Kari Trå's user ID", "Project details"],
        success_checks=[],
        risk_notes=[],
        suggested_first_action="Inspect Kari Trå's user ID in the API",
    )
    execution_state = ExecutionState(recent_endpoint_families=["employee", "employee"])

    blocked = _detect_off_target_drift(
        planner=planner,
        request_prompt="Kan du registrere et prosjekt linket til Kari Trå?",
        tool_name="tripletex_request",
        arguments={"method": "GET", "path": "/employee"},
        execution_state=execution_state,
    )

    assert blocked is not None
    assert "drifting into the `employee` endpoint family" in blocked["message"]
    assert "`project`" in blocked["message"]
    assert blocked["grounded_primary_resource"] == "project"
    assert "Re-anchor on the grounded primary resource `project`" in blocked["hint"]


def test_execution_brief_includes_grounded_primary_resource() -> None:
    agent = TripletexAccountingAgent(Settings())
    planner = PlannerOutput(
        task_type="project_registration",
        goal="Register a project linked to Kari Trå",
        likely_resources=["Kari Trå's user ID", "Project details"],
        success_checks=[],
        risk_notes=[],
        suggested_first_action="Inspect Kari Trå's user ID in the API",
    )

    brief = agent._build_execution_brief(
        planner=planner,
        request_prompt="Kan du registrere et prosjekt linket til Kari Trå?",
    )

    assert brief["primary_resource"] == "project"
    assert brief["candidate_endpoints"]


def test_relation_roles_prefer_customer_when_prompt_says_kunde() -> None:
    planner = PlannerOutput(
        task_type="create_project",
        goal="Register a project linked to Kari Trå",
        likely_resources=["Tripletex API", "Kari Trå's client ID"],
        success_checks=[],
        risk_notes=[],
        suggested_first_action="Inspect existing projects for Kari Trå",
    )

    roles = _infer_relation_roles(
        planner=planner,
        request_prompt="Kan du registrere et prosjekt linket til kunden Kari Trå?",
    )

    assert roles
    assert roles[0] == "customer"


def test_relation_roles_prefer_employee_when_prompt_says_ansatt() -> None:
    planner = PlannerOutput(
        task_type="create_project",
        goal="Register a project linked to Ola Nordmann",
        likely_resources=["Tripletex API", "Employee record"],
        success_checks=[],
        risk_notes=[],
        suggested_first_action="Inspect employee endpoint",
    )

    roles = _infer_relation_roles(
        planner=planner,
        request_prompt="Kan du registrere et prosjekt linket til den ansatte Ola Nordmann?",
    )

    assert roles
    assert roles[0] == "employee"


def test_execution_brief_surfaces_preferred_link_fields() -> None:
    agent = TripletexAccountingAgent(Settings())
    planner = PlannerOutput(
        task_type="create_project",
        goal="Register a project linked to Kari Trå",
        likely_resources=["Tripletex API", "Kari Trå's client ID"],
        success_checks=[],
        risk_notes=[],
        suggested_first_action="Inspect existing projects for Kari Trå",
    )

    brief = agent._build_execution_brief(
        planner=planner,
        request_prompt="Kan du registrere et prosjekt linket til kunden Kari Trå?",
    )

    assert brief["preferred_link_fields"]
    assert brief["preferred_link_fields"][0] == "customer"


def test_infer_requested_schema_fields_captures_email_and_dates_from_prompt() -> None:
    fields = _infer_requested_schema_fields(
        "Tenemos un nuevo empleado llamado Sofía Rodríguez, nacido el 25. August 2000. "
        "Créelo como empleado con el correo sofia.rodriguez@example.org y fecha de inicio 20. April 2026."
    )

    assert "email" in fields
    assert "dateOfBirth" in fields
    assert "startDate" in fields


def test_extract_explicit_prompt_values_collects_dates_and_emails() -> None:
    values = _extract_explicit_prompt_values(
        "Create employee Ola Nordmann with email ola@example.org starting 2026-04-20."
    )

    assert values["emails"] == ["ola@example.org"]
    assert values["dates"] == ["2026-04-20"]


def test_find_requested_field_paths_discovers_nested_schema_fields() -> None:
    agent = TripletexAccountingAgent(Settings())

    field_paths = _find_requested_field_paths(
        agent._spec_index,
        "Employee",
        ["email", "dateOfBirth", "startDate"],
    )

    assert "email" in field_paths
    assert "dateOfBirth" in field_paths
    assert "employments[].startDate" in field_paths


def test_execution_brief_surfaces_requested_field_paths_for_nested_schema_matches() -> (
    None
):
    agent = TripletexAccountingAgent(Settings())
    planner = PlannerOutput(
        task_type="create_employee",
        goal="Add new employee Sofía Rodríguez",
        likely_resources=["Tripletex API documentation", "Employee creation endpoint"],
        success_checks=["Employee details match input"],
        risk_notes=[],
        suggested_first_action="Inspect employee creation endpoint documentation",
    )

    brief = agent._build_execution_brief(
        planner=planner,
        request_prompt=(
            "Tenemos un nuevo empleado llamado Sofía Rodríguez, nacido el 25. August 2000. "
            "Créelo como empleado con el correo sofia.rodriguez@example.org y fecha de inicio 20. April 2026."
        ),
    )

    assert "startDate" in brief["requested_field_names"]
    assert brief["explicit_prompt_values"]["emails"] == ["sofia.rodriguez@example.org"]
    employee_candidates = [
        item
        for item in brief["candidate_endpoints"]
        if item.get("requestSchema") == "Employee"
    ]
    assert employee_candidates
    assert any(
        "employments[].startDate" in item.get("requested_field_paths", [])
        for item in employee_candidates
    )


def test_enum_shape_validation_hint_is_generic() -> None:
    hint = _build_validation_hint(
        tool_name="tripletex_request",
        arguments={"method": "POST", "path": "/employee"},
        validation_summary=[
            {
                "field": "",
                "message": 'Brukertype kan ikke være "0" eller tom.',
            }
        ],
    )

    assert hint is not None
    assert "exact supported enum literals" in hint


def test_repeated_exploration_is_blocked_even_after_cached_repeat() -> None:
    repeated_key = (
        '{"arguments": {"query": "Kari Trå"}, "tool_name": "search_tripletex_api"}'
    )

    assert (
        _should_block_repeated_exploration(
            "search_tripletex_api",
            {"query": "Kari Trå"},
            [repeated_key],
        )
        is True
    )


def test_completion_falls_back_to_grounded_target_when_task_type_is_weak() -> None:
    planner = PlannerOutput(
        task_type="project_registration",
        goal="Register a project linked to Kari Trå",
        likely_resources=["Project details"],
        success_checks=[],
        risk_notes=[],
        suggested_first_action="Find project endpoint",
    )

    payload = _build_deterministic_completion_payload(
        planner=planner,
        request_prompt="Kan du registrere et prosjekt linket til Kari Trå?",
        tool_name="tripletex_request",
        arguments={"method": "POST", "path": "/project"},
        tool_result={"ok": True, "resource_id": 77},
    )

    assert payload == {
        "status": "completed",
        "summary": "Project 77 was created successfully.",
    }


def test_blocking_issue_detects_external_setup_outside_task_scope() -> None:
    planner = PlannerOutput(
        task_type="create_invoice",
        goal="Create invoice",
        likely_resources=[],
        success_checks=[],
        risk_notes=[],
        suggested_first_action="Find customer",
    )

    blocked = _detect_blocking_issue(
        planner=planner,
        request_prompt="Create invoice for customer Acme AS",
        tool_name="tripletex_request",
        arguments={"method": "POST", "path": "/invoice"},
        tool_result={
            "ok": False,
            "validation_summary": [
                {
                    "field": "",
                    "message": "Faktura kan ikke opprettes før selskapet har registrert et bankkontonummer.",
                }
            ],
        },
    )

    assert blocked is not None
    assert "blocked by an external or company-level prerequisite" in blocked["message"]


def test_entitlement_tool_is_not_exposed_for_unrelated_task() -> None:
    tools = build_tool_definitions(
        planner_task_type="create_invoice",
        request_prompt="Create invoice for customer Acme AS",
    )

    tool_names = [tool["function"]["name"] for tool in tools]
    assert "grant_employee_entitlements" not in tool_names


def test_entitlement_tool_is_exposed_for_employee_access_task() -> None:
    tools = build_tool_definitions(
        planner_task_type="create_employee",
        request_prompt="Create employee with administrator access",
    )

    tool_names = [tool["function"]["name"] for tool in tools]
    assert "grant_employee_entitlements" in tool_names


def test_repeated_identical_exploration_is_blocked() -> None:
    repeated_key = '{"arguments": {"query": "Get all products"}, "tool_name": "search_tripletex_api"}'

    assert (
        _should_block_repeated_exploration(
            "search_tripletex_api",
            {"query": "Get all products"},
            [repeated_key],
        )
        is True
    )


def test_invoice_preflight_blocks_customer_id_and_invoice_lines() -> None:
    agent = TripletexAccountingAgent(Settings())
    execution_state = ExecutionState(inspected_schemas={"Invoice"})

    preflight_error, endpoint = agent._validate_tripletex_request(
        {
            "method": "POST",
            "path": "/invoice",
            "json_body": {
                "customerId": 123,
                "invoiceLines": [{"description": "x"}],
            },
        },
        execution_state,
    )

    assert endpoint is not None
    assert preflight_error is not None
    assert preflight_error["ok"] is False
    assert "customerId" in preflight_error["error"]
    assert "customer" in preflight_error["error"]


def test_order_preflight_suggests_count_for_quantity() -> None:
    agent = TripletexAccountingAgent(Settings())
    execution_state = ExecutionState(inspected_schemas={"Order"})

    preflight_error, endpoint = agent._validate_tripletex_request(
        {
            "method": "POST",
            "path": "/order",
            "json_body": {
                "customer": {"id": 1},
                "orderLines": [
                    {
                        "product": {"id": 2},
                        "quantity": 1,
                        "unitPrice": 100,
                    }
                ],
            },
        },
        execution_state,
    )

    assert endpoint is not None
    assert preflight_error is not None
    assert preflight_error["ok"] is False
    assert "quantity" in preflight_error["error"]
    assert "count" in preflight_error["error"]


def test_order_preflight_suggests_unit_price_currency_field() -> None:
    agent = TripletexAccountingAgent(Settings())
    execution_state = ExecutionState(inspected_schemas={"Order"})

    preflight_error, endpoint = agent._validate_tripletex_request(
        {
            "method": "POST",
            "path": "/order",
            "json_body": {
                "customer": {"id": 1},
                "orderLines": [
                    {
                        "product": {"id": 2},
                        "count": 1,
                        "unitPrice": 100,
                    }
                ],
            },
        },
        execution_state,
    )

    assert endpoint is not None
    assert preflight_error is not None
    assert preflight_error["ok"] is False
    assert "unitPrice" in preflight_error["error"]
    assert "unitPriceExcludingVatCurrency" in preflight_error["error"]
