"""Deterministic knowledge base for the Tripletex executor.

Contains field-level validation rules (accumulated from real competition runs)
and compact successful trace examples (from best runs per task type).

These are injected into the execution brief BEFORE execution starts, so the
executor has complete tactical context without needing runtime API exploration.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# FIELD_RULES: endpoint → list of validation gotchas
#
# Keyed by "METHOD /path". Selected based on planned endpoints.
# Each rule should be a concrete, actionable instruction — not vague guidance.
# ---------------------------------------------------------------------------

FIELD_RULES: dict[str, list[str]] = {
    # ---- Voucher / Posting ----
    "POST /ledger/voucher": [
        "ALWAYS set BOTH amountGross AND amountGrossCurrency to the SAME value on every posting.",
        "Do NOT include postings with row=0 — row 0 is system-generated and will cause a 422.",
        "AUTO-SPLIT RULES: ONLY vatType id=3 (25% utgående) triggers auto-split. vatType id=1 does NOT auto-split.",
        "For supplier costs: use TWO postings — (1) DEBIT expense account with vatType=3 and amountGross=GROSS amount, (2) CREDIT account 2400 with vatType=0 and amountGross=NEGATIVE GROSS amount with supplier ref. Tripletex auto-splits the debit into net+VAT. A SINGLE debit posting with vatType=3 causes 422 on many accounts. ALWAYS include the credit to 2400. Ignore legalVatTypes — vatType=3 works for auto-split even when not listed.",
        "For vatType=0: include BOTH debit AND credit postings manually. They must sum to 0.",
        "If you get 422 'postings don't sum to 0': switch to vatType=3 with single debit posting, OR add both debit+credit with vatType=0.",
        "NEVER use vatType=1 — it does NOT auto-generate balancing entries and will always cause 422 with single posting.",
        "Do NOT include currency on postings — let it default. Including currency={} causes 'factor must be >= 1' error.",
        'For supplier invoices, set supplier={"id": supplier_id} on the debit posting.',
        'For ANY posting on account 2400 (Leverandørgjeld), ALWAYS include supplier={"id": supplier_id}.',
        'For ANY posting on account 1500 (Kundefordringer), ALWAYS include customer={"id": customer_id}.',
        "Month-end depreciation: debit a 6000-series depreciation expense (e.g., 6010), credit accumulated depreciation (typically 1039/1049/1059) — do NOT credit 1209.",
        "Month-end payroll accrual: derive amount from actual salary transactions/postings in period; NEVER invent a payroll amount.",
        'Resolve account numbers: GET /ledger/account with params={"number": XXXX}, then use account={"id": resolved_id}. NEVER use account={"number": XXXX} in postings.',
    ],
    "PUT /ledger/voucher/{id}/:reverse": [
        "Method is PUT, not POST.",
        "date is a QUERY PARAMETER: PUT /ledger/voucher/{id}/:reverse?date=YYYY-MM-DD",
        "Do NOT send date in the request body.",
    ],
    # ---- Employee ----
    "POST /employee": [
        "ALWAYS search first: GET /employee?email=<email> — the sandbox often has pre-existing employees. Reuse if found.",
        "Only POST /employee if GET returns 0 results.",
        'When POSTing, ALWAYS include department={"id": dept_id}. Create department first if needed.',
        "userType should be 'NO_ACCESS' (safest default — avoids 422 errors). Only use 'STANDARD' if email login is explicitly needed.",
        "startDate is a TOP-LEVEL field on the employee object. Do NOT put it inside an employments array.",
        "dateOfBirth is a TOP-LEVEL field. Format: 'YYYY-MM-DD'.",
        "jobTitle is set via PUT /employee/{id} AFTER creation — it cannot be set in the initial POST.",
        "Do NOT include an 'employments' array in POST /employee — employment details go via separate POST /employee/employment endpoint.",
        "For occupation codes (STYRK/yrkeskode): POST /employee/employment first, then GET /employee/employment/occupationCode?nameAndCode=XXXX to find the code, then PUT /employee/employment/details to set it.",
    ],
    "PUT /employee/{id}": [
        "For update tasks, GET /employee with fields=* first so you have id and version.",
        "PUT /employee/{id} MUST include id and version from the GET response.",
        "Only send the fields the prompt explicitly asks to change.",
        "If the prompt asks for admin/access privileges, use grant_employee_entitlements instead of guessing employee fields.",
    ],
    "POST /employee/employment/details": [
        "Use 'annualSalary' NOT 'salary' for the salary field.",
        "Use 'percentageOfFullTimeEquivalent' for stillingsprosent (e.g., 80.0 for 80%).",
        "For employment type: use 'employmentType' with values like 'ORDINARY' (fast stilling), not 'FIXED_SALARY'.",
        "For shift work: use 'shiftWork' with values like 'NONE', not 'NOT_SHIFT_WORK'.",
    ],
    "GET /employee/employment/occupationCode": [
        'ALWAYS use params={"nameAndCode": "XXXX"} to filter by STYRK code. NEVER paginate through all codes.',
        'Example: GET /employee/employment/occupationCode with params={"nameAndCode": "3512"} to find STYRK 3512.',
        'If no results, try broader search: params={"nameAndCode": "351"}.',
    ],
    # ---- Salary / Payroll ----
    "POST /salary/transaction": [
        "Each specification MUST have both 'rate' and 'count' fields. Do NOT use 'amount'.",
        "Example specification: {salaryType: {id: fastlonn_id}, rate: 59600, count: 1}",
        "Resolve salaryType IDs by NAME from GET /salary/type (e.g., 'Fastlønn', 'Timelønn', 'Overtid', 'Bonus', 'Feriepenger') — NEVER by list position/index.",
        "Match salary type names case-insensitively and by exact semantic meaning (base salary uses Fastlønn, hourly uses Timelønn, etc.).",
        "Employee MUST have an active employment record covering the salary period.",
        "If you get 'ikke registrert med et arbeidsforhold': create employment first via POST /employee/employment.",
        "Employment requires a division. Create one with POST /division if none exists.",
        "Employee MUST have dateOfBirth set before creating employment.",
    ],
    "POST /employee/employment": [
        "Employee MUST have dateOfBirth set. If missing, PUT /employee/{id} with dateOfBirth first.",
        "Requires division: {id: division_id}. Create division first if GET /division returns empty.",
        "Set startDate to '2026-01-01' or earlier than the salary period.",
        "Set isMainEmployer=true and taxDeductionCode='loennFraHovedarbeidsgiver'.",
    ],
    "POST /division": [
        "Requires startDate, municipalityDate, municipality, name, and organizationNumber.",
        "Get municipality first: GET /municipality/query?query=Oslo&from=0&count=1.",
        "Example body: {name: 'Hovedavdeling', organizationNumber: '999999999', startDate: '2026-01-01', municipalityDate: '2026-01-01', municipality: {id: mun_id}}",
    ],
    # ---- Travel Expense ----
    "POST /travelExpense/cost": [
        "MUST use amountCurrencyIncVat — NOT amount or amountExcludingVat.",
        "For 'Middag representasjon': use 'Representasjon - ikke fradragsb.' cost category (non-deductible).",
        "In Norway, representation/entertainment expenses are NEVER VAT-deductible.",
        "MUST include costCategory, paymentType, date, and travelExpense refs.",
    ],
    # ---- Customer / Supplier ----
    "POST /customer": [
        "When email is provided, ALWAYS set BOTH email AND invoiceEmail to the same value.",
        "Do NOT send isCustomer (it's readOnly and auto-set).",
        "For addresses, use postalAddress={addressLine1, postalCode, city, country={id: 161}} for Norway.",
    ],
    "POST /supplier": [
        "When email is provided, ALWAYS set BOTH email AND invoiceEmail to the same value.",
        "Do NOT send isSupplier (it's readOnly).",
        "ALWAYS include phoneNumber if provided in the prompt — scoring checks phone.",
        "ALWAYS include postalAddress with addressLine1, postalCode, city, country={id: 161} when address info is provided.",
    ],
    # ---- Product ----
    "POST /product": [
        "ALWAYS search first if product_number is provided: GET /product?number=<number> — sandbox pre-seeds products.",
        "Only POST /product if GET returns 0 results.",
        "Use 'number' field for the product number (NOT productNumber).",
        'priceExcludingVat sets the base price. Also set vatType={"id": 3|5|6}.',
    ],
    # ---- Order / Orderline ----
    "POST /order": [
        'Requires customer={"id": customer_id}.',
        "Set orderDate and deliveryDate (use today_iso from execution_brief).",
    ],
    "POST /order/orderline": [
        'When a product was created, ALWAYS include product={"id": product_id} on the orderline.',
        "Use unitPriceExcludingVatCurrency for the price field.",
        'Set vatType={"id": 3} for 25%, {"id": 5} for 15%, {"id": 6} for 0%.',
        "For multiple lines, prefer POST /order/orderline/list for efficiency.",
    ],
    # ---- Invoice ----
    "POST /invoice": [
        'Pass orders=[{"id": order_id}] to link order.',
        "Set invoiceDate and invoiceDueDate.",
        "Bank account 1920 MUST be configured first (see bank account setup pattern).",
    ],
    "PUT /invoice/{id}/:payment": [
        "Method is PUT, not POST.",
        "ALL params are QUERY PARAMETERS — NOT in body: PUT /invoice/{id}/:payment?paymentDate=YYYY-MM-DD&paymentTypeId=XXXXX&paidAmount=YYYYY",
        "Do NOT send a JSON body — send empty body or no body. ALL data goes in the URL query string.",
        "Get paymentTypeId first via GET /invoice/paymentType — use the first result's id.",
        "paidAmount is the full payment amount (e.g., 47200). For foreign currency, use paidAmountCurrency instead.",
    ],
    "PUT /invoice/{id}/:createCreditNote": [
        "Method is PUT, not POST.",
        "date is a QUERY PARAMETER: PUT /invoice/{id}/:createCreditNote?date=YYYY-MM-DD",
        "Do NOT send date in the request body.",
    ],
    "PUT /invoice/{id}/:send": [
        "Use query parameter sendType=EMAIL.",
    ],
    # ---- Travel Expense ----
    "POST /travelExpense": [
        'Requires employee={"id": employee_id}.',
        "Set title, departureDateTime, returnDateTime.",
        "DateTime format: 'YYYY-MM-DDT08:00:00' (not just date).",
        'ALWAYS include project={"id": project_id} if prompt mentions a project.',
        "Paths must NOT include /v2/ prefix — use /travelExpense not /v2/travelExpense.",
    ],
    "POST /travelExpense/cost": [
        "Use amountCurrencyIncVat (NOT amount, NOT amountExcludingVat).",
        'costCategory={"id": ...} and paymentType={"id": ...} are REQUIRED.',
        "GET /travelExpense/costCategory and GET /travelExpense/paymentType first.",
        "For per-diem/diet, use POST /travelExpense/perDiemCompensation instead.",
    ],
    # ---- Project ----
    "POST /project": [
        'Requires customer={"id": customer_id} and projectManager={"id": employee_id}.',
        "startDate is REQUIRED — set to 2026-01-01 or earlier to allow timesheet entries on any date.",
        "If the prompt mentions a project number (prosjektnummer), set number=<project_number>.",
        "isInternal should be false unless explicitly stated as internal project.",
        "For project BUDGET/fixed-price amount, use field name fixedPrice (camelCase) — NEVER fixedprice.",
        "For fixed-price projects, set isFixedPrice=true and fixedPrice=<amount> in the same request.",
        "When prompt explicitly says fixed-price project, set projectCategory='FIXED_PRICE' (or matching enum value from schema if required).",
    ],
    "POST /project/projectActivity": [
        'Requires project={"id": project_id} and activity={"id": activity_id}.',
        "If activity already linked (409), ignore the error and proceed.",
    ],
    "POST /project/hourlyRates": [
        "If a default hourly rate already exists (409 Duplicate entry), GET /project/hourlyRates?projectId=... then PUT to update it.",
        "For fixed rate: hourlyRateModel='TYPE_FIXED_HOURLY_RATE', fixedRate=<rate>.",
    ],
    # ---- Department ----
    "POST /department": [
        "Requires name and departmentNumber.",
    ],
    # ---- Activity / Timesheet ----
    "POST /activity": [
        "If name already exists (422 'Navnet er i bruk'), use GET /activity?name=... and reuse.",
        "For project activities, set activityType='PROJECT_GENERAL_ACTIVITY'.",
    ],
    "POST /timesheet/entry": [
        "Requires project, activity, employee, date, hours.",
        "Only one entry per employee/date/activity/project combination.",
        "hours is a decimal number (e.g. 8.0, 7.5).",
    ],
    # ---- Forbidden Endpoints ----
    "POST /incomingInvoice": [
        "FORBIDDEN: /incomingInvoice is a BETA endpoint requiring special permissions. It ALWAYS returns 403.",
        "Use POST /ledger/voucher with supplier reference instead. See register_supplier_invoice playbook.",
    ],
    "POST /supplierInvoice": [
        "Try this endpoint FIRST for supplier invoices — the scoring system prefers it.",
        "If it returns 403, fall back to POST /ledger/voucher with supplier reference.",
        "Required fields: invoiceNumber, supplier, currency, orderLines with account/amount/vatType.",
    ],
    # ---- Year-End / Closing ----
    "GET /ledger/voucher": [
        "REQUIRES dateFrom and dateTo as query params. Without them you get 422.",
        "dateTo is EXCLUSIVE — if you want vouchers from March 22, use dateTo=2026-03-23 (next day).",
        "NEVER use dateFrom == dateTo — it always returns 422. Always set dateTo = dateFrom + 1 day minimum.",
        'Example for single day: params={"dateFrom": "2026-03-22", "dateTo": "2026-03-23"}',
        'Example for broad range: params={"dateFrom": "2026-01-01", "dateTo": "2026-12-31"}',
        "These are QUERY PARAMS via the params field — NOT json_body.",
    ],
    "GET /ledger/posting": [
        "Prefer GET /ledger/posting for ledger analysis; aggregate/group endpoints can return empty groups in sandbox.",
        "Use dateFrom and dateTo for date range filtering.",
        "Use count and from for pagination — ALWAYS paginate if fullResultSize > count.",
        "For expense analysis: filter accounts in range 4000-7999 (Norwegian expense accounts).",
        "Response may be truncated — check fullResultSize vs count and paginate if needed.",
        "NEVER send the exact same API call twice simultaneously — deduplicate identical method+path+params requests.",
    ],
    "GET /ledger/account": [
        'Use params={"number": XXXX} to resolve a specific account number to its ID.',
        "This is a QUERY PARAM via the params field — NOT json_body.",
        "Do NOT use /ledger/accountingPeriod — it does not exist.",
    ],
    # ---- Ledger Account ----
    "PUT /ledger/account/{id}": [
        "For bank account setup: isBankAccount=true, bankAccountNumber='12345678903', bankAccountCountry={\"id\": 161}.",
        "MUST include id and version from the GET response.",
    ],
    # ---- Custom Dimensions ----
    "POST /ledger/accountingDimension": [
        "Use POST /ledger/accountingDimensionName for creating named dimensions.",
        "Use POST /ledger/accountingDimensionValue for adding values to a dimension.",
        'Reference dimensions on postings via freeAccountingDimension1..10={"id": value_id}.',
    ],
}


# ---------------------------------------------------------------------------
# TASK_ENDPOINT_CHAINS: task_type → ordered list of endpoints the executor
# will likely need. Used for deterministic schema pre-fetching.
# ---------------------------------------------------------------------------

TASK_ENDPOINT_CHAINS: dict[str, list[str]] = {
    "create_customer": [
        "POST /customer",
    ],
    "create_supplier": [
        "POST /supplier",
    ],
    "create_employee": [
        "GET /employee",
        "POST /department",
        "POST /employee",
    ],
    "create_product": [
        "POST /product",
    ],
    "create_department": [
        "POST /department",
    ],
    "create_order": [
        "POST /customer",
        "POST /order",
        "POST /order/orderline",
    ],
    "create_invoice": [
        "GET /ledger/account",
        "PUT /ledger/account/{id}",
        "POST /customer",
        "GET /product",
        "POST /product",
        "POST /order",
        "POST /order/orderline",
        "POST /invoice",
        "PUT /invoice/{id}/:send",
        "PUT /invoice/{id}/:payment",
        "GET /invoice/paymentType",
    ],
    "create_credit_note": [
        "GET /ledger/account",
        "PUT /ledger/account/{id}",
        "POST /customer",
        "POST /product",
        "POST /order",
        "POST /order/orderline",
        "POST /invoice",
        "PUT /invoice/{id}/:createCreditNote",
    ],
    "register_payment": [
        "GET /ledger/account",
        "PUT /ledger/account/{id}",
        "POST /customer",
        "GET /product",
        "POST /product",
        "POST /order",
        "POST /order/orderline",
        "POST /invoice",
        "GET /invoice/paymentType",
        "PUT /invoice/{id}/:payment",
        "GET /ledger/voucher",
        "PUT /ledger/voucher/{id}/:reverse",
    ],
    "register_supplier_invoice": [
        "POST /supplier",
        "GET /ledger/account",
        "POST /ledger/voucher",
    ],
    "create_voucher": [
        "GET /salary/type",
        "GET /employee",
        "PUT /employee/{id}",
        "GET /division",
        "POST /division",
        "POST /employee/employment",
        "GET /employee/employment/occupationCode",
        "POST /employee/employment/details",
        "POST /salary/transaction",
        "GET /ledger/account",
        "POST /ledger/voucher",
        "POST /ledger/accountingDimensionName",
        "POST /ledger/accountingDimensionValue",
    ],
    "reverse_voucher": [
        "GET /ledger/voucher",
        "PUT /ledger/voucher/{id}/:reverse",
    ],
    "create_project": [
        "POST /customer",
        "GET /employee",
        "POST /department",
        "POST /employee",
        "POST /project",
        "POST /project/projectActivity",
        "POST /project/hourlyRates",
    ],
    "register_timesheet": [
        "GET /employee",
        "POST /customer",
        "POST /project",
        "POST /project/projectActivity",
        "POST /activity",
        "POST /timesheet/entry",
        "GET /ledger/account",
        "PUT /ledger/account/{id}",
        "POST /order",
        "POST /order/orderline",
        "POST /invoice",
    ],
    "create_travel_expense": [
        "POST /department",
        "POST /employee",
        "GET /travelExpense/costCategory",
        "GET /travelExpense/paymentType",
        "POST /travelExpense",
        "POST /travelExpense/cost",
        "POST /travelExpense/perDiemCompensation",
    ],
    "delete_travel_expense": [
        "GET /travelExpense",
        "DELETE /travelExpense/{id}",
    ],
    "delete_invoice": [
        "GET /invoice",
        "DELETE /invoice/{id}",
    ],
    "update_employee": [
        "GET /employee",
        "PUT /employee/{id}",
    ],
    "update_customer": [
        "GET /customer",
        "PUT /customer/{id}",
    ],
    "update_supplier": [
        "GET /supplier",
        "PUT /supplier/{id}",
    ],
    "update_product": [
        "GET /product",
        "PUT /product/{id}",
    ],
    "update_order": [
        "GET /order",
        "PUT /order/{id}",
    ],
    "update_invoice": [
        "GET /invoice",
        "PUT /invoice/{id}",
    ],
    "update_contact": [
        "GET /contact",
        "PUT /contact/{id}",
    ],
    "create_invoice_timesheet": [
        "GET /ledger/account",
        "PUT /ledger/account/{id}",
        "POST /customer",
        "GET /employee",
        "POST /department",
        "POST /employee",
        "POST /project",
        "POST /activity",
        "POST /project/projectActivity",
        "POST /project/hourlyRates",
        "POST /timesheet/entry",
        "POST /order",
        "POST /order/orderline",
        "POST /invoice",
    ],
    "year_end_closing": [
        "GET /ledger/account",
        "GET /ledger/posting",
        "POST /ledger/voucher",
    ],
    "ledger_error_correction": [
        "GET /ledger/voucher",
        "PUT /ledger/voucher/{id}/:reverse",
        "GET /ledger/account",
        "POST /ledger/voucher",
    ],
    "bank_reconciliation": [
        "GET /ledger/account",
        "PUT /ledger/account/{id}",
        "POST /customer",
        "POST /supplier",
        "POST /order",
        "POST /order/orderline",
        "POST /invoice",
        "GET /invoice/paymentType",
        "PUT /invoice/{id}/:payment",
        "POST /ledger/voucher",
    ],
    "enable_module": [
        "GET /token/session/>whoAmI",
    ],
}


# ---------------------------------------------------------------------------
# SUCCESSFUL_TRACES: task_type → compact API call sequence from best run.
# Shows the executor the EXACT calls that achieved perfect/near-perfect scores.
# ---------------------------------------------------------------------------

SUCCESSFUL_TRACES: dict[str, dict] = {
    "create_customer": {
        "description": "Create customer with org number, address, and email (French prompt). 1 API call, 0 errors.",
        "calls": [
            {
                "step": 1,
                "method": "POST",
                "path": "/customer",
                "body_shape": {
                    "name": "str",
                    "organizationNumber": "str",
                    "email": "str",
                    "invoiceEmail": "str (same as email)",
                    "postalAddress": {
                        "addressLine1": "str",
                        "postalCode": "str",
                        "city": "str",
                        "country": {"id": 161},
                    },
                },
            },
        ],
    },
    "create_product": {
        "description": "Create product with number, price, and VAT. 2 API calls, 0 errors.",
        "calls": [
            {
                "step": 1,
                "method": "POST",
                "path": "/product",
                "body_shape": {
                    "name": "str",
                    "number": "product_number_str",
                    "priceExcludingVat": "number",
                    "vatType": {"id": 3},
                },
            },
            {
                "step": 2,
                "method": "GET",
                "path": "/product/{id}",
                "note": "Verify (optional, skip for efficiency)",
            },
        ],
    },
    "create_credit_note": {
        "description": "Full credit note chain. 8 API calls, 0 errors. PERFECT score.",
        "calls": [
            {
                "step": 1,
                "method": "GET",
                "path": "/ledger/account?number=1920",
                "note": "Get bank account id+version",
            },
            {
                "step": 2,
                "method": "POST",
                "path": "/customer",
                "body_shape": {"name": "str", "organizationNumber": "str"},
            },
            {
                "step": 3,
                "method": "POST",
                "path": "/product",
                "body_shape": {
                    "name": "str",
                    "priceExcludingVat": "number",
                    "vatType": {"id": 3},
                },
            },
            {
                "step": 4,
                "method": "PUT",
                "path": "/ledger/account/{id}",
                "body_shape": {
                    "id": "from_step1",
                    "version": "from_step1",
                    "number": 1920,
                    "name": "Bankinnskudd",
                    "isBankAccount": True,
                    "bankAccountNumber": "12345678903",
                    "bankAccountCountry": {"id": 161},
                },
            },
            {
                "step": 5,
                "method": "POST",
                "path": "/order",
                "body_shape": {
                    "customer": {"id": "from_step2"},
                    "orderDate": "today_iso",
                    "deliveryDate": "today_iso",
                },
            },
            {
                "step": 6,
                "method": "POST",
                "path": "/order/orderline",
                "body_shape": {
                    "order": {"id": "from_step5"},
                    "product": {"id": "from_step3"},
                    "description": "str",
                    "count": 1,
                    "unitPriceExcludingVatCurrency": "number",
                    "vatType": {"id": 3},
                },
            },
            {
                "step": 7,
                "method": "POST",
                "path": "/invoice",
                "body_shape": {
                    "orders": [{"id": "from_step5"}],
                    "invoiceDate": "today_iso",
                    "invoiceDueDate": "due_date_iso",
                },
            },
            {
                "step": 8,
                "method": "PUT",
                "path": "/invoice/{id}/:createCreditNote?date=today_iso",
                "note": "date is QUERY PARAM",
            },
        ],
    },
    "create_invoice": {
        "description": "Invoice with milestone/project. 9 API calls, 1 recoverable error.",
        "calls": [
            {"step": 1, "method": "GET", "path": "/ledger/account?number=1920"},
            {
                "step": 2,
                "method": "POST",
                "path": "/customer",
                "body_shape": {"name": "str", "organizationNumber": "str"},
            },
            {
                "step": 3,
                "method": "GET",
                "path": "/employee?email=...",
                "note": "Find existing employee for project manager",
            },
            {
                "step": 4,
                "method": "PUT",
                "path": "/ledger/account/{id}",
                "note": "Configure bank account",
            },
            {
                "step": 5,
                "method": "POST",
                "path": "/project",
                "body_shape": {
                    "name": "str",
                    "customer": {"id": "from_step2"},
                    "projectManager": {"id": "from_step3"},
                    "isFixedPrice": True,
                    "fixedPrice": "number",
                },
            },
            {
                "step": 6,
                "method": "POST",
                "path": "/order",
                "body_shape": {
                    "customer": {"id": "from_step2"},
                    "project": {"id": "from_step5"},
                    "orderDate": "today_iso",
                    "deliveryDate": "today_iso",
                },
            },
            {
                "step": 7,
                "method": "POST",
                "path": "/order/orderline",
                "body_shape": {
                    "order": {"id": "from_step6"},
                    "description": "str",
                    "count": 1,
                    "unitPriceExcludingVatCurrency": "number",
                    "vatType": {"id": 3},
                },
            },
            {
                "step": 8,
                "method": "POST",
                "path": "/invoice",
                "body_shape": {
                    "orders": [{"id": "from_step6"}],
                    "invoiceDate": "today_iso",
                    "invoiceDueDate": "due_date_iso",
                },
            },
        ],
    },
    "register_payment": {
        "description": "Register payment then reverse it. 11 API calls, 1 error.",
        "calls": [
            {"step": 1, "method": "GET", "path": "/ledger/account?number=1920"},
            {
                "step": 2,
                "method": "GET",
                "path": "/invoice/paymentType",
                "note": "Get payment type id",
            },
            {
                "step": 3,
                "method": "PUT",
                "path": "/ledger/account/{id}",
                "note": "Configure bank",
            },
            {"step": 4, "method": "POST", "path": "/customer"},
            {"step": 5, "method": "POST", "path": "/order"},
            {"step": 6, "method": "POST", "path": "/order/orderline"},
            {"step": 7, "method": "POST", "path": "/invoice"},
            {
                "step": 8,
                "method": "PUT",
                "path": "/invoice/{id}/:payment?paymentDate=...&paymentTypeId=...&paidAmount=...",
                "note": "All query params",
            },
            {
                "step": 9,
                "method": "GET",
                "path": "/ledger/voucher?dateFrom=...&dateTo=...",
                "note": "Find payment voucher for reversal",
            },
            {
                "step": 10,
                "method": "PUT",
                "path": "/ledger/voucher/{id}/:reverse?date=today_iso",
                "note": "date is QUERY PARAM",
            },
        ],
    },
    "create_voucher": {
        "description": "Custom dimension + voucher. 7 optimal calls (actual was 9 with 2 retries).",
        "calls": [
            {
                "step": 1,
                "method": "GET",
                "path": "/ledger/account?number=XXXX",
                "note": "Resolve expense account",
            },
            {
                "step": 2,
                "method": "GET",
                "path": "/ledger/account?number=1920",
                "note": "Resolve counter-account (bank)",
            },
            {
                "step": 3,
                "method": "POST",
                "path": "/ledger/accountingDimensionName",
                "note": "If custom dimension needed",
            },
            {
                "step": 4,
                "method": "POST",
                "path": "/ledger/accountingDimensionValue",
                "note": "Add each dimension value",
            },
            {
                "step": 5,
                "method": "POST",
                "path": "/ledger/voucher",
                "body_shape": {
                    "date": "today_iso",
                    "description": "str",
                    "postings": [
                        {
                            "account": {"id": "from_step1"},
                            "amountGross": "number",
                            "amountGrossCurrency": "same_as_amountGross",
                            "vatType": {"id": 0},
                            "description": "str",
                            "freeAccountingDimension1": {"id": "from_step4"},
                            "row": 1,
                        },
                        {
                            "account": {"id": "from_step2"},
                            "amountGross": "-number",
                            "amountGrossCurrency": "same_as_amountGross",
                            "vatType": {"id": 0},
                            "description": "counter-posting",
                            "row": 2,
                        },
                    ],
                },
            },
        ],
    },
    "create_travel_expense": {
        "description": "Travel expense with costs. 10 optimal calls (actual was 14 with 4 retries).",
        "calls": [
            {
                "step": 1,
                "method": "GET",
                "path": "/employee?email=...",
                "note": "Find or create employee",
            },
            {
                "step": 2,
                "method": "GET",
                "path": "/travelExpense/costCategory",
                "note": "Get available cost categories",
            },
            {
                "step": 3,
                "method": "GET",
                "path": "/travelExpense/paymentType",
                "note": "Get payment types",
            },
            {
                "step": 4,
                "method": "POST",
                "path": "/travelExpense",
                "body_shape": {
                    "employee": {"id": "from_step1"},
                    "title": "str",
                    "departureDateTime": "YYYY-MM-DDT08:00:00",
                    "returnDateTime": "YYYY-MM-DDT17:00:00",
                },
            },
            {
                "step": 5,
                "method": "POST",
                "path": "/travelExpense/cost",
                "body_shape": {
                    "travelExpense": {"id": "from_step4"},
                    "costCategory": {"id": "from_step2"},
                    "paymentType": {"id": "from_step3"},
                    "comments": "str",
                    "amountCurrencyIncVat": "number",
                    "date": "YYYY-MM-DD",
                },
                "note": "Repeat for each cost line. ALWAYS use amountCurrencyIncVat.",
            },
        ],
    },
    "create_employee": {
        "description": "Create employee — GET first to check if exists. 2-4 API calls.",
        "calls": [
            {
                "step": 1,
                "method": "GET",
                "path": "/employee?email=<email>",
                "note": "Check if employee already exists",
            },
            {
                "step": 2,
                "method": "POST",
                "path": "/department",
                "body_shape": {"name": "Avdeling", "departmentNumber": "1"},
                "note": "Only if needed",
            },
            {
                "step": 3,
                "method": "POST",
                "path": "/employee",
                "body_shape": {
                    "firstName": "str",
                    "lastName": "str",
                    "email": "str",
                    "dateOfBirth": "YYYY-MM-DD (TOP-LEVEL, not in employments)",
                    "startDate": "YYYY-MM-DD (TOP-LEVEL, not in employments)",
                    "userType": "NO_ACCESS",
                    "department": {"id": "from_step2"},
                },
                "note": "userType=NO_ACCESS avoids 422. startDate+dateOfBirth are TOP-LEVEL fields.",
            },
            {
                "step": 4,
                "method": "PUT",
                "path": "/employee/{id}",
                "body_shape": {
                    "id": "from_step3",
                    "version": "from_step3",
                    "jobTitle": "str (stillingstittel from prompt)",
                },
                "note": "Only if prompt specifies a job title. Must include id+version.",
            },
        ],
    },
    "create_project": {
        "description": "Create project with customer and project manager — all scored fields. 3-5 API calls.",
        "calls": [
            {
                "step": 1,
                "method": "POST",
                "path": "/customer",
                "body_shape": {"name": "str", "organizationNumber": "str"},
            },
            {
                "step": 2,
                "method": "GET",
                "path": "/employee?email=<email>",
                "note": "Check if employee exists first",
            },
            {
                "step": 3,
                "method": "POST",
                "path": "/department",
                "note": "Only if employee not found",
            },
            {
                "step": 4,
                "method": "POST",
                "path": "/employee",
                "note": "Only if GET returned 0 results",
            },
            {
                "step": 5,
                "method": "POST",
                "path": "/project",
                "body_shape": {
                    "name": "str",
                    "number": "str (project number if provided)",
                    "description": "str (if provided)",
                    "customer": {"id": "from_step1"},
                    "projectManager": {"id": "from_step2_or_4"},
                    "isInternal": False,
                    "startDate": "today_iso",
                    "isFixedPrice": "true if fixed-price project",
                    "fixedPrice": "number (fixed-price amount, camelCase)",
                    "projectCategory": "FIXED_PRICE (when fixed-price project)",
                },
                "note": "startDate is REQUIRED. name must EXACTLY match prompt. NEVER use fixedprice lowercase.",
            },
        ],
    },
    "register_timesheet": {
        "description": "Register hours on project activity then create invoice. 8-12 API calls.",
        "calls": [
            {
                "step": 1,
                "method": "GET",
                "path": "/employee?email=<email>",
                "note": "Find employee",
            },
            {
                "step": 2,
                "method": "POST",
                "path": "/customer",
                "body_shape": {"name": "str", "organizationNumber": "str"},
            },
            {
                "step": 3,
                "method": "POST",
                "path": "/project",
                "body_shape": {
                    "name": "str",
                    "customer": {"id": "from_step2"},
                    "projectManager": {"id": "from_step1"},
                    "startDate": "today_iso",
                },
            },
            {
                "step": 4,
                "method": "POST",
                "path": "/activity",
                "body_shape": {
                    "name": "str",
                    "activityType": "PROJECT_GENERAL_ACTIVITY",
                },
                "note": "If activity already exists (422), GET /activity?name=... and reuse",
            },
            {
                "step": 5,
                "method": "POST",
                "path": "/project/projectActivity",
                "body_shape": {
                    "project": {"id": "from_step3"},
                    "activity": {"id": "from_step4"},
                },
            },
            {
                "step": 6,
                "method": "POST",
                "path": "/timesheet/entry",
                "body_shape": {
                    "employee": {"id": "from_step1"},
                    "project": {"id": "from_step3"},
                    "activity": {"id": "from_step4"},
                    "date": "today_iso",
                    "hours": "number",
                },
                "note": "One entry per day. hours is a decimal number.",
            },
        ],
    },
    "create_order": {
        "description": "Create order with customer, products, and orderlines. 5-8 API calls.",
        "calls": [
            {
                "step": 1,
                "method": "GET",
                "path": "/ledger/account?number=1920",
                "note": "Get bank account for setup",
            },
            {
                "step": 2,
                "method": "PUT",
                "path": "/ledger/account/{id}",
                "note": "Configure bank account",
            },
            {
                "step": 3,
                "method": "POST",
                "path": "/customer",
                "body_shape": {
                    "name": "str",
                    "organizationNumber": "str",
                    "email": "str",
                    "invoiceEmail": "str (same as email)",
                },
            },
            {
                "step": 4,
                "method": "GET",
                "path": "/product?number=<num>",
                "note": "Check if product exists first",
            },
            {
                "step": 5,
                "method": "POST",
                "path": "/product",
                "body_shape": {
                    "name": "str",
                    "number": "str",
                    "priceExcludingVat": "number",
                    "vatType": {"id": 3},
                },
                "note": "Only if GET returned 0 results",
            },
            {
                "step": 6,
                "method": "POST",
                "path": "/order",
                "body_shape": {
                    "customer": {"id": "from_step3"},
                    "orderDate": "today_iso",
                    "deliveryDate": "today_iso",
                },
            },
            {
                "step": 7,
                "method": "POST",
                "path": "/order/orderline",
                "body_shape": {
                    "order": {"id": "from_step6"},
                    "product": {"id": "from_step4_or_5"},
                    "count": 1,
                    "unitPriceExcludingVatCurrency": "number",
                    "vatType": {"id": 3},
                },
                "note": "Repeat for each line item",
            },
        ],
    },
    "create_supplier": {
        "description": "Create supplier with org number and email. 1 API call.",
        "calls": [
            {
                "step": 1,
                "method": "POST",
                "path": "/supplier",
                "body_shape": {
                    "name": "str",
                    "organizationNumber": "str",
                    "email": "str",
                    "invoiceEmail": "str (same as email)",
                    "phoneNumber": "str if provided",
                    "postalAddress": {
                        "addressLine1": "str",
                        "postalCode": "str",
                        "city": "str",
                        "country": {"id": 161},
                    },
                },
            },
        ],
    },
    "update_employee": {
        "description": "Find employee then update fields. 2 API calls.",
        "calls": [
            {
                "step": 1,
                "method": "GET",
                "path": "/employee?email=<email>",
                "note": "Find by email or name. Use fields=* to get all fields including version",
            },
            {
                "step": 2,
                "method": "PUT",
                "path": "/employee/{id}",
                "body_shape": {
                    "id": "from_step1",
                    "version": "from_step1",
                    "firstName": "str",
                    "lastName": "str",
                },
                "note": "Include id and version from GET. Only change the fields the prompt asks to update.",
            },
        ],
    },
    "update_customer": {
        "description": "Find customer then update fields. 2 API calls.",
        "calls": [
            {
                "step": 1,
                "method": "GET",
                "path": "/customer?name=<name>",
                "note": "Find by name or org number. Use fields=* to get version",
            },
            {
                "step": 2,
                "method": "PUT",
                "path": "/customer/{id}",
                "body_shape": {
                    "id": "from_step1",
                    "version": "from_step1",
                    "name": "str",
                    "email": "str",
                    "invoiceEmail": "str",
                },
                "note": "Include id and version. Only update what prompt asks.",
            },
        ],
    },
    "update_supplier": {
        "description": "Find supplier then update fields. 2 API calls.",
        "calls": [
            {
                "step": 1,
                "method": "GET",
                "path": "/supplier?name=<name>",
                "note": "Find by name or org number",
            },
            {
                "step": 2,
                "method": "PUT",
                "path": "/supplier/{id}",
                "body_shape": {"id": "from_step1", "version": "from_step1"},
                "note": "Include id and version. Only update what prompt asks.",
            },
        ],
    },
    "update_contact": {
        "description": "Find contact then update fields. 2 API calls.",
        "calls": [
            {
                "step": 1,
                "method": "GET",
                "path": "/contact?email=<email>",
                "note": "Find by email or name",
            },
            {
                "step": 2,
                "method": "PUT",
                "path": "/contact/{id}",
                "body_shape": {"id": "from_step1", "version": "from_step1"},
                "note": "Include id and version. Only update what prompt asks.",
            },
        ],
    },
    "delete_invoice": {
        "description": "Find invoice then delete. 2 API calls.",
        "calls": [
            {
                "step": 1,
                "method": "GET",
                "path": "/invoice?invoiceDateFrom=<date>&invoiceDateTo=<date>",
                "note": "Find invoice by date, customer, or number",
            },
            {
                "step": 2,
                "method": "DELETE",
                "path": "/invoice/{id}",
                "note": "Delete the found invoice",
            },
        ],
    },
    "delete_travel_expense": {
        "description": "Find travel expense then delete. 2 API calls.",
        "calls": [
            {
                "step": 1,
                "method": "GET",
                "path": "/travelExpense?employeeId=<id>",
                "note": "Find by employee or date range",
            },
            {
                "step": 2,
                "method": "DELETE",
                "path": "/travelExpense/{id}",
                "note": "Delete the found travel expense",
            },
        ],
    },
    "reverse_voucher": {
        "description": "Find voucher then reverse it. 2 API calls.",
        "calls": [
            {
                "step": 1,
                "method": "GET",
                "path": "/ledger/voucher?dateFrom=<date>&dateTo=<date>",
                "note": "Find by date range. Use broad range.",
            },
            {
                "step": 2,
                "method": "PUT",
                "path": "/ledger/voucher/{id}/:reverse?date=today_iso",
                "note": "Method is PUT. date is QUERY PARAM.",
            },
        ],
    },
    "enable_module": {
        "description": "Enable a Tripletex module. 1-2 API calls.",
        "calls": [
            {
                "step": 1,
                "method": "GET",
                "path": "/token/session/>whoAmI",
                "note": "Get company and employee context",
            },
            {
                "step": 2,
                "method": "PUT",
                "path": "/company/settings/altinn",
                "note": "Enable relevant module via settings endpoint. Check prompt for which module.",
            },
        ],
    },
    "year_end_closing": {
        "description": "Year-end closing vouchers. 3-6 API calls. Use account numbers from prompt directly.",
        "calls": [
            {
                "step": 1,
                "method": "GET",
                "path": "/ledger/account?number=XXXX",
                "note": "Resolve each account number from prompt to get ID. Repeat for each account.",
            },
            {
                "step": 2,
                "method": "POST",
                "path": "/ledger/voucher",
                "body_shape": {
                    "date": "YYYY-12-31 (year-end date from prompt)",
                    "description": "Depreciation / Tax provision / Result transfer",
                    "postings": [
                        {
                            "account": {"id": "from_step1"},
                            "amountGross": "number",
                            "amountGrossCurrency": "same_as_amountGross",
                            "vatType": {"id": 0},
                            "row": 1,
                        },
                        {
                            "account": {"id": "from_step1_other"},
                            "amountGross": "-number",
                            "amountGrossCurrency": "same_as_amountGross",
                            "vatType": {"id": 0},
                            "row": 2,
                        },
                    ],
                },
                "note": "ALWAYS vatType 0. ALWAYS both amountGross AND amountGrossCurrency. Repeat POST for each closing entry.",
            },
        ],
    },
    "ledger_error_correction": {
        "description": "Reverse wrong voucher then rebook correct one. 3-4 API calls.",
        "calls": [
            {
                "step": 1,
                "method": "GET",
                "path": "/ledger/voucher?number=XXXX&dateFrom=YYYY-01-01&dateTo=YYYY-12-31",
                "note": "Find voucher by number from prompt",
            },
            {
                "step": 2,
                "method": "PUT",
                "path": "/ledger/voucher/{id}/:reverse?date=YYYY-MM-DD",
                "note": "Method is PUT. Date is QUERY PARAM.",
            },
            {
                "step": 3,
                "method": "GET",
                "path": "/ledger/account?number=XXXX",
                "note": "Resolve correct account number",
            },
            {
                "step": 4,
                "method": "POST",
                "path": "/ledger/voucher",
                "note": "Rebook with correct accounts/amounts",
            },
        ],
    },
    "create_department": {
        "description": "Create department. 1 API call.",
        "calls": [
            {
                "step": 1,
                "method": "POST",
                "path": "/department",
                "body_shape": {"name": "str", "departmentNumber": "str or number"},
            },
        ],
    },
    "register_supplier_invoice": {
        "description": "Create supplier then register invoice via /ledger/voucher. 3 API calls, 0 errors. vatType=3 for 25% MVA.",
        "calls": [
            {
                "step": 1,
                "method": "POST",
                "path": "/supplier",
                "body_shape": {
                    "name": "str",
                    "organizationNumber": "str",
                    "email": "str",
                    "invoiceEmail": "str (same as email)",
                },
            },
            {
                "step": 2,
                "method": "GET",
                "path": "/ledger/account",
                "note": 'Resolve expense account: params={"number": <account_number>}. Use the params field, NOT path query string.',
            },
            {
                "step": 3,
                "method": "POST",
                "path": "/ledger/voucher",
                "body_shape": {
                    "date": "invoice_date or today_iso",
                    "description": "Supplier invoice - supplier_name",
                    "postings": [
                        {
                            "account": {"id": "from_step2"},
                            "amountGross": "total_incl_vat (GROSS, positive)",
                            "amountGrossCurrency": "same_as_amountGross",
                            "vatType": {
                                "id": 3,
                                "note": "25% MVA — NEVER use 0 for standard purchases",
                            },
                            "supplier": {"id": "from_step1"},
                            "description": "expense description",
                            "row": 1,
                        }
                    ],
                },
                "note": "ONE debit posting with vatType=3. Tripletex auto-generates VAT (2710) + AP credit (2400). If 422 'balance error', you used vatType=0 — change to vatType=3.",
            },
        ],
    },
}


def select_field_rules(planned_endpoints: list[str]) -> list[dict[str, object]]:
    """Select field rules relevant to the planned endpoints.

    Args:
        planned_endpoints: List of "METHOD /path" strings.

    Returns:
        List of {"endpoint": str, "rules": list[str]} dicts.
    """
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for endpoint_key in planned_endpoints:
        # Exact match first
        if endpoint_key in FIELD_RULES and endpoint_key not in seen:
            selected.append(
                {"endpoint": endpoint_key, "rules": FIELD_RULES[endpoint_key]}
            )
            seen.add(endpoint_key)
            continue
        # Try matching by path pattern (e.g., "PUT /ledger/account/123" → "PUT /ledger/account/{id}")
        parts = endpoint_key.split(" ", 1)
        if len(parts) == 2:
            method, path = parts
            for rule_key, rules in FIELD_RULES.items():
                if rule_key in seen:
                    continue
                rule_parts = rule_key.split(" ", 1)
                if len(rule_parts) == 2 and rule_parts[0] == method:
                    # Check if the path pattern matches (strip IDs)
                    rule_path = rule_parts[1]
                    if _path_matches(path, rule_path):
                        selected.append({"endpoint": rule_key, "rules": rules})
                        seen.add(rule_key)
    return selected


def select_successful_trace(task_type: str) -> dict | None:
    """Select a successful trace example for the given task type."""
    return SUCCESSFUL_TRACES.get(task_type)


def get_endpoint_chain(task_type: str) -> list[str]:
    """Get the expected endpoint chain for a task type."""
    return TASK_ENDPOINT_CHAINS.get(task_type, [])


def _path_matches(actual_path: str, pattern_path: str) -> bool:
    actual_segments = actual_path.strip("/").split("/")
    pattern_segments = pattern_path.strip("/").split("/")
    if len(actual_segments) != len(pattern_segments):
        return False
    for actual, pattern in zip(actual_segments, pattern_segments):
        if pattern.startswith("{") and pattern.endswith("}"):
            continue
        if actual != pattern:
            return False
    return True


DOMAIN_KNOWLEDGE: dict[str, object] = {
    "standard_accounts": {
        1209: {"name": "Akkumulerte avskrivninger", "type": "balance"},
        1230: {"name": "Kjøretøy", "type": "balance"},
        1240: {"name": "Inventar", "type": "balance"},
        1250: {"name": "Programvare", "type": "balance"},
        1700: {"name": "Forskuddsbetalt kostnad", "type": "balance"},
        1920: {"name": "Bankinnskudd", "type": "balance", "is_bank": True},
        2400: {"name": "Leverandørgjeld", "type": "balance"},
        2710: {"name": "Utgående merverdiavgift", "type": "balance"},
        2920: {"name": "Betalbar skatt", "type": "balance"},
        3000: {"name": "Salgsinntekt", "type": "result"},
        4000: {"name": "Varekostnad", "type": "result"},
        5000: {"name": "Lønn", "type": "result"},
        6010: {"name": "Avskrivning", "type": "result"},
        6300: {"name": "Leie lokale", "type": "result"},
        6340: {"name": "Lys, varme", "type": "result"},
        6900: {"name": "Annen driftskostnad", "type": "result"},
        7100: {"name": "Bilkostnad", "type": "result"},
        7140: {"name": "Reisekostnad", "type": "result"},
        7350: {"name": "Representasjon", "type": "result"},
        8700: {"name": "Skattekostnad", "type": "result"},
    },
    "vat_types": {
        0: {"name": "Ingen MVA", "percentage": 0, "auto_split": False},
        3: {"name": "Utgående MVA 25%", "percentage": 25, "auto_split": True},
        5: {"name": "Utgående MVA 15%", "percentage": 15, "auto_split": True},
        6: {"name": "MVA-fri", "percentage": 0, "auto_split": False},
    },
    "travel_expense_categories": {
        "representasjon_middag": {
            "category_keyword": "Representasjon - ikke fradragsb",
            "reason": "Norwegian dinner representation is non-deductible for both VAT and income tax",
        },
        "representasjon_annet": {
            "category_keyword": "Representasjon - ikke fradragsb",
            "reason": "All representation expenses default to non-deductible",
        },
        "overnatting": {
            "category_keyword": "Hotell",
            "reason": "Hotel/accommodation category",
        },
        "transport": {
            "category_keyword": "Fly",
            "reason": "Flight/transport category",
        },
    },
    "salary_types": {
        "base_salary_keywords": ["Fastlønn", "Fast månedslønn", "Månedslønn"],
        "bonus_keywords": ["Bonus", "Tillegg", "Overtid"],
        "specification_fields": {
            "required": ["salaryType", "rate", "count"],
            "note": "Use rate (amount per unit) and count (number of units), NOT amount field",
        },
    },
    "salary_prerequisites": {
        "chain": [
            "1. Employee must have dateOfBirth set",
            "2. Division must exist (POST /division with municipality)",
            "3. Employment must exist (POST /employee/employment with division)",
            "4. Employment details must exist (POST /employee/employment/details with occupationCode)",
            "5. Then POST /salary/transaction with payslips[].specifications[].rate + count",
        ],
        "municipality_query": "GET /municipality/query?query=Oslo&from=0&count=1",
        "default_division": {
            "name": "Hovedavdeling",
            "organizationNumber": "999999999",
        },
        "default_employment": {
            "isMainEmployer": True,
            "taxDeductionCode": "loennFraHovedarbeidsgiver",
        },
    },
}


def get_domain_knowledge() -> dict[str, object]:
    return DOMAIN_KNOWLEDGE
