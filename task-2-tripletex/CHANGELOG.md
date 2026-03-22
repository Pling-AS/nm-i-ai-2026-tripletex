# Tripletex Agent Changelog

## v2.0.0 (2026-03-22) — COMPILER ARCHITECTURE
- **Omni-Context Pre-fetch**: 8 parallel GETs at execution start (customers, products, employees, suppliers, departments, activities, payment types, bank account). All entities registered as $REF and injected into execution brief. LLM skips discovery calls entirely.
- **Dynamic API Blinding**: Salary/payroll tasks detected via keyword scan. /ledger/voucher hidden from LLM, salary endpoints injected instead (GET /salary/type, POST /salary/payslip, POST /wageRow, PUT /:approve).
- **aggregate_endpoint tool**: Push-down computation for ledger analysis. LLM requests aggregation (group_by, sum, sort, limit), middleware handles pagination and math, returns only top results. Fixes truncated data problem.

## v1.7.0 (2026-03-22)
- Fixed GET /ledger/voucher date range: dateTo is exclusive, never use dateFrom == dateTo
- Added GET /ledger/posting field rule: pagination reminder, expense account filter range
- Rewrote reverse_voucher playbook: full chain creation, correct VAT handling (excl. VAT → use vatType=3, paidAmount = amount×1.25)
- Fixed "sem IVA"/"excl. VAT" handling: use 25% MVA vatType=3, not 0% vatType=6

## v1.6.0 (2026-03-22) — REF RESOLUTION FIX
- **GET responses now register refs**: All entities from GET responses auto-registered (not just POST/PUT)
- **Account number refs**: GET /ledger/account registers both `account_5000` (by number) and `ledger_Lønn_til_ansatte` (by name)
- **Fuzzy $REF resolution**: If exact ref not found, tries substring match against registry keys
- LLM can now write `$REF:account_5000` after fetching accounts — middleware resolves it correctly

## v1.5.0 (2026-03-22) — COMPLETE ID ELIMINATION
- Fixed: `resource_id` field was not being scrubbed (only `id` was caught)
- Fixed: summary strings like "Created with id 12345" now scrubbed to "Created with id $REF:customer_Acme"
- String-level scrubbing: ALL string values in tool results have raw IDs replaced with $REF: names
- LLM now sees ZERO raw integer IDs in any field — id, resource_id, summary text, nested objects

## v1.4.0 (2026-03-22) — ID SCRUBBING
- **Raw IDs scrubbed from tool responses**: LLM never sees integer IDs, only $REF: aliases
- **$REF: mandatory**: Prompt changed from "optional" to "MANDATORY — do NOT use raw IDs"
- **Sandbox discovery scrubbed**: Pre-existing entities registered as refs before LLM sees them
- Flow: API returns {id: 12345} → middleware registers as $REF:customer_Acme → LLM sees {id: "$REF:customer_Acme"} → LLM writes $REF:customer_Acme → middleware resolves back to 12345

## v1.3.0 (2026-03-22)
- CRITICAL: vatType id=1 does NOT auto-split — only id=3 does. Banned vatType=1 for single-leg postings.
- CRITICAL: Do NOT include currency on postings — causes 'factor must be >= 1' error.
- CRITICAL: NEVER use account={"number": XXXX} in postings — always use account={"id": resolved_id}.
- Project budget field clarified: use fixedprice (not priceCeilingAmount or budget).
- Project startDate default: 2026-01-01 to allow timesheets on any date.

## v1.2.0 (2026-03-22) — SEMANTIC MIDDLEWARE COMPLETE
- **$TIME: tags**: `$TIME:today`, `$TIME:safe_project_start`, `$TIME:end_of_year` etc. — auto-resolved by middleware
- **$STYRK: fuzzy oracle**: `$STYRK:datateknikere` → code "3512" via local fuzzy match (60+ codes pre-baked, 0 API calls)
- **Anti-duplication cache**: Identical POST payloads return cached response instead of creating duplicates
- **Dashboard ref visibility**: Timeline shows $REF count on tool_start, ref count on tool_result
- Artistry agent review confirmed architecture is solid, recommended efficiency focus

## v1.1.0 (2026-03-22) — REFERENCE-BOUND PAYLOAD BUILDER
- **$REF: system**: LLM can write `{"id": "$REF:customer_Acme"}` instead of raw integer IDs
- **Entity Registry**: Every POST/PUT response auto-registers entity with semantic alias (e.g., `customer_Bergen_AS` → ID 99999)
- **Deep resolution**: $REF: resolved recursively in dicts, lists, nested objects — works everywhere in payloads and params
- **available_refs in results**: Tool results include the last 10 registered refs so LLM knows what's available
- **Version tracking**: Registry also stores `_version` refs for PUT operations requiring version field

## v1.0.0 (2026-03-22) — ARCHITECTURAL SHIFT
- **Semantic Middleware**: New ExecutionMiddleware layer between LLM tool calls and Tripletex API
- **VAT Type Pre-fetch**: GET /ledger/vatType at execution start, injected into brief with correct IDs
- **Account ID Resolver**: Middleware caches account GET responses, auto-resolves account numbers to IDs in voucher postings
- **Cascading Fallback**: POST /supplierInvoice automatically retried as POST /ledger/voucher on 403
- **Safe Date Anchoring**: Projects without startDate auto-set to 30 days ago (prevents timesheet date errors)
- **Response Caching**: All GET /ledger/account responses cached for ID lookups in subsequent POSTs

## v0.15.0 (2026-03-22)
- Un-banned /supplierInvoice: playbook now tries POST /supplierInvoice first, falls back to /ledger/voucher on 403
- year_end_closing: explicit ACCOUNT ID MAPPING instructions to prevent ID hallucination
- Timesheet dates: project startDate backdated to avoid "date before project start" errors
- Hourly rates: GET first (Tripletex auto-creates default), then PUT to update
- VAT type discovery: added GET /ledger/vatType guidance, removed /vat/type from forbidden
- Batch timesheet creation: prefer POST /timesheet/entry/list

## v0.14.0 (2026-03-22)
- Enforcer: GET requests with json_body auto-converted to query params (fixes ledger_error_correction 422s)
- Enforcer: PUT /:payment body params auto-moved to query string (fixes payment scoring failures)
- Field rule: GET /ledger/voucher requires dateFrom+dateTo query params
- Field rule: GET /employee/employment/occupationCode must use ?nameAndCode=XXXX filter (stops 3000+ code pagination)
- Field rule: POST /employee/employment/details uses annualSalary not salary, correct enum values
- year_end_closing: NEVER refuse to act, standard counteraccounts provided for common entries

## v0.13.0 (2026-03-22)
- Datalab Marker API integration for PDF extraction — preserves æøå and special characters
- PDF cache: SHA256 hash-based caching, skip Datalab API for previously-seen PDFs
- PDF files stored on disk in pdf_cache/pdfs/ for reference
- Extraction metadata (method, cached, char count) written to trace events
- Timeline shows PDF extraction details: method (datalab/pymupdf), cached status, char count

## v0.12.0 (2026-03-22)
- Fixed register_supplier_invoice: vatType=3 (25%) by default, NEVER vatType=0 for standard purchases
- Fixed voucher posting balance: explicit single-leg (vatType!=0) vs double-leg (vatType=0) rules
- Expanded VAT mapping: added vatType=0 warning, default-to-3 rule, posting balance implications
- Fixed successful trace for register_supplier_invoice to use vatType=3 with proper notes

## v0.11.0 (2026-03-22)
- Fixed query params embedded in path — auto-extracts ?key=val from path into params field
- Added explicit prompt rule: "NEVER embed query params in path, use params field"
- Added FORBIDDEN ENDPOINTS list: /incomingInvoice (403), /supplierInvoice, /ledger/accountingPeriod, /vat/type, /currency
- Added payroll/salary detection to create_voucher playbook
- Fixed employee: banned employments array in POST, added occupation code search guidance
- Strengthened PUT /invoice/:payment rules: ALL params as query string, NO body
- Fixed batch processing: two-phase wait (start 60s + finish 5min) prevents premature completion
- Batch UI: progress panel with per-run results, pause/cancel buttons, header indicator

## v0.10.0 (2026-03-21)
- Rewrote year_end_closing playbook: use account numbers from prompt directly, no endpoint probing
- Rewrote bank_reconciliation playbook: proper customer→invoice→payment chain
- Rewrote ledger_error_correction playbook: reverse→rebook flow
- Fixed employee creation: startDate/dateOfBirth as top-level fields, userType=NO_ACCESS, jobTitle via PUT
- Added FORBIDDEN ENDPOINTS list: /incomingInvoice (403), /supplierInvoice, /ledger/accountingPeriod
- Strengthened payment params: all query string, no body
- Added 9 efficiency rules: skip bank setup for simple tasks, no verification GETs
- Concurrent batch processing: asyncio.Semaphore worker pool, max 10 concurrent
- Batch UI: progress bar, worker count, per-run results, pause/cancel, header indicator
- Version tracking: agent_version in run metadata, badges in dashboard
- Syncthing-based run sharing: hostname filtering, enrichment skips remote runs
- File poller for auto-detecting externally-added JSONL files

## v0.9.0 (2026-03-21)
- Added post-mortem LLM analysis after scoring
- Added submission_id linking for deterministic score matching
- Added hostname tracking for multi-user setups
- Added tool_result trace events
- Fixed enrichment to only write final scores (not processing/scoring status)
- Fixed score matching: completed runs matched first, then errored
- Improved executor knowledge for year_end_closing and bank_reconciliation
- Added efficiency optimization rules
- Next.js dashboard with WebSocket real-time updates
- RunStore cache for fast JSONL parsing
