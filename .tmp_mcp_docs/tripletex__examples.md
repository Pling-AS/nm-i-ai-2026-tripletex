# Tripletex â Examples

## Minimal `/solve` Endpoint

```python
import base64
from pathlib import Path

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/solve")
async def solve(request: Request):
    body = await request.json()
    prompt = body["prompt"]
    files = body.get("files", [])
    creds = body["tripletex_credentials"]

    base_url = creds["base_url"]
    token = creds["session_token"]
    auth = ("0", token)

    for f in files:
        data = base64.b64decode(f["content_base64"])
        Path(f["filename"]).write_bytes(data)

    # TODO: Use an LLM to interpret the prompt and execute
    # the appropriate Tripletex API calls

    return JSONResponse({"status": "completed"})
```

Run with:

```bash
pip install fastapi uvicorn requests
uvicorn main:app --host 0.0.0.0 --port 8000
```

Expose locally via HTTPS for testing:

```bash
npx cloudflared tunnel --url http://localhost:8000
```

## Tripletex API Examples

### List employees

```python
resp = requests.get(
    f"{base_url}/employee",
    auth=auth,
    params={"fields": "id,firstName,lastName,email"}
)
employees = resp.json()["values"]
```

### Create a customer

```python
resp = requests.post(
    f"{base_url}/customer",
    auth=auth,
    json={
        "name": "Acme AS",
        "email": "post@acme.no",
        "isCustomer": True
    }
)
customer_id = resp.json()["value"]["id"]
```

### Create an invoice

```python
today = "2026-03-03"
resp = requests.post(
    f"{base_url}/invoice",
    auth=auth,
    json={
        "invoiceDate": today,
        "invoiceDueDate": today,
        "customer": {"id": customer_id},
        "orders": [{"id": order_id}]
    }
)
```

### Search for a specific entity

```python
resp = requests.get(
    f"{base_url}/customer",
    auth=auth,
    params={
        "name": "Acme",
        "fields": "id,name,email",
        "count": 10
    }
)
matches = resp.json()["values"]
```

## Building an Effective Agent

1. **Parse the prompt** â Use an LLM to extract the task type, entity names, field values, and relationships from the Norwegian prompt
2. **Handle files** â Some tasks include PDFs with invoices, contracts, or expense reports. Decode from base64 and extract relevant data
3. **Map to API calls** â Determine which Tripletex endpoints to call and in what order. Some tasks require creating prerequisites first
4. **Verify your work** â After creating entities, query back to confirm they exist with correct values
5. **Handle errors** â Tripletex returns detailed error messages. Parse them to retry with corrections

## Common Task Patterns

| Pattern | Example | API Flow |
|---------|---------|----------|
| Create single entity | "Create employee Ola Nordmann" | POST /employee |
| Create with linking | "Create invoice for customer" | GET /customer â POST /order â POST /invoice |
| Modify existing | "Add phone to contact" | GET /customer â PUT /customer/{id} |
| Delete/reverse | "Delete travel expense" | GET /travelExpense â DELETE /travelExpense/{id} |
| Multi-step setup | "Register payment" | POST /customer â POST /invoice â POST /payment |

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| 401 Unauthorized | Wrong auth format | Use Basic Auth with username `0` and session token as password |
| 404 Not Found | Wrong endpoint path | Check the Tripletex v2 API docs for correct paths |
| 422 Validation Error | Missing required fields | Read error message â it specifies which fields are required |
| Empty `values` array | No results found | Check search parameters, try broader search |
| Timeout (5 min) | Agent too slow | Optimize API calls, reduce unnecessary requests |

## Tips

- The Tripletex sandbox starts empty â you may need to create prerequisites (customer, product) before creating invoices
- Use `?fields=*` to see all available fields on an entity
- Some tasks require enabling modules first (e.g., department accounting)
- Norwegian characters (Ã¦, Ã¸, Ã¥) work fine in API requests â send as UTF-8
- All API calls through the proxy are logged â use them for debugging in the submissions view
- Prompts come in 7 languages (nb, en, es, pt, nn, de, fr) â your agent should handle all of them

## Optimizing for Efficiency

Your score can go above 1.0 if you achieve perfect correctness with minimal API calls and zero errors. Higher-tier tasks have higher score ceilings (up to 6.0 for Tier 3). Tips:

- **Plan before calling** â Parse the prompt fully before making API calls. Understand what needs to be created/modified before starting
- **Avoid trial-and-error** â Every 4xx error (400, 404, 422) reduces your efficiency bonus. Validate inputs before sending
- **Minimize GET calls** â Don't fetch entities you don't need. If you created something, you already know its ID from the response
- **Batch where possible** â Some Tripletex endpoints accept lists. Use them instead of multiple individual calls
- **Read error messages** â If a call fails, the Tripletex error message tells you exactly what's wrong. Fix it in one retry, not several
