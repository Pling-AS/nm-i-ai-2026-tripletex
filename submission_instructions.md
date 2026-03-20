# NM i AI 2026 - Submission Instructions

## ✅ Status Overview
- **Task 1 (Object Detection):** READY (Submission file generated)
- **Task 2 (Accounting Agent):** READY (Code fixed, Docker configured)
- **Task 3 (Astar Island):** COMPLETED (Score: 91.55)

---

## 📦 Task 1: NorgesGruppen (Object Detection)
**Action:** Manual Upload

1. Locate the file:
   ```
   task-1-norgesgruppen/submission_varC_best_combo.zip
   ```
2. Go to [app.ainm.no](https://app.ainm.no).
3. Upload this file to the Task 1 submission area.

---

## 🤖 Task 2: Tripletex (Accounting Agent)
**Action:** Deploy & Submit URL

### 1. Deployment (Cloud Run / Render / Railway)
The repository is configured with a `Dockerfile` that uses `uv` and respects the `$PORT` environment variable.

**Environment Variables:**
The competition platform typically provides Tripletex credentials in the request body. However, you can set these as fallbacks or for local testing:

- `OPENROUTER_API_KEY`: **(Required)** Your LLM API key.
- `APP_API_KEY`: (Optional) Secret token to protect your endpoint (e.g. `mysecret`).
- `TRIPLETEX_SANDBOX_API_URL`: (Optional) Fallback URL if not in request (e.g. `https://api.tripletex.no/v2`).
- `TRIPLETEX_SANDBOX_API_SESSION_TOKEN`: (Optional) Fallback token if not in request.

**Local Test (Optional):**
```bash
cd task-2-tripletex
docker build -t task2 .
# Run with env vars
docker run -p 8080:8080 -e PORT=8080 -e OPENROUTER_API_KEY=your_key task2
```

### 2. Submission
1. Deploy the `task-2-tripletex` directory.
2. Get your public URL (e.g., `https://my-agent.onrender.com`).
3. **Smoke Test**: Visit `https://my-agent.onrender.com/health` to confirm it returns `{"status": "ok"}`.
4. Submit this URL to [app.ainm.no](https://app.ainm.no) for Task 2.
   - If you set `APP_API_KEY`, include the token in the submission form.

---

## 🗺️ Task 3: Astar Island
**Action:** None / Optional Re-run

This task is verified complete.
- **Score:** 91.55
- **Code:** `task-3-astar_island/`

<!-- Verified ready for submission: Sat Mar 21 00:33:57 CET 2026 -->
