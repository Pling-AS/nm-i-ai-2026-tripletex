# NM i AI 2026 - Submission Instructions

## ✅ Status Overview
- **Task 1 (Object Detection):** READY (Submission file verified)
- **Task 2 (Accounting Agent):** READY (Tests passed, Docker configured)
- **Task 3 (Astar Island):** COMPLETED (Score: 91.55)

---

## 📦 Task 1: NorgesGruppen (Object Detection)
**Action:** Manual Upload

1. Locate the file:
   ```
   task-1-norgesgruppen/submission_final.zip
   ```
   *(Verified: Contains 9 files including `classifier_v2.safetensors` and `detector_b.pt`)*

2. Go to [app.ainm.no](https://app.ainm.no).
3. Upload this file to the Task 1 submission area.

---

## 🤖 Task 2: Tripletex (Accounting Agent)
**Action:** Deploy & Submit URL

### 1. Pre-Deployment Check
Run tests to verify integrity (Optional, already passed):
```bash
cd task-2-tripletex && uv run pytest
# Expected: 39 passed
```

### 2. Deployment (Cloud Run / Render / Railway)
The repository is configured with a `Dockerfile` that uses `uv` and respects the `$PORT` environment variable.

**Push to Registry:**
If deploying to Google Cloud Run or similar, you must push the image:
```bash
# Example for Google Artifact Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/task2 .
```

**Environment Variables:**
- `OPENROUTER_API_KEY`: **(Required)** Your LLM API key.
- `APP_API_KEY`: (Optional) Secret token to protect your endpoint.

### 3. Submission
1. Deploy the `task-2-tripletex` directory.
2. Get your public URL: `https://annamae-subseptate-nonveraciously.ngrok-free.dev` (Active via ngrok)
3. **Smoke Test**: Visit `https://annamae-subseptate-nonveraciously.ngrok-free.dev/health` to confirm it returns `{"status": "ok"}`.
4. Submit this URL to [app.ainm.no](https://app.ainm.no) for Task 2.

---

## 🗺️ Task 3: Astar Island
**Action:** None

This task is verified complete and running autonomously.
- **Current Rank:** #11 (Approx. weighted score: 144.9)
- **Status:** AUTORUN active on GCP
- **Code:** `task-3-astar_island/`

<!-- Verified ready for submission: Sat Mar 21 2026 (Tests passed: 39/39) -->
<!-- Simulator Projection: Rank #11, best ROI is focusing on Tasks 1 & 2 now -->
<!-- Git Commit: 1b9896a0bfc045858fb48602f87739cc304c7d83 -->
<!-- Final check by Sisyphus-Junior: ALL SYSTEMS GO -->

