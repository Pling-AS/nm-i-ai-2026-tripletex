Deploy on Cloud Run
Two of the three competition tasks — Tripletex and Astar Island — require you to host a public HTTPS endpoint that our validators call. Cloud Run is the easiest way to deploy one.
What is Cloud Run?
Cloud Run takes a Docker container and gives you a public HTTPS URL. You push your code, it handles scaling, TLS, and everything else. You only pay for actual requests (and with your GCP account, it's free).
Step 1: Write Your Endpoint
Here's a minimal FastAPI endpoint that matches the competition format:
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/solve")
async def solve(request: dict):
    prompt = request.get("prompt", "")
    credentials = request.get("tripletex_credentials", {})

    # Your AI agent logic here:
    # 1. Parse the prompt
    # 2. Call the Tripletex API using the provided credentials
    # 3. Complete the accounting task
 
    return {"status": "completed"}
Step 2: Create a Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
And a requirements.txt:
fastapi
uvicorn[standard]
requests

Step 3: Deploy
Open Cloud Shell and run:

# Clone your repo (or upload files via Cloud Shell Editor)

cd ~
git clone <your-repo-url>
cd your-project

# Deploy to Cloud Run (builds and deploys in one command)

gcloud run deploy my-agent \
  --source . \
  --region europe-north1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 300
That's it. Cloud Run builds the Docker image, deploys it, and gives you a URL like:
<https://my-agent-xxxxx-lz.a.run.app>

Step 4: Submit Your URL

Copy the Cloud Run URL
Go to the submission page for your task at app.ainm.no
Paste the URL and submit
Our validators will start calling your endpoint

Tips
Use europe-north1 Region
Deploy in europe-north1 (Finland) — same region as our validators. Lower latency = faster scoring.
gcloud run deploy my-agent --region europe-north1 ...
Handle Cold Starts
Cloud Run scales to zero when idle. The first request after idle may take a few seconds. To keep it warm:
gcloud run deploy my-agent --min-instances 1 ...
This keeps one instance always running — useful during active competition.
Increase Memory for LLMs
If you're calling external LLM APIs (like Vertex AI), the default 512 MB is fine. If you're running a local model, increase memory:
gcloud run deploy my-agent --memory 2Gi --cpu 2 ...
Update Your Deployment
After making changes, just run the same deploy command again:
gcloud run deploy my-agent --source . --region europe-north1 --allow-unauthenticated
Cloud Run builds and deploys a new revision automatically.
View Logs
Check what your endpoint is doing:
gcloud run services logs read my-agent --region europe-north1 --limit 50
Or view logs in the Cloud Console under your service → Logs tab.
Which Tasks Need Cloud Run?

Task
Submission type
Cloud Run?

Tripletex
HTTPS endpoint (/solve)
Yes

Astar Island
HTTPS endpoint (/solve)
Yes

NorgesGruppen Data
Code upload (.zip)
No
