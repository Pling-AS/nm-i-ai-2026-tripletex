Services & Tools
A curated list of GCP services relevant to competing in NM i AI. You don't need all of these — pick what fits your approach.
Hosting Your Endpoint

Service
Use case
When to use

Cloud Run
Deploy containerized APIs
Tripletex & Astar Island tasks — this is the go-to

Compute Engine
Full VM (any OS)
Need GPU or persistent server

Recommendation: Start with Cloud Run. It's simpler and free with your account. Only use Compute Engine if you need a GPU or persistent background processes.
AI & Machine Learning

Service
Use case
When to use

Vertex AI
Managed ML platform
Access Gemini and other models via API

Model Garden
Pre-trained model catalog
Browse and deploy models (Gemini, Llama, Mistral)

AI Studio
Experiment with Gemini
Quick prototyping, prompt engineering

Using Vertex AI from Your Endpoint
Call Gemini from your Cloud Run endpoint:
import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(project="your-project-id", location="europe-north1")
model = GenerativeModel("gemini-2.0-flash")

response = model.generate_content("Parse this accounting task: ...")
print(response.text)
Install with: pip install google-cloud-aiplatform
Data & Storage

Service
Use case
When to use

Cloud Storage
File storage (buckets)
Store datasets, model weights, logs

Cloud SQL
Managed PostgreSQL/MySQL
Need a relational database

BigQuery
Data warehouse
Analyze large datasets with SQL

Development Tools

Tool
How to access
What it does

Cloud Shell
Console top-right icon
Free terminal with everything pre-installed

Cloud Shell Editor
"Open Editor" button
VS Code in the browser

Gemini Code Assist
Cloud Shell Editor sidebar
AI coding companion

Gemini CLI
gemini in Cloud Shell
AI assistant in the terminal

Cloud Build
Automatic with gcloud run deploy --source .
Builds your Docker images

Collaboration
Your @gcplab.me account also works with:

Gmail — communicate with teammates
Google Docs — shared documentation
Google Chat — team messaging
NotebookLM — AI-powered research notebook
