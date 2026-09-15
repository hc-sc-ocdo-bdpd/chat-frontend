# Foundry Chat

A self-hosted ChatGPT-style interface for the Microsoft Foundry / Azure OpenAI
Responses API.

## Features

- Streaming chat with reasoning summaries and tool activity
- Independent background responses across multiple chats
- In-progress and unread-response indicators in the chat sidebar
- Multiple Azure model deployments from one `.env` file
- Reasoning-effort and response-verbosity controls
- Azure Code Interpreter and web research
- File uploads and durable generated-file downloads
- Projects with reusable files and instructions
- Edit, regenerate, stop, branch, copy, and delete message workflows
- Markdown, code blocks, links, citations, and rendered math
- Local conversation history in SQLite

## Requirements

Choose either:

- Docker Desktop, recommended, or
- Python 3.10 or newer

You also need an Azure OpenAI or Microsoft Foundry resource with at least one
deployment that supports the Responses API.

## Configure Azure

Copy the example environment file:

Windows CMD:

```cmd
copy .env.example .env
```

macOS or Linux:

```bash
cp .env.example .env
```

Open `.env` and fill in the shared Azure connection and at least one deployment:

```dotenv
AZURE_OPENAI_BASE_URL=https://YOUR-RESOURCE.openai.azure.com/openai/v1/
AZURE_OPENAI_API_KEY=YOUR_API_KEY
AZURE_OPENAI_DEPLOYMENT_1=YOUR_DEPLOYMENT_NAME
```

The deployment value must be the exact deployment name configured in Azure.

### Add more models

Deployments on the same Azure resource are discovered automatically. Add more
numbered variables, with optional friendly labels:

```dotenv
AZURE_OPENAI_DEPLOYMENT_1=team-sol
AZURE_OPENAI_LABEL_1=GPT-5.6 Sol

AZURE_OPENAI_DEPLOYMENT_2=team-mini
AZURE_OPENAI_LABEL_2=GPT-5.6 Mini

AZURE_OPENAI_DEPLOYMENT_3=model-router
AZURE_OPENAI_LABEL_3=Model Router
```

The numeric suffix controls dropdown order only. The deployment name is the
stable model identifier, so reordering entries does not break saved chats or
projects. No YAML editing is required to add a model.

By default, every listed deployment receives the standard reasoning,
verbosity, Code Interpreter, and web-research options. Optional per-model
overrides are available when a deployment supports a smaller feature set:

```dotenv
AZURE_OPENAI_REASONING_EFFORTS_2=auto,low,medium,high
AZURE_OPENAI_DEFAULT_REASONING_EFFORT_2=medium
AZURE_OPENAI_VERBOSITY_OPTIONS_2=low,medium,high
AZURE_OPENAI_DEFAULT_VERBOSITY_2=medium
AZURE_OPENAI_CODE_INTERPRETER_2=false
AZURE_OPENAI_WEB_SEARCH_2=false
AZURE_OPENAI_MAX_OUTPUT_TOKENS_2=16384
```

## Run with Docker, recommended

```cmd
docker compose up --build
```

Open:

```text
http://localhost:3000
```

Useful Docker commands:

```cmd
docker compose up -d
docker compose logs -f app
docker compose down
```

After pulling updated files:

```cmd
docker compose down
docker compose up --build --force-recreate
```

## Run locally with Python

The launcher creates `.venv`, installs dependencies when needed, and starts the
same app on port 3000.

Windows:

```cmd
run-local.cmd
```

macOS or Linux:

```bash
./run-local.sh
```

On the first run, if `.env` does not exist, the launcher creates it and tells
you which values are missing. Fill them in and run the same command again.

Stop the local server with `Ctrl+C`.

## Configuration

Model deployments come from `.env`. Shared application defaults and system
instructions are stored in:

```text
config/models.yaml
```

Most teams should not need to edit that file.

## Local data

Chats, projects, uploaded files, retained model-generated files, response-job
state, and the SQLite database are stored under:

```text
data/
```

Keep this folder when updating the app. The `.env`, `.venv`, and `data/` paths
are ignored by Git.

Generated Code Interpreter files are copied into `data/generated/` as soon as a
response finishes. Their chat messages point to the retained local copy rather
than the temporary Azure container, and retained files are uploaded again when
the model needs them in a later turn. If Azure rejects a retained file's
extension, such as `.ipynb`, the app gives Code Interpreter a ZIP containing the
original file. The user's saved download keeps its original name and format.
Files whose Azure containers expired before this version was installed cannot
be recovered retroactively.

Database schema updates are applied automatically at startup. A response that
was still running during an application restart is marked as interrupted, while
completed messages and retained files remain available.

## Privacy and security

Prompts and selected files are sent to the configured Azure resource. Web
research should remain disabled for sensitive material unless that data path is
approved by your organization.

Both launch methods listen only on the local machine at `127.0.0.1:3000`.
Do not expose the app publicly without authentication, authorization, rate
limits, and appropriate security controls.

## Troubleshooting

Docker logs:

```cmd
docker compose logs -f app
```

Local Python errors appear directly in the terminal running `run-local.cmd` or
`run-local.sh`.

Health check:

```cmd
curl http://localhost:3000/api/health
```

Expected response:

```json
{"status":"ok"}
```

Common problems:

- The Azure base URL, API key, or first deployment is blank
- The base URL does not end with `/openai/v1/`
- A deployment name does not exactly match Azure
- Duplicate deployment names are listed in `.env`
- A selected deployment does not support a requested tool or parameter
- Web research is blocked by an Azure subscription policy
