# Chat

A self-hosted chat frontend with streaming responses, file uploads and downloads,
projects, and saved conversations.

## Setup

Requires Docker Desktop or Python 3.10 or newer, plus an Azure OpenAI endpoint
and deployment that support the Responses API.

Copy `.env.example` to `.env` and fill in:

```dotenv
AZURE_OPENAI_BASE_URL=https://YOUR-RESOURCE.openai.azure.com/openai/v1/
AZURE_OPENAI_API_KEY=YOUR_API_KEY
AZURE_OPENAI_DEPLOYMENT_1=YOUR_DEPLOYMENT_NAME
```

Use the exact Azure deployment name. The URL must end with `/openai/v1/`.

For additional models, add `AZURE_OPENAI_DEPLOYMENT_2`, `_3`, and so on.
Optional labels use `AZURE_OPENAI_LABEL_1`, `_2`, etc. Tool support and model
options are documented in `.env.example`; shared defaults are in
`config/models.yaml`.

## Run

With Docker:

```sh
docker compose up --build -d
```

Open [localhost:3000](http://localhost:3000).
View logs with `docker compose logs -f app`; stop with `docker compose down`.

Alternatively, run `run-local.cmd` on Windows or `./run-local.sh` on macOS/Linux.
The launcher creates `.venv`, installs dependencies, and starts the server at the
same address. Stop it with `Ctrl+C`.

## Data

Chats and files are stored in `data/`. Back up this folder and keep `.env` private.
Use one server process per data folder. The app listens on localhost and has no
built-in authentication.

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Browser tests require Node.js and Playwright:

```sh
npm install --no-save --package-lock=false playwright
npx playwright install chromium
node --test tests/frontend_browser.test.cjs tests/live_browser.test.cjs
```
