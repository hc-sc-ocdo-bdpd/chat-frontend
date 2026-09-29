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

## Long-running Azure requests

Normal requests use background streaming and can resume from the saved Azure
response ID if the connection drops.

Very long requests have an extra safeguard for the period before a streaming
request receives its first `response.created` event. By default, requests with
`reasoning=max` or `max_output_tokens >= 65536` are created as non-streaming
background responses first. Azure returns the durable response ID, then the app
polls that saved response until it finishes. This avoids depending on one long
initial streaming POST before an ID exists.

The behavior can be adjusted in `.env`:

```dotenv
# auto, always, or never
APP_DURABLE_START_MODE=auto
APP_DURABLE_START_MAX_OUTPUT_TOKENS=65536
```

`auto` keeps normal token streaming for ordinary requests and uses durable
polling for the long-running cases above. `always` uses durable polling for all
background responses. `never` restores the original streaming startup behavior.
The app still does not retry an ambiguous generation POST automatically.

## Large file uploads

Large files that are only needed by Code Interpreter, such as ZIP archives, are
transparently split into small valid ZIP transport shards before upload. Each shard
uses the normal Azure Files API. The response request attaches all shard file IDs
and adds instructions for Code Interpreter to reconstruct the original file
byte-for-byte before using it. The default threshold is 32 MB and shard payload
size is 16 MB.

Optional `.env` overrides:

```dotenv
APP_LARGE_FILE_SHARD_THRESHOLD_MB=32
APP_LARGE_FILE_SHARD_MB=16
```

The shard set is cached in the existing provider-file field, so later requests reuse
the same Azure files instead of uploading them again. Smaller uploads still use the
normal Files API, with automatic SDK retries disabled so a provider error is visible
immediately.

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


Long-running durable responses now attach streaming only after the response ID is secured, with polling as a recovery fallback.
