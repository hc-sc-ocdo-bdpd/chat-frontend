from __future__ import annotations

import logging
import mimetypes
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import AppConfig, EndpointConfig, ModelConfig, load_config
from .db import Database
from .jobs import (
    ACTIVE_GENERATION_STATUSES,
    TERMINAL_GENERATION_STATUSES,
    GenerationJob,
)
from .provider import (
    build_history_input,
    download_generated_file,
    extract_generated_files,
    extract_web_research,
    stream_response,
    supports_context_stuffing,
    upload_file,
    web_activity_labels,
)
from .schemas import (
    ConversationCreate,
    ConversationUpdate,
    MessageCreate,
    PartialAssistantCreate,
    ProjectCreate,
    ProjectFileUpdate,
    ProjectUpdate,
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Docker Compose injects values from .env directly. Local Python startup uses
# python-dotenv to load the same file, so both launch methods share one config.
load_dotenv(PROJECT_ROOT / ".env", override=False)

LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("foundry-chat")

STATIC_DIR = BASE_DIR / "static"
DATA_DIR = Path(os.getenv("APP_DATA_DIR", str(PROJECT_ROOT / "data")))
CONFIG_PATH = os.getenv(
    "APP_CONFIG",
    str(PROJECT_ROOT / "config" / "models.yaml"),
)
MAX_UPLOAD_BYTES = int(os.getenv("APP_MAX_UPLOAD_MB", "500")) * 1024 * 1024
UPLOAD_DIR = DATA_DIR / "uploads"
GENERATED_DIR = DATA_DIR / "generated"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

config: AppConfig = load_config(CONFIG_PATH)
db = Database(DATA_DIR / "app.db")
for interrupted_job in db.fail_interrupted_generation_jobs():
    if interrupted_job.get("assistant_message_id"):
        continue
    try:
        interrupted_message = db.add_message(
            interrupted_job["conversation_id"],
            "assistant",
            "Request interrupted because the application restarted before it finished.",
            {"error": True, "interrupted": True},
        )
        db.update_generation_job(
            interrupted_job["id"],
            assistant_message_id=interrupted_message["id"],
        )
    except KeyError:
        logger.warning(
            "Could not record interrupted generation %s",
            interrupted_job["id"],
        )
provider_reconciliation = db.reconcile_provider_ids(
    valid_models={
        endpoint.id: set(endpoint.models)
        for endpoint in config.endpoints.values()
    },
    default_endpoint_id=config.default_endpoint,
    default_model_id=config.default_model,
)
if any(provider_reconciliation.values()):
    logger.info(
        "Updated stored provider configuration for %s conversations and %s projects",
        provider_reconciliation["conversations"],
        provider_reconciliation["projects"],
    )

app = FastAPI(title=config.title)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_generation_lock = threading.RLock()
_generations: dict[str, GenerationJob] = {}
_active_generation_by_conversation: dict[str, str] = {}


def get_endpoint_and_model(
    endpoint_id: str | None, model_id: str | None
) -> tuple[EndpointConfig, ModelConfig]:
    # The application intentionally exposes one shared Azure connection. The
    # endpoint ID remains in stored metadata for backward compatibility only.
    endpoint = config.endpoints[config.default_endpoint]
    resolved_model_id = model_id or config.default_model
    model = endpoint.models.get(resolved_model_id)
    if model is None:
        raise HTTPException(status_code=400, detail="Unknown model")
    return endpoint, model


def normalize_message_provider(
    payload: MessageCreate,
) -> tuple[EndpointConfig, ModelConfig]:
    endpoint, model = get_endpoint_and_model(None, payload.model_id)
    payload.endpoint_id = endpoint.id
    payload.model_id = model.id
    return endpoint, model


def validate_message_options(payload: MessageCreate, model: ModelConfig) -> None:
    if payload.reasoning_effort not in model.reasoning_efforts:
        raise HTTPException(
            status_code=400,
            detail="Unsupported reasoning effort for this model",
        )
    if payload.verbosity not in model.verbosity_options:
        raise HTTPException(
            status_code=400,
            detail="Unsupported verbosity for this model",
        )
    if payload.use_code_interpreter and not model.supports_code_interpreter:
        raise HTTPException(
            status_code=400,
            detail="Code Interpreter is not enabled for this model",
        )
    if payload.use_web_search and not model.supports_web_search:
        raise HTTPException(
            status_code=400,
            detail="Web search is not enabled for this model",
        )


def stream_event_error(event: Any) -> str:
    try:
        raw = event.model_dump()
    except Exception:
        raw = {}

    error = raw.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or error)
    if error:
        return str(error)

    return str(
        raw.get("message")
        or getattr(event, "message", None)
        or "The model stream failed"
    )


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip()
    return name or "upload.bin"


def combined_instructions(
    conversation: dict[str, Any], payload: MessageCreate | None = None
) -> str:
    parts = [config.default_instructions.strip()]
    project_id = conversation.get("project_id")
    if project_id:
        try:
            project = db.get_project(project_id)
        except KeyError:
            project = None
        if project and project["instructions"].strip():
            parts.append(
                "Project instructions:\n" + project["instructions"].strip()
            )

    if payload and payload.use_web_search:
        parts.append(
            "Web research instructions:\n"
            "Perform thorough, multi-step web research before answering. "
            "Use several credible and independent sources, open relevant "
            "pages, compare evidence, resolve disagreements explicitly, and "
            "ground factual claims in citations returned by the web search "
            "tool. Prefer primary and official sources when available."
        )

    return "\n\n".join(part for part in parts if part)


def conversation_context(
    conversation: dict[str, Any],
    payload: MessageCreate,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    attachments: list[dict[str, Any]] = []
    for attachment_id in payload.attachment_ids:
        try:
            attachment = db.get_attachment(attachment_id)
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Attachment not found: {attachment_id}",
            )
        if attachment["conversation_id"] != conversation["id"]:
            raise HTTPException(status_code=400, detail="Attachment mismatch")
        attachments.append(attachment)

    project_files: list[dict[str, Any]] = []
    if conversation.get("project_id"):
        project_files = db.list_project_files(
            conversation["project_id"], active_only=True
        )

    retained_generated_files = db.list_generated_files(conversation["id"])
    return attachments, project_files, retained_generated_files


def prepare_provider_files(
    endpoint: EndpointConfig,
    attachments: list[dict[str, Any]],
    project_files: list[dict[str, Any]],
    generated_files: list[dict[str, Any]],
) -> tuple[list[str], list[tuple[dict[str, Any], str]]]:
    provider_file_ids: list[str] = []
    named_provider_files: list[tuple[dict[str, Any], str]] = []

    def add_file(file_row: dict[str, Any], provider_file_id: str) -> None:
        if provider_file_id not in provider_file_ids:
            provider_file_ids.append(provider_file_id)
        named_provider_files.append((file_row, provider_file_id))

    for attachment in attachments:
        provider_file_id = attachment["provider_files"].get(endpoint.id)
        if not provider_file_id:
            provider_file_id = upload_file(
                endpoint,
                UPLOAD_DIR / attachment["stored_name"],
                attachment["original_name"],
            )
            db.set_provider_file(
                attachment["id"], endpoint.id, provider_file_id
            )
        add_file(attachment, provider_file_id)

    for project_file in project_files:
        provider_file_id = project_file["provider_files"].get(endpoint.id)
        if not provider_file_id:
            provider_file_id = upload_file(
                endpoint,
                UPLOAD_DIR / project_file["stored_name"],
                project_file["original_name"],
            )
            db.set_project_provider_file(
                project_file["id"], endpoint.id, provider_file_id
            )
        add_file(project_file, provider_file_id)

    for generated_file in generated_files:
        provider_file_id = generated_file["provider_files"].get(endpoint.id)
        if not provider_file_id:
            provider_file_id = upload_file(
                endpoint,
                GENERATED_DIR / generated_file["stored_name"],
                generated_file["original_name"],
            )
            db.set_generated_provider_file(
                generated_file["id"], endpoint.id, provider_file_id
            )
        add_file(generated_file, provider_file_id)

    return provider_file_ids, named_provider_files


def build_input_payload(
    *,
    conversation: dict[str, Any],
    payload: MessageCreate,
    named_provider_files: list[tuple[dict[str, Any], str]],
) -> tuple[list[dict[str, Any]], str | None]:
    current_content: list[dict[str, str]] = [
        {"type": "input_text", "text": payload.content.strip()}
    ]
    current_content.extend(
        {"type": "input_file", "file_id": provider_file_id}
        for file_row, provider_file_id in named_provider_files
        if supports_context_stuffing(file_row["original_name"])
    )
    current_input: dict[str, Any] = {
        "role": "user",
        "content": current_content,
    }

    can_continue = bool(
        conversation.get("previous_response_id")
        and conversation.get("previous_endpoint_id") == payload.endpoint_id
        and conversation.get("previous_model_id") == payload.model_id
    )

    if can_continue:
        return [current_input], conversation["previous_response_id"]

    messages = db.list_messages(conversation["id"])
    prior_messages = messages[:-1]
    history: list[dict[str, Any]] = build_history_input(prior_messages)
    history.append(current_input)
    return history, None


def generated_file_metadata(
    conversation_id: str, file_row: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": file_row["id"],
        "filename": file_row["original_name"],
        "content_type": file_row.get("content_type"),
        "size_bytes": int(file_row["size_bytes"]),
        "url": (
            f"/api/conversations/{conversation_id}/generated-files/"
            f"{file_row['id']}/download"
        ),
    }


def persist_generated_files(
    endpoint: EndpointConfig,
    response: Any,
    conversation_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Download generated bytes before Azure's container can expire."""
    sources = extract_generated_files(response)
    if not sources:
        return [], []

    destination_dir = GENERATED_DIR / conversation_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    try:
        for source in sources:
            original_name = sanitize_filename(source["filename"])
            retained_id = str(uuid.uuid4())
            stored_name = str(
                Path(conversation_id) / f"{retained_id}_{original_name}"
            )
            destination = GENERATED_DIR / stored_name
            temporary = destination.with_name(destination.name + ".part")

            last_error: Exception | None = None
            upstream: Any | None = None
            for attempt in range(3):
                try:
                    upstream = download_generated_file(
                        endpoint,
                        source["container_id"],
                        source["file_id"],
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < 2:
                        time.sleep(0.25 * (2**attempt))

            if upstream is None:
                raise RuntimeError(
                    f"Could not retain generated file {original_name}: {last_error}"
                )

            content = bytes(upstream.content)
            with temporary.open("wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(destination)

            content_type = upstream.headers.get("content-type")
            if not content_type or content_type == "application/octet-stream":
                content_type = mimetypes.guess_type(original_name)[0]

            records.append(
                {
                    "id": retained_id,
                    "original_name": original_name,
                    "stored_name": stored_name,
                    "content_type": content_type or "application/octet-stream",
                    "size_bytes": len(content),
                    "source_endpoint_id": endpoint.id,
                    "source_container_id": source["container_id"],
                    "source_file_id": source["file_id"],
                    "provider_files": {},
                    "created_at": time.time(),
                }
            )
    except Exception:
        for record in records:
            (GENERATED_DIR / record["stored_name"]).unlink(missing_ok=True)
        for temporary in destination_dir.glob("*.part"):
            temporary.unlink(missing_ok=True)
        raise

    return records, [
        generated_file_metadata(conversation_id, record) for record in records
    ]


def remove_generated_file_bytes(files: list[dict[str, Any]]) -> None:
    parent_directories: set[Path] = set()
    for file_row in files:
        path = GENERATED_DIR / file_row["stored_name"]
        parent_directories.add(path.parent)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete retained file %s", file_row["id"])

    for directory in parent_directories:
        try:
            directory.rmdir()
        except OSError:
            pass


def active_generation_snapshot(
    conversation_id: str, *, include_output: bool = True
) -> dict[str, Any] | None:
    with _generation_lock:
        job_id = _active_generation_by_conversation.get(conversation_id)
        job = _generations.get(job_id) if job_id else None
    if job is not None and job.status in ACTIVE_GENERATION_STATUSES:
        return job.snapshot(include_output=include_output)

    row = db.get_active_generation_job(conversation_id)
    if row is None:
        return None
    return {
        "id": row["id"],
        "conversation_id": row["conversation_id"],
        "status": row["status"],
        "status_label": row["status_label"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "assistant_message_id": row["assistant_message_id"],
        "error": row["error"],
        "last_sequence": 0,
        **(
            {
                "assistant_text": "",
                "reasoning_summary": "",
                "activities": [],
            }
            if include_output
            else {}
        ),
    }


def ensure_conversation_idle(conversation_id: str) -> None:
    if active_generation_snapshot(conversation_id, include_output=False):
        raise HTTPException(
            status_code=409,
            detail="Wait for the active response to finish or stop it first",
        )


async def save_upload(upload: UploadFile) -> tuple[str, str, int, str | None]:
    original_name = sanitize_filename(upload.filename or "upload.bin")
    stored_name = f"{uuid.uuid4()}_{original_name}"
    destination = UPLOAD_DIR / stored_name
    size = 0
    content_type = upload.content_type

    try:
        with destination.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"{original_name} exceeds the "
                            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit"
                        ),
                    )
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    return original_name, stored_name, size, content_type


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/catalog")
def catalog() -> dict[str, Any]:
    endpoint = config.endpoints[config.default_endpoint]
    models = [
        {
            "id": model.id,
            "label": model.label,
            "reasoning_efforts": model.reasoning_efforts,
            "default_reasoning_effort": model.default_reasoning_effort,
            "verbosity_options": model.verbosity_options,
            "default_verbosity": model.default_verbosity,
            "supports_code_interpreter": model.supports_code_interpreter,
            "default_code_interpreter": model.default_code_interpreter,
            "supports_web_search": model.supports_web_search,
            "default_web_search": model.default_web_search,
            "default_research_depth": model.default_research_depth,
            "default_max_output_tokens": model.default_max_output_tokens,
        }
        for model in endpoint.models.values()
    ]
    return {
        "title": config.title,
        "default_model": config.default_model,
        "models": models,
    }


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


@app.get("/api/projects")
def list_projects() -> list[dict[str, Any]]:
    return db.list_projects()


@app.post("/api/projects")
def create_project(payload: ProjectCreate) -> dict[str, Any]:
    endpoint, model = get_endpoint_and_model(None, payload.default_model_id)
    return db.create_project(
        name=payload.name.strip(),
        instructions=payload.instructions.strip(),
        default_endpoint_id=endpoint.id,
        default_model_id=model.id,
    )


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    try:
        project = db.get_project(project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Project not found")
    return {
        **project,
        "files": db.list_project_files(project_id),
    }


@app.patch("/api/projects/{project_id}")
def update_project(
    project_id: str, payload: ProjectUpdate
) -> dict[str, Any]:
    try:
        current = db.get_project(project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Project not found")

    endpoint, model = get_endpoint_and_model(
        None, payload.default_model_id or current["default_model_id"]
    )

    updates = payload.model_dump(
        exclude_unset=True,
        exclude={"endpoint_id", "default_endpoint_id"},
    )
    updates["default_endpoint_id"] = endpoint.id
    updates["default_model_id"] = model.id
    if "name" in updates:
        updates["name"] = updates["name"].strip()
    if "instructions" in updates:
        updates["instructions"] = updates["instructions"].strip()
    db.update_project(project_id, **updates)
    return {
        **db.get_project(project_id),
        "files": db.list_project_files(project_id),
    }


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str) -> dict[str, Any]:
    try:
        files = db.list_project_files(project_id)
        db.delete_project(project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Project not found")

    for file_row in files:
        try:
            (UPLOAD_DIR / file_row["stored_name"]).unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete project file %s", file_row["id"])

    return {
        "deleted": True,
        "chats_preserved_as_general": True,
    }


@app.post("/api/projects/{project_id}/files")
async def upload_project_files(
    project_id: str,
    files: list[UploadFile] = File(...),
) -> list[dict[str, Any]]:
    try:
        db.get_project(project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Project not found")

    created = []
    for upload in files:
        original_name, stored_name, size, content_type = await save_upload(upload)
        created.append(
            db.create_project_file(
                project_id=project_id,
                original_name=original_name,
                stored_name=stored_name,
                content_type=content_type,
                size_bytes=size,
                is_active=True,
            )
        )
    return created


@app.patch("/api/projects/{project_id}/files/{file_id}")
def update_project_file(
    project_id: str,
    file_id: str,
    payload: ProjectFileUpdate,
) -> dict[str, Any]:
    try:
        file_row = db.get_project_file(file_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Project file not found")
    if file_row["project_id"] != project_id:
        raise HTTPException(status_code=404, detail="Project file not found")
    return db.update_project_file(file_id, is_active=payload.is_active)


@app.delete("/api/projects/{project_id}/files/{file_id}")
def delete_project_file(project_id: str, file_id: str) -> dict[str, bool]:
    try:
        file_row = db.get_project_file(file_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Project file not found")
    if file_row["project_id"] != project_id:
        raise HTTPException(status_code=404, detail="Project file not found")

    deleted = db.delete_project_file(file_id)
    try:
        (UPLOAD_DIR / deleted["stored_name"]).unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not delete project file %s", file_id)
    return {"deleted": True}


@app.get("/api/projects/{project_id}/files/{file_id}/download")
def download_project_file(project_id: str, file_id: str) -> FileResponse:
    try:
        file_row = db.get_project_file(file_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Project file not found")
    if file_row["project_id"] != project_id:
        raise HTTPException(status_code=404, detail="Project file not found")

    path = UPLOAD_DIR / file_row["stored_name"]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Local file is missing")
    return FileResponse(path, filename=file_row["original_name"])


# ---------------------------------------------------------------------------
# Conversations and message history
# ---------------------------------------------------------------------------


@app.get("/api/conversations")
def list_conversations() -> list[dict[str, Any]]:
    return db.list_conversations()


@app.post("/api/conversations")
def create_conversation(payload: ConversationCreate) -> dict[str, Any]:
    project = None
    if payload.project_id:
        try:
            project = db.get_project(payload.project_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Project not found")

    model_id = (
        payload.model_id
        or (project["default_model_id"] if project else None)
        or config.default_model
    )
    endpoint, model = get_endpoint_and_model(None, model_id)

    return db.create_conversation(
        endpoint.id,
        model.id,
        project_id=payload.project_id,
    )


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> dict[str, Any]:
    try:
        conversation = db.get_conversation(conversation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found")

    project = None
    if conversation.get("project_id"):
        try:
            project = db.get_project(conversation["project_id"])
        except KeyError:
            project = None

    return {
        **conversation,
        "messages": db.list_messages(conversation_id),
        "attachments": db.list_attachments(conversation_id),
        "generation": active_generation_snapshot(conversation_id),
        "project": project,
        "project_files": (
            db.list_project_files(conversation["project_id"])
            if conversation.get("project_id")
            else []
        ),
    }


@app.post("/api/conversations/{conversation_id}/read")
def mark_conversation_read(conversation_id: str) -> dict[str, bool]:
    try:
        db.mark_conversation_read(conversation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"read": True}


@app.patch("/api/conversations/{conversation_id}")
def update_conversation(
    conversation_id: str, payload: ConversationUpdate
) -> dict[str, Any]:
    try:
        db.get_conversation(conversation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found")

    updates = payload.model_dump(exclude_unset=True)
    if "title" in updates and updates["title"] is not None:
        updates["title"] = updates["title"].strip()
    if updates.get("project_id"):
        try:
            db.get_project(updates["project_id"])
        except KeyError:
            raise HTTPException(status_code=404, detail="Project not found")

    db.update_conversation(conversation_id, **updates)
    return db.get_conversation(conversation_id)


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict[str, bool]:
    ensure_conversation_idle(conversation_id)
    try:
        attachments = db.list_attachments(conversation_id)
        generated_files = db.list_generated_files(conversation_id)
        db.delete_conversation(conversation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found")

    for attachment in attachments:
        try:
            (UPLOAD_DIR / attachment["stored_name"]).unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete local attachment %s", attachment["id"])
    remove_generated_file_bytes(generated_files)
    return {"deleted": True}


@app.post("/api/conversations/{conversation_id}/branch/{message_id}")
def branch_conversation(
    conversation_id: str, message_id: str
) -> dict[str, Any]:
    ensure_conversation_idle(conversation_id)
    try:
        source = db.get_conversation(conversation_id)
        source_messages = db.list_messages_through(conversation_id, message_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation or message not found")

    branch = db.create_conversation(
        source["endpoint_id"],
        source["model_id"],
        title=f"{source['title']} (branch)",
        project_id=source.get("project_id"),
    )

    attachment_map: dict[str, str] = {}
    referenced_ids = {
        attachment_id
        for message in source_messages
        for attachment_id in message.get("metadata", {}).get("attachment_ids", [])
    }

    for attachment_id in referenced_ids:
        try:
            attachment = db.get_attachment(attachment_id)
        except KeyError:
            continue
        source_path = UPLOAD_DIR / attachment["stored_name"]
        copied_stored_name = f"{uuid.uuid4()}_{attachment['original_name']}"
        destination = UPLOAD_DIR / copied_stored_name
        if not source_path.exists():
            logger.warning(
                "Could not copy missing attachment %s while branching",
                attachment_id,
            )
            continue

        shutil.copy2(source_path, destination)
        copied = db.create_attachment(
            conversation_id=branch["id"],
            original_name=attachment["original_name"],
            stored_name=copied_stored_name,
            content_type=attachment["content_type"],
            size_bytes=attachment["size_bytes"],
            provider_files=attachment["provider_files"],
        )
        attachment_map[attachment_id] = copied["id"]

    for message in source_messages:
        metadata = dict(message.get("metadata", {}))
        if "attachment_ids" in metadata:
            metadata["attachment_ids"] = [
                attachment_map[item]
                for item in metadata["attachment_ids"]
                if item in attachment_map
            ]
        copied_generated_files: list[dict[str, Any]] = []
        copied_paths: list[Path] = []
        metadata.pop("generated_files", None)

        for source_file in db.list_message_generated_files(message["id"]):
            source_path = GENERATED_DIR / source_file["stored_name"]
            if not source_path.exists():
                logger.warning(
                    "Could not copy missing retained file %s while branching",
                    source_file["id"],
                )
                continue

            copied_id = str(uuid.uuid4())
            original_name = sanitize_filename(source_file["original_name"])
            copied_stored_name = str(
                Path(branch["id"]) / f"{copied_id}_{original_name}"
            )
            destination = GENERATED_DIR / copied_stored_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            copied_paths.append(destination)
            copied_generated_files.append(
                {
                    **source_file,
                    "id": copied_id,
                    "stored_name": copied_stored_name,
                    "created_at": source_file["created_at"],
                }
            )

        if copied_generated_files:
            metadata["generated_files"] = [
                generated_file_metadata(branch["id"], file_row)
                for file_row in copied_generated_files
            ]

        try:
            db.add_message(
                branch["id"],
                message["role"],
                message["content"],
                metadata,
                created_at=message["created_at"],
                generated_files=copied_generated_files,
            )
        except Exception:
            for copied_path in copied_paths:
                copied_path.unlink(missing_ok=True)
            raise

    last_message = source_messages[-1] if source_messages else None
    if last_message and last_message["role"] == "assistant":
        response_id = last_message.get("metadata", {}).get("response_id")
        endpoint_id = last_message.get("metadata", {}).get("endpoint_id")
        model_id = last_message.get("metadata", {}).get("model_id")
        if response_id and endpoint_id and model_id:
            db.update_conversation(
                branch["id"],
                previous_response_id=response_id,
                previous_endpoint_id=endpoint_id,
                previous_model_id=model_id,
            )

    return get_conversation(branch["id"])


@app.delete("/api/conversations/{conversation_id}/messages/{message_id}")
def delete_messages_from(
    conversation_id: str, message_id: str
) -> dict[str, Any]:
    ensure_conversation_idle(conversation_id)
    try:
        generated_files = db.list_generated_files_from_message(
            conversation_id, message_id
        )
        deleted_count = db.truncate_messages_from(conversation_id, message_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Message not found")
    remove_generated_file_bytes(generated_files)
    return {
        "deleted_count": deleted_count,
        "conversation": get_conversation(conversation_id),
    }


@app.post("/api/conversations/{conversation_id}/messages/partial")
def save_partial_message(
    conversation_id: str,
    payload: PartialAssistantCreate,
) -> dict[str, Any]:
    try:
        db.get_conversation(conversation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.clear_response_link(conversation_id)
    if not payload.content.strip():
        return {"saved": False}

    message = db.add_message(
        conversation_id,
        "assistant",
        payload.content,
        {
            "endpoint_id": config.default_endpoint,
            "model_id": payload.model_id or config.default_model,
            "use_web_search": payload.use_web_search,
            "research_depth": payload.research_depth,
            "reasoning_summary": payload.reasoning_summary,
            "activities": payload.activities,
            "duration_seconds": payload.duration_seconds,
            "stopped": True,
        },
    )
    return {"saved": True, "assistant_message": message}


@app.post("/api/conversations/{conversation_id}/attachments")
async def create_attachments(
    conversation_id: str,
    files: list[UploadFile] = File(...),
) -> list[dict[str, Any]]:
    try:
        db.get_conversation(conversation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found")

    created = []
    for upload in files:
        original_name, stored_name, size, content_type = await save_upload(upload)
        created.append(
            db.create_attachment(
                conversation_id=conversation_id,
                original_name=original_name,
                stored_name=stored_name,
                content_type=content_type,
                size_bytes=size,
            )
        )
    return created


# ---------------------------------------------------------------------------
# Model requests
# ---------------------------------------------------------------------------


def _finish_cancelled_generation(
    *,
    job: GenerationJob,
    endpoint: EndpointConfig,
    model: ModelConfig,
    payload: MessageCreate,
    assistant_text: str,
    reasoning_summary: str,
    activities: list[str],
    started_monotonic: float,
) -> None:
    duration_seconds = round(time.monotonic() - started_monotonic, 1)
    assistant_message = None
    if assistant_text.strip():
        assistant_message = db.add_message(
            job.conversation_id,
            "assistant",
            assistant_text,
            {
                "endpoint_id": endpoint.id,
                "model_id": model.id,
                "reasoning_summary": reasoning_summary,
                "activities": activities,
                "duration_seconds": duration_seconds,
                "web_search_enabled": payload.use_web_search,
                "research_depth": payload.research_depth,
                "stopped": True,
            },
        )

    db.clear_response_link(job.conversation_id)
    completed_at = time.time()
    db.update_generation_job(
        job.id,
        assistant_message_id=(
            assistant_message["id"] if assistant_message is not None else None
        ),
        status="cancelled",
        status_label="Stopped",
        completed_at=completed_at,
    )
    job.publish(
        {
            "type": "cancelled",
            "assistant_message": assistant_message,
            "duration_seconds": duration_seconds,
            "completed_at": completed_at,
        }
    )


def _run_generation_job(
    *,
    job: GenerationJob,
    conversation: dict[str, Any],
    payload: MessageCreate,
    endpoint: EndpointConfig,
    model: ModelConfig,
    attachments: list[dict[str, Any]],
    project_files: list[dict[str, Any]],
    retained_generated_files: list[dict[str, Any]],
) -> None:
    started_monotonic = time.monotonic()
    started_at = time.time()
    reasoning_summary = ""
    assistant_text = ""
    activities: list[str] = []
    final_response: Any | None = None
    provider_stream: Any | None = None

    def activity(label: str) -> None:
        if label in activities:
            return
        activities.append(label)
        job.publish({"type": "activity", "label": label})

    try:
        db.update_generation_job(
            job.id,
            status="running",
            status_label="Thinking",
            started_at=started_at,
        )
        job.publish(
            {
                "type": "started",
                "label": "Thinking",
                "user_message": job.user_message,
                "started_at": started_at,
                "project_file_count": len(project_files),
                "retained_file_count": len(retained_generated_files),
            }
        )

        if job.cancel_event.is_set():
            _finish_cancelled_generation(
                job=job,
                endpoint=endpoint,
                model=model,
                payload=payload,
                assistant_text=assistant_text,
                reasoning_summary=reasoning_summary,
                activities=activities,
                started_monotonic=started_monotonic,
            )
            return

        if attachments or project_files or retained_generated_files:
            db.update_generation_job(job.id, status_label="Preparing files")
            job.publish(
                {
                    "type": "status",
                    "label": "Preparing attached and retained files",
                }
            )

        provider_file_ids, named_provider_files = prepare_provider_files(
            endpoint,
            attachments,
            project_files,
            retained_generated_files,
        )
        input_payload, previous_response_id = build_input_payload(
            conversation=conversation,
            payload=payload,
            named_provider_files=named_provider_files,
        )

        if job.cancel_event.is_set():
            _finish_cancelled_generation(
                job=job,
                endpoint=endpoint,
                model=model,
                payload=payload,
                assistant_text=assistant_text,
                reasoning_summary=reasoning_summary,
                activities=activities,
                started_monotonic=started_monotonic,
            )
            return

        db.update_generation_job(job.id, status_label="Thinking")
        job.publish({"type": "status", "label": "Thinking"})

        provider_stream = stream_response(
            endpoint=endpoint,
            model=model,
            input_payload=input_payload,
            instructions=combined_instructions(conversation, payload),
            reasoning_effort=payload.reasoning_effort,
            verbosity=payload.verbosity,
            max_output_tokens=payload.max_output_tokens,
            use_code_interpreter=(
                payload.use_code_interpreter or bool(provider_file_ids)
            ),
            provider_file_ids=provider_file_ids,
            use_web_search=payload.use_web_search,
            research_depth=payload.research_depth,
            web_allowed_domains=payload.web_allowed_domains,
            web_blocked_domains=payload.web_blocked_domains,
            previous_response_id=previous_response_id,
        )
        job.set_cancel_callback(getattr(provider_stream, "close", None))

        for event in provider_stream:
            if job.cancel_event.is_set():
                break

            event_type = str(getattr(event, "type", ""))

            if event_type in {
                "response.reasoning_summary_text.delta",
                "response.reasoning_summary.delta",
            }:
                delta = str(getattr(event, "delta", "") or "")
                if delta:
                    reasoning_summary += delta
                    job.publish({"type": "reasoning_delta", "delta": delta})
                continue

            if event_type in {
                "response.reasoning_summary_text.done",
                "response.reasoning_summary.done",
            }:
                complete_text = str(getattr(event, "text", "") or "")
                if complete_text:
                    reasoning_summary = complete_text
                    job.publish(
                        {"type": "reasoning_done", "text": complete_text}
                    )
                continue

            if event_type == "response.code_interpreter_call.in_progress":
                activity("Starting Code Interpreter")
                continue
            if event_type == "response.code_interpreter_call_code.delta":
                activity("Writing Python")
                continue
            if event_type == "response.code_interpreter_call.interpreting":
                activity("Running Python")
                continue
            if event_type == "response.code_interpreter_call.completed":
                activity("Code Interpreter finished")
                continue
            if event_type == "response.web_search_call.in_progress":
                activity("Starting web search")
                continue
            if event_type == "response.web_search_call.searching":
                activity("Searching the web")
                continue
            if event_type == "response.web_search_call.completed":
                activity("Web search finished")
                continue

            if event_type == "response.output_item.done":
                item = getattr(event, "item", None)
                try:
                    item_raw = item.model_dump() if item is not None else {}
                except Exception:
                    item_raw = {}
                if item_raw.get("type") == "web_search_call":
                    action = item_raw.get("action") or {}
                    action_type = action.get("type")
                    if action_type == "search" and action.get("query"):
                        activity(f'Searched: {action["query"]}')
                    elif action_type == "open_page" and action.get("url"):
                        activity(f'Opened page: {action["url"]}')
                    elif action_type == "find_in_page" and action.get("pattern"):
                        activity(f'Found in page: {action["pattern"]}')
                continue

            if event_type == "response.output_text.delta":
                delta = str(getattr(event, "delta", "") or "")
                if delta:
                    assistant_text += delta
                    job.publish({"type": "output_delta", "delta": delta})
                continue

            if event_type == "response.completed":
                final_response = getattr(event, "response", None)
                continue

            if event_type in {
                "error",
                "response.failed",
                "response.incomplete",
            }:
                raise RuntimeError(stream_event_error(event))

        if job.cancel_event.is_set():
            _finish_cancelled_generation(
                job=job,
                endpoint=endpoint,
                model=model,
                payload=payload,
                assistant_text=assistant_text,
                reasoning_summary=reasoning_summary,
                activities=activities,
                started_monotonic=started_monotonic,
            )
            return

        if final_response is None:
            raise RuntimeError(
                "The stream ended without a completed response event"
            )

        if not assistant_text:
            assistant_text = (
                getattr(final_response, "output_text", None)
                or "(The model returned no text.)"
            )

        retained_records, downloads = persist_generated_files(
            endpoint, final_response, job.conversation_id
        )
        web_research = extract_web_research(final_response)
        for label in web_activity_labels(web_research):
            if label not in activities:
                activities.append(label)
        duration_seconds = round(time.monotonic() - started_monotonic, 1)
        response_id = str(getattr(final_response, "id", ""))

        try:
            assistant_message = db.add_message(
                job.conversation_id,
                "assistant",
                assistant_text,
                {
                    "endpoint_id": endpoint.id,
                    "model_id": model.id,
                    "response_id": response_id,
                    "generated_files": downloads,
                    "reasoning_summary": reasoning_summary,
                    "activities": activities,
                    "duration_seconds": duration_seconds,
                    "web_research": web_research,
                    "web_search_enabled": payload.use_web_search,
                    "research_depth": payload.research_depth,
                },
                generated_files=retained_records,
            )
        except Exception:
            remove_generated_file_bytes(retained_records)
            raise

        db.update_conversation(
            job.conversation_id,
            endpoint_id=endpoint.id,
            model_id=model.id,
            previous_response_id=response_id,
            previous_endpoint_id=endpoint.id,
            previous_model_id=model.id,
        )
        completed_at = time.time()
        db.update_generation_job(
            job.id,
            assistant_message_id=assistant_message["id"],
            status="completed",
            status_label="Completed",
            completed_at=completed_at,
        )
        job.publish(
            {
                "type": "done",
                "assistant_message": assistant_message,
                "generated_files": downloads,
                "response_id": response_id,
                "duration_seconds": duration_seconds,
                "web_research": web_research,
                "completed_at": completed_at,
            }
        )
    except Exception as exc:
        if job.cancel_event.is_set():
            try:
                _finish_cancelled_generation(
                    job=job,
                    endpoint=endpoint,
                    model=model,
                    payload=payload,
                    assistant_text=assistant_text,
                    reasoning_summary=reasoning_summary,
                    activities=activities,
                    started_monotonic=started_monotonic,
                )
            except Exception:
                logger.exception("Could not finalize cancelled generation")
            return

        logger.exception("Background provider request failed")
        error_message = db.add_message(
            job.conversation_id,
            "assistant",
            f"Request failed: {exc}",
            {"error": True},
        )
        db.clear_response_link(job.conversation_id)
        completed_at = time.time()
        db.update_generation_job(
            job.id,
            assistant_message_id=error_message["id"],
            status="failed",
            status_label="Failed",
            error=str(exc),
            completed_at=completed_at,
        )
        job.publish(
            {
                "type": "error",
                "message": str(exc),
                "assistant_message": error_message,
                "completed_at": completed_at,
            }
        )
    finally:
        job.set_cancel_callback(None)
        if provider_stream is not None:
            close = getattr(provider_stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("Could not close provider stream", exc_info=True)

        with _generation_lock:
            if _active_generation_by_conversation.get(job.conversation_id) == job.id:
                _active_generation_by_conversation.pop(job.conversation_id, None)


def _prune_generation_registry() -> None:
    cutoff = time.time() - 600
    with _generation_lock:
        expired = [
            job_id
            for job_id, job in _generations.items()
            if job.status in TERMINAL_GENERATION_STATUSES
            and job.completed_at is not None
            and float(job.completed_at) < cutoff
        ]
        for job_id in expired:
            _generations.pop(job_id, None)


def _start_generation_job(
    conversation_id: str, payload: MessageCreate
) -> tuple[GenerationJob, dict[str, Any]]:
    try:
        conversation = db.get_conversation(conversation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found")

    endpoint, model = normalize_message_provider(payload)
    validate_message_options(payload, model)
    attachments, project_files, retained_generated_files = conversation_context(
        conversation, payload
    )

    with _generation_lock:
        ensure_conversation_idle(conversation_id)
        user_message = db.add_message(
            conversation_id,
            "user",
            payload.content.strip(),
            {
                "attachment_ids": payload.attachment_ids,
                "project_file_ids": [item["id"] for item in project_files],
                "endpoint_id": endpoint.id,
                "model_id": model.id,
                "use_web_search": payload.use_web_search,
                "research_depth": payload.research_depth,
            },
        )

        if conversation["title"] == "New chat":
            new_title = payload.content.strip().replace("\n", " ")[:70]
            db.update_conversation(
                conversation_id, title=new_title or "New chat"
            )

        job_row = db.create_generation_job(conversation_id, user_message["id"])
        job = GenerationJob(job_row, user_message)
        _generations[job.id] = job
        _active_generation_by_conversation[conversation_id] = job.id

    _prune_generation_registry()
    worker = threading.Thread(
        target=_run_generation_job,
        kwargs={
            "job": job,
            "conversation": conversation,
            "payload": payload,
            "endpoint": endpoint,
            "model": model,
            "attachments": attachments,
            "project_files": project_files,
            "retained_generated_files": retained_generated_files,
        },
        name=f"generation-{job.id[:8]}",
        daemon=True,
    )
    worker.start()
    return job, user_message


@app.post(
    "/api/conversations/{conversation_id}/messages/start",
    status_code=202,
)
def start_message_generation(
    conversation_id: str, payload: MessageCreate
) -> dict[str, Any]:
    job, user_message = _start_generation_job(conversation_id, payload)
    return {
        "generation": job.snapshot(),
        "user_message": user_message,
    }


@app.get("/api/generations/{generation_id}")
def get_generation(generation_id: str) -> dict[str, Any]:
    with _generation_lock:
        job = _generations.get(generation_id)
    if job is not None:
        return job.snapshot()

    try:
        row = db.get_generation_job(generation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Generation not found")
    return {
        **row,
        "last_sequence": 0,
        "assistant_text": "",
        "reasoning_summary": "",
        "activities": [],
    }


@app.get("/api/generations/{generation_id}/stream")
def stream_generation(
    generation_id: str,
    after: int = Query(default=0, ge=0),
) -> StreamingResponse:
    with _generation_lock:
        job = _generations.get(generation_id)
    if job is None:
        raise HTTPException(
            status_code=410,
            detail="This generation stream is no longer available",
        )

    return StreamingResponse(
        job.event_stream(after),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/conversations/{conversation_id}/cancel")
def cancel_generation(conversation_id: str) -> dict[str, bool]:
    try:
        db.get_conversation(conversation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found")

    with _generation_lock:
        job_id = _active_generation_by_conversation.get(conversation_id)
        job = _generations.get(job_id) if job_id else None
    if job is not None and job.status in ACTIVE_GENERATION_STATUSES:
        job.request_cancel()
        return {"cancelled": True}
    return {"cancelled": False}


@app.post("/api/conversations/{conversation_id}/messages/stream")
def stream_message(
    conversation_id: str, payload: MessageCreate
) -> StreamingResponse:
    job, _ = _start_generation_job(conversation_id, payload)
    return StreamingResponse(
        job.event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/conversations/{conversation_id}/messages")
def send_message(
    conversation_id: str, payload: MessageCreate
) -> dict[str, Any]:
    job, user_message = _start_generation_job(conversation_id, payload)
    job.wait()
    snapshot = job.snapshot()
    if snapshot["status"] == "completed" and snapshot["assistant_message_id"]:
        assistant_message = db.get_message(snapshot["assistant_message_id"])
        return {
            "user_message": user_message,
            "assistant_message": assistant_message,
            "generated_files": assistant_message.get("metadata", {}).get(
                "generated_files", []
            ),
            "response_id": assistant_message.get("metadata", {}).get(
                "response_id"
            ),
        }
    if snapshot["status"] == "cancelled":
        raise HTTPException(status_code=409, detail="Generation was stopped")
    raise HTTPException(
        status_code=502,
        detail=snapshot.get("error") or "The generation failed",
    )


@app.get(
    "/api/conversations/{conversation_id}/generated-files/{file_id}/download",
    response_class=FileResponse,
)
def retained_generated_file(
    conversation_id: str,
    file_id: str,
) -> FileResponse:
    try:
        file_row = db.get_generated_file(file_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Generated file not found")
    if file_row["conversation_id"] != conversation_id:
        raise HTTPException(status_code=404, detail="Generated file not found")

    path = GENERATED_DIR / file_row["stored_name"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Retained file is missing")
    return FileResponse(
        path,
        filename=file_row["original_name"],
        media_type=file_row.get("content_type") or "application/octet-stream",
    )


@app.get(
    "/api/generated/{endpoint_id}/{container_id}/{file_id}",
    response_class=Response,
)
def generated_file(
    endpoint_id: str,
    container_id: str,
    file_id: str,
    filename: str | None = None,
) -> Response:
    endpoint = config.endpoints.get(endpoint_id)
    if endpoint is None:
        raise HTTPException(status_code=404, detail="Endpoint not found")

    try:
        upstream = download_generated_file(endpoint, container_id, file_id)
    except Exception as exc:
        logger.exception("Generated file download failed")
        raise HTTPException(status_code=502, detail=str(exc))

    safe_name = sanitize_filename(filename or file_id)
    return Response(
        content=upstream.content,
        media_type=upstream.headers.get(
            "content-type", "application/octet-stream"
        ),
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}"'
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", "3000")),
        reload=False,
    )
