from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import AppConfig, EndpointConfig, ModelConfig, load_config
from .db import Database
from .provider import (
    build_history_input,
    create_response,
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
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

config: AppConfig = load_config(CONFIG_PATH)
db = Database(DATA_DIR / "app.db")
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

_cancellation_lock = threading.RLock()
_active_cancellations: dict[str, threading.Event] = {}


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


def encode_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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

    return attachments, project_files


def prepare_provider_files(
    endpoint: EndpointConfig,
    attachments: list[dict[str, Any]],
    project_files: list[dict[str, Any]],
) -> tuple[list[str], list[tuple[dict[str, Any], str]]]:
    provider_file_ids: list[str] = []
    named_provider_files: list[tuple[dict[str, Any], str]] = []

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
        provider_file_ids.append(provider_file_id)
        named_provider_files.append((attachment, provider_file_id))

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
        provider_file_ids.append(provider_file_id)
        named_provider_files.append((project_file, provider_file_id))

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


def generated_downloads(
    endpoint: EndpointConfig, response: Any
) -> list[dict[str, str]]:
    return [
        {
            **item,
            "url": (
                f"/api/generated/{endpoint.id}/"
                f"{item['container_id']}/{item['file_id']}"
                f"?filename={item['filename']}"
            ),
        }
        for item in extract_generated_files(response)
    ]


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
        "project": project,
        "project_files": (
            db.list_project_files(conversation["project_id"])
            if conversation.get("project_id")
            else []
        ),
    }


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
    try:
        attachments = db.list_attachments(conversation_id)
        db.delete_conversation(conversation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found")

    for attachment in attachments:
        try:
            (UPLOAD_DIR / attachment["stored_name"]).unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete local attachment %s", attachment["id"])
    return {"deleted": True}


@app.post("/api/conversations/{conversation_id}/branch/{message_id}")
def branch_conversation(
    conversation_id: str, message_id: str
) -> dict[str, Any]:
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
        db.add_message(
            branch["id"],
            message["role"],
            message["content"],
            metadata,
            created_at=message["created_at"],
        )

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
    try:
        deleted_count = db.truncate_messages_from(conversation_id, message_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Message not found")
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


@app.post("/api/conversations/{conversation_id}/cancel")
def cancel_generation(conversation_id: str) -> dict[str, bool]:
    try:
        db.get_conversation(conversation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found")

    with _cancellation_lock:
        cancellation = _active_cancellations.get(conversation_id)
        if cancellation is not None:
            cancellation.set()
            return {"cancelled": True}
    return {"cancelled": False}


@app.post("/api/conversations/{conversation_id}/messages/stream")
def stream_message(
    conversation_id: str, payload: MessageCreate
) -> StreamingResponse:
    try:
        conversation = db.get_conversation(conversation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found")

    endpoint, model = normalize_message_provider(payload)
    validate_message_options(payload, model)
    attachments, project_files = conversation_context(conversation, payload)

    user_message = db.add_message(
        conversation_id,
        "user",
        payload.content.strip(),
        {
            "attachment_ids": payload.attachment_ids,
            "project_file_ids": [item["id"] for item in project_files],
            "endpoint_id": config.default_endpoint,
            "model_id": payload.model_id or config.default_model,
            "use_web_search": payload.use_web_search,
            "research_depth": payload.research_depth,
        },
    )

    if conversation["title"] == "New chat":
        new_title = payload.content.strip().replace("\n", " ")[:70]
        db.update_conversation(conversation_id, title=new_title or "New chat")

    cancellation = threading.Event()
    with _cancellation_lock:
        previous_cancellation = _active_cancellations.get(conversation_id)
        if previous_cancellation is not None:
            previous_cancellation.set()
        _active_cancellations[conversation_id] = cancellation

    def generate() -> Iterator[str]:
        started_at = time.monotonic()
        reasoning_summary = ""
        assistant_text = ""
        activities: list[str] = []
        final_response: Any | None = None
        provider_stream: Any | None = None

        def activity(label: str) -> str | None:
            if label in activities:
                return None
            activities.append(label)
            return encode_sse({"type": "activity", "label": label})

        try:
            yield encode_sse(
                {
                    "type": "started",
                    "user_message": user_message,
                    "started_at": time.time(),
                    "project_file_count": len(project_files),
                }
            )

            if attachments or project_files:
                yield encode_sse(
                    {
                        "type": "status",
                        "label": "Preparing attached files",
                    }
                )

            provider_file_ids, named_provider_files = prepare_provider_files(
                endpoint, attachments, project_files
            )
            input_payload, previous_response_id = build_input_payload(
                conversation=conversation,
                payload=payload,
                named_provider_files=named_provider_files,
            )

            yield encode_sse({"type": "status", "label": "Thinking"})

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

            for event in provider_stream:
                if cancellation.is_set():
                    logger.info("Generation cancelled for %s", conversation_id)
                    return

                event_type = str(getattr(event, "type", ""))

                if event_type in {
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_summary.delta",
                }:
                    delta = str(getattr(event, "delta", "") or "")
                    if delta:
                        reasoning_summary += delta
                        yield encode_sse(
                            {"type": "reasoning_delta", "delta": delta}
                        )
                    continue

                if event_type in {
                    "response.reasoning_summary_text.done",
                    "response.reasoning_summary.done",
                }:
                    complete_text = str(getattr(event, "text", "") or "")
                    if complete_text:
                        reasoning_summary = complete_text
                        yield encode_sse(
                            {"type": "reasoning_done", "text": complete_text}
                        )
                    continue

                if event_type == "response.code_interpreter_call.in_progress":
                    encoded = activity("Starting Code Interpreter")
                    if encoded:
                        yield encoded
                    continue

                if event_type == "response.code_interpreter_call_code.delta":
                    encoded = activity("Writing Python")
                    if encoded:
                        yield encoded
                    continue

                if event_type == "response.code_interpreter_call.interpreting":
                    encoded = activity("Running Python")
                    if encoded:
                        yield encoded
                    continue

                if event_type == "response.code_interpreter_call.completed":
                    encoded = activity("Code Interpreter finished")
                    if encoded:
                        yield encoded
                    continue

                if event_type == "response.web_search_call.in_progress":
                    encoded = activity("Starting web search")
                    if encoded:
                        yield encoded
                    continue

                if event_type == "response.web_search_call.searching":
                    encoded = activity("Searching the web")
                    if encoded:
                        yield encoded
                    continue

                if event_type == "response.web_search_call.completed":
                    encoded = activity("Web search finished")
                    if encoded:
                        yield encoded
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
                            label = f'Searched: {action["query"]}'
                        elif action_type == "open_page" and action.get("url"):
                            label = f'Opened page: {action["url"]}'
                        elif action_type == "find_in_page" and action.get("pattern"):
                            label = f'Found in page: {action["pattern"]}'
                        else:
                            label = None
                        if label:
                            encoded = activity(label)
                            if encoded:
                                yield encoded
                    continue

                if event_type == "response.output_text.delta":
                    delta = str(getattr(event, "delta", "") or "")
                    if delta:
                        assistant_text += delta
                        yield encode_sse(
                            {"type": "output_delta", "delta": delta}
                        )
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

            if final_response is None:
                raise RuntimeError(
                    "The stream ended without a completed response event"
                )

            if not assistant_text:
                assistant_text = (
                    getattr(final_response, "output_text", None)
                    or "(The model returned no text.)"
                )

            downloads = generated_downloads(endpoint, final_response)
            web_research = extract_web_research(final_response)
            for label in web_activity_labels(web_research):
                if label not in activities:
                    activities.append(label)
            duration_seconds = round(time.monotonic() - started_at, 1)
            response_id = str(getattr(final_response, "id", ""))
            assistant_message = db.add_message(
                conversation_id,
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
            )
            db.update_conversation(
                conversation_id,
                endpoint_id=endpoint.id,
                model_id=model.id,
                previous_response_id=response_id,
                previous_endpoint_id=endpoint.id,
                previous_model_id=model.id,
            )

            yield encode_sse(
                {
                    "type": "done",
                    "assistant_message": assistant_message,
                    "generated_files": downloads,
                    "response_id": response_id,
                    "duration_seconds": duration_seconds,
                    "web_research": web_research,
                }
            )
        except GeneratorExit:
            logger.info("Client stopped generation for %s", conversation_id)
            raise
        except Exception as exc:
            if cancellation.is_set():
                logger.info("Cancelled provider stream ended for %s", conversation_id)
                return
            logger.exception("Streaming provider request failed")
            error_message = db.add_message(
                conversation_id,
                "assistant",
                f"Request failed: {exc}",
                {"error": True},
            )
            yield encode_sse(
                {
                    "type": "error",
                    "message": str(exc),
                    "assistant_message": error_message,
                }
            )
        finally:
            with _cancellation_lock:
                if _active_cancellations.get(conversation_id) is cancellation:
                    _active_cancellations.pop(conversation_id, None)

            if provider_stream is not None:
                close = getattr(provider_stream, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        logger.debug("Could not close provider stream", exc_info=True)

    return StreamingResponse(
        generate(),
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
    try:
        conversation = db.get_conversation(conversation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found")

    endpoint, model = normalize_message_provider(payload)
    validate_message_options(payload, model)
    attachments, project_files = conversation_context(conversation, payload)

    user_message = db.add_message(
        conversation_id,
        "user",
        payload.content.strip(),
        {
            "attachment_ids": payload.attachment_ids,
            "project_file_ids": [item["id"] for item in project_files],
            "endpoint_id": config.default_endpoint,
            "model_id": payload.model_id or config.default_model,
            "use_web_search": payload.use_web_search,
            "research_depth": payload.research_depth,
        },
    )

    if conversation["title"] == "New chat":
        new_title = payload.content.strip().replace("\n", " ")[:70]
        db.update_conversation(conversation_id, title=new_title or "New chat")

    try:
        provider_file_ids, named_provider_files = prepare_provider_files(
            endpoint, attachments, project_files
        )
        input_payload, previous_response_id = build_input_payload(
            conversation=conversation,
            payload=payload,
            named_provider_files=named_provider_files,
        )
        response = create_response(
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

        downloads = generated_downloads(endpoint, response)
        web_research = extract_web_research(response)
        assistant_text = response.output_text or "(The model returned no text.)"
        assistant_message = db.add_message(
            conversation_id,
            "assistant",
            assistant_text,
            {
                "endpoint_id": endpoint.id,
                "model_id": model.id,
                "response_id": response.id,
                "generated_files": downloads,
                "web_research": web_research,
                "web_search_enabled": payload.use_web_search,
                "research_depth": payload.research_depth,
                "activities": web_activity_labels(web_research),
            },
        )
        db.update_conversation(
            conversation_id,
            endpoint_id=endpoint.id,
            model_id=model.id,
            previous_response_id=response.id,
            previous_endpoint_id=endpoint.id,
            previous_model_id=model.id,
        )

        return {
            "user_message": user_message,
            "assistant_message": assistant_message,
            "generated_files": downloads,
            "response_id": response.id,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Provider request failed")
        db.add_message(
            conversation_id,
            "assistant",
            f"Request failed: {exc}",
            {"error": True},
        )
        raise HTTPException(status_code=502, detail=str(exc))


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
