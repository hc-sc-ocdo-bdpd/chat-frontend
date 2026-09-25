from __future__ import annotations

import tempfile
import zipfile
import logging
import posixpath
import re
from urllib.parse import unquote
from pathlib import Path
from typing import Any, Iterable

import httpx

from .config import EndpointConfig, ModelConfig
from .file_bundles import MAX_CONTAINER_FILES
from .response_stream import ResumableResponseStream

# Azure's Responses API has two different file paths:
#
# 1. input_file, which first goes through the model's context-stuffing
#    parser and only accepts a restricted set of formats.
# 2. Code Interpreter container files, which support additional formats
#    such as ZIP archives.
#
# Do not add an extension here unless Azure accepts it as a normal
# Responses API input_file. Archives intentionally stay out of this set.
CONTEXT_STUFFING_EXTENSIONS = {
    ".art",
    ".bat",
    ".brf",
    ".c",
    ".cls",
    ".css",
    ".csv",
    ".diff",
    ".doc",
    ".docx",
    ".dot",
    ".eml",
    ".es",
    ".h",
    ".hs",
    ".htm",
    ".html",
    ".hwp",
    ".hwpx",
    ".ics",
    ".ifb",
    ".java",
    ".js",
    ".json",
    ".keynote",
    ".ksh",
    ".ltx",
    ".mail",
    ".markdown",
    ".md",
    ".mht",
    ".mhtml",
    ".mjs",
    ".nws",
    ".odt",
    ".pages",
    ".patch",
    ".pdf",
    ".pl",
    ".pm",
    ".pot",
    ".potm",
    ".potx",
    ".ppa",
    ".pps",
    ".ppsm",
    ".ppsx",
    ".ppt",
    ".pptm",
    ".pptx",
    ".pwz",
    ".py",
    ".rst",
    ".rtf",
    ".scala",
    ".sh",
    ".shtml",
    ".srt",
    ".sty",
    ".svg",
    ".svgz",
    ".tex",
    ".text",
    ".txt",
    ".tsv",
    ".vcf",
    ".vtt",
    ".wiz",
    ".xla",
    ".xlb",
    ".xlc",
    ".xlm",
    ".xls",
    ".xlsx",
    ".xlt",
    ".xlw",
    ".xml",
    ".yaml",
    ".yml",
}


def supports_context_stuffing(filename: str) -> bool:
    """Return whether Azure accepts this file as a normal input_file."""
    return Path(filename).suffix.lower() in CONTEXT_STUFFING_EXTENSIONS


def make_client(endpoint: EndpointConfig) -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key=endpoint.api_key,
        base_url=endpoint.base_url,
        timeout=httpx.Timeout(900.0, connect=30.0),
        max_retries=2,
    )


def upload_file(
    endpoint: EndpointConfig,
    file_path: Path,
    original_name: str,
) -> str:
    client = make_client(endpoint)
    try:
        with file_path.open("rb") as handle:
            uploaded = client.files.create(
                file=(original_name, handle),
                purpose="assistants",
            )
    except Exception as exc:
        if not _is_invalid_extension_error(exc):
            raise
        uploaded = _upload_file_as_archive(
            client,
            file_path,
            original_name,
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    return uploaded.id


def _is_invalid_extension_error(exc: Exception) -> bool:
    return (
        getattr(exc, "status_code", None) == 400
        and "invalid extension" in str(exc).lower()
    )


def _upload_file_as_archive(
    client: Any,
    file_path: Path,
    original_name: str,
) -> Any:
    """Wrap an unsupported file so Code Interpreter can still access it."""
    safe_name = Path(original_name).name or "retained-file"
    with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as archive:
        with zipfile.ZipFile(
            archive,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
        ) as bundle:
            bundle.write(file_path, arcname=safe_name)
        archive.seek(0)
        return client.files.create(
            file=(f"{safe_name}.zip", archive),
            purpose="assistants",
        )


def build_history_input(messages: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content", ""))
        if (message.get("metadata") or {}).get("error"):
            content = str((message.get("metadata") or {}).get("partial_output") or "")
        if role not in {"user", "assistant"} or not content:
            continue
        history.append({"role": role, "content": content})
    return history



def normalize_domains(domains: list[str]) -> list[str]:
    """Normalize Azure web-search domain filters and remove duplicates."""
    normalized: list[str] = []
    seen: set[str] = set()

    for raw in domains:
        value = str(raw).strip().lower()
        value = value.removeprefix("https://").removeprefix("http://")
        value = value.split("/", 1)[0].strip().strip(".")
        if not value or value in seen:
            continue
        if any(character.isspace() for character in value):
            continue
        seen.add(value)
        normalized.append(value)

    return normalized[:100]


def build_web_search_tool(
    *,
    research_depth: str,
    allowed_domains: list[str],
    blocked_domains: list[str],
) -> dict[str, Any]:
    # Web research uses the high-context search setting.
    tool: dict[str, Any] = {
        "type": "web_search",
        "search_context_size": "high",
    }

    filters: dict[str, list[str]] = {}
    normalized_allowed = normalize_domains(allowed_domains)
    normalized_blocked = normalize_domains(blocked_domains)
    if normalized_allowed:
        filters["allowed_domains"] = normalized_allowed
    if normalized_blocked:
        filters["blocked_domains"] = normalized_blocked
    if filters:
        tool["filters"] = filters

    return tool


def build_response_arguments(
    *,
    model: ModelConfig,
    input_payload: str | list[dict[str, Any]],
    instructions: str,
    reasoning_effort: str,
    verbosity: str,
    max_output_tokens: int,
    use_code_interpreter: bool,
    provider_file_ids: list[str],
    use_web_search: bool,
    research_depth: str,
    web_allowed_domains: list[str],
    web_blocked_domains: list[str],
    previous_response_id: str | None,
    stream: bool = False,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "model": model.deployment,
        "input": input_payload,
        "instructions": instructions,
        "text": {"verbosity": verbosity},
        "max_output_tokens": max_output_tokens,
        "store": True,
    }

    # Reasoning summaries are the supported, safe way to expose progress from
    # reasoning models. This does not request or expose raw chain of thought.
    reasoning: dict[str, str] = {"summary": "auto"}
    if reasoning_effort and reasoning_effort != "auto":
        reasoning["effort"] = reasoning_effort
    arguments["reasoning"] = reasoning

    if previous_response_id:
        arguments["previous_response_id"] = previous_response_id

    tools: list[dict[str, Any]] = []
    if use_code_interpreter:
        provider_file_ids = list(dict.fromkeys(provider_file_ids))
        if len(provider_file_ids) > MAX_CONTAINER_FILES:
            raise ValueError("Code Interpreter accepts at most 50 file IDs. Bundle the files before submitting.")
        container: dict[str, Any] = {"type": "auto"}
        if provider_file_ids:
            container["file_ids"] = provider_file_ids
        tools.append(
            {
                "type": "code_interpreter",
                "container": container,
            }
        )

    if use_web_search:
        tools.append(
            build_web_search_tool(
                research_depth=research_depth,
                allowed_domains=web_allowed_domains,
                blocked_domains=web_blocked_domains,
            )
        )
        arguments["include"] = [
            "web_search_call.action.sources",
            "web_search_call.results",
        ]

    if tools:
        arguments["tools"] = tools
        arguments["tool_choice"] = "auto"

    if stream:
        arguments["stream"] = True
        # Background streaming gives Azure a durable response ID that can be
        # resumed if the HTTP connection drops mid-generation.
        arguments["background"] = True

    return arguments


def create_response(
    *,
    endpoint: EndpointConfig,
    model: ModelConfig,
    input_payload: str | list[dict[str, Any]],
    instructions: str,
    reasoning_effort: str,
    verbosity: str,
    max_output_tokens: int,
    use_code_interpreter: bool,
    provider_file_ids: list[str],
    use_web_search: bool,
    research_depth: str,
    web_allowed_domains: list[str],
    web_blocked_domains: list[str],
    previous_response_id: str | None,
) -> Any:
    client = make_client(endpoint)
    arguments = build_response_arguments(
        model=model,
        input_payload=input_payload,
        instructions=instructions,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        max_output_tokens=max_output_tokens,
        use_code_interpreter=use_code_interpreter,
        provider_file_ids=provider_file_ids,
        use_web_search=use_web_search,
        research_depth=research_depth,
        web_allowed_domains=web_allowed_domains,
        web_blocked_domains=web_blocked_domains,
        previous_response_id=previous_response_id,
    )
    try:
        return client.with_options(max_retries=0).responses.create(**arguments)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def stream_response(
    *,
    endpoint: EndpointConfig,
    model: ModelConfig,
    input_payload: str | list[dict[str, Any]],
    instructions: str,
    reasoning_effort: str,
    verbosity: str,
    max_output_tokens: int,
    use_code_interpreter: bool,
    provider_file_ids: list[str],
    use_web_search: bool,
    research_depth: str,
    web_allowed_domains: list[str],
    web_blocked_domains: list[str],
    previous_response_id: str | None,
    on_diagnostics: Any = None,
) -> Any:
    client = make_client(endpoint)
    arguments = build_response_arguments(
        model=model,
        input_payload=input_payload,
        instructions=instructions,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        max_output_tokens=max_output_tokens,
        use_code_interpreter=use_code_interpreter,
        provider_file_ids=provider_file_ids,
        use_web_search=use_web_search,
        research_depth=research_depth,
        web_allowed_domains=web_allowed_domains,
        web_blocked_domains=web_blocked_domains,
        previous_response_id=previous_response_id,
        stream=True,
    )
    return ResumableResponseStream(client, arguments, on_diagnostics)

def _container_path(value: str) -> str:
    path = unquote(re.sub(r"^sandbox:", "", value.strip(), flags=re.I))
    path = posixpath.normpath(path)
    if not path.startswith("/mnt/data/") or any(ord(c) < 32 for c in path):
        return ""
    return path


def extract_generated_files(
    response: Any, *, endpoint: EndpointConfig | None = None
) -> list[dict[str, str]]:
    """Resolve citations and uncited sandbox links to actual container files."""
    raw = response.model_dump()
    found: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    containers: list[str] = []
    linked_paths: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            item_type = value.get("type")
            file_id = value.get("file_id")
            container_id = value.get("container_id")
            if (
                item_type in {"code_interpreter_call", "container_file_citation"}
                and isinstance(container_id, str)
                and container_id not in containers
            ):
                containers.append(container_id)
            if item_type == "output_text" and isinstance(value.get("text"), str):
                for match in re.finditer(
                    r"\[[^\]\n]+\]\\?\((sandbox:/+[^)\n]+?)\\?\)",
                    value["text"], re.I,
                ):
                    path = _container_path(match[1])
                    if path:
                        linked_paths.add(path)
            if (
                item_type == "container_file_citation"
                and isinstance(file_id, str)
                and isinstance(container_id, str)
            ):
                key = (container_id, file_id)
                if key not in seen:
                    seen.add(key)
                    found.append(
                        {
                            "container_id": container_id,
                            "file_id": file_id,
                            "filename": str(
                                value.get("filename")
                                or value.get("name")
                                or file_id
                            ),
                        }
                    )
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(raw)

    # Most responses have proper citations. Only list files for missing links.
    # A basename citation is usable when just one linked path has that name.
    covered: set[str] = set()
    for source in found:
        path = _container_path(source["filename"])
        candidates = {p for p in linked_paths if posixpath.basename(p) == source["filename"]}
        if path:
            source["sandbox_path"] = path
            covered.add(path)
        elif len(candidates) == 1:
            source["sandbox_path"] = next(iter(candidates))
            covered.update(candidates)
    missing = linked_paths - covered
    if endpoint is None or not missing or not containers:
        return found

    matches: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    with make_client(endpoint).with_options(timeout=30.0, max_retries=0) as client:
        for container_id in containers:
            try:
                # SDK iteration follows pagination, including files after page 1.
                for file in client.containers.files.list(container_id, limit=100):
                    path = _container_path(file.path)
                    if path in missing:
                        key = (container_id, file.id)
                        matches.setdefault(path, {})[key] = {
                            "container_id": container_id,
                            "file_id": file.id,
                            "filename": posixpath.basename(path),
                            "sandbox_path": path,
                        }
            except Exception:
                # An expired container must not prevent retaining cited files.
                logging.getLogger(__name__).warning(
                    "Could not list generated files in container %s", container_id,
                    exc_info=True,
                )
    for path in sorted(missing):
        candidates = matches.get(path, {})
        if len(candidates) != 1:
            logging.getLogger(__name__).warning(
                "Sandbox file could not be resolved uniquely: %s", path
            )
            continue
        key, source = next(iter(candidates.items()))
        if key not in seen:
            seen.add(key)
            found.append(source)
        else:
            for cited in found:
                if (cited["container_id"], cited["file_id"]) == key:
                    cited["sandbox_path"] = path
    return found


def extract_web_research(response: Any) -> dict[str, Any]:
    """Extract URL citations, consulted sources, and web-search actions."""
    raw = response.model_dump()
    output = raw.get("output") if isinstance(raw, dict) else []
    if not isinstance(output, list):
        output = []

    actions: list[dict[str, Any]] = []
    citation_occurrences: list[dict[str, Any]] = []
    source_candidates: list[dict[str, str]] = []

    for item in output:
        if not isinstance(item, dict):
            continue

        if item.get("type") == "web_search_call":
            action = item.get("action")
            if isinstance(action, dict):
                action_record: dict[str, Any] = {
                    key: action[key]
                    for key in ("type", "query", "url", "pattern")
                    if action.get(key) not in (None, "")
                }
                if action_record:
                    actions.append(action_record)

                sources = action.get("sources")
                if isinstance(sources, list):
                    for source in sources:
                        if isinstance(source, dict) and source.get("url"):
                            source_candidates.append(
                                {
                                    "url": str(source["url"]),
                                    "title": str(source.get("title") or ""),
                                }
                            )

            results = item.get("results")
            if isinstance(results, list):
                for result in results:
                    if not isinstance(result, dict) or not result.get("url"):
                        continue
                    source_candidates.append(
                        {
                            "url": str(result["url"]),
                            "title": str(result.get("title") or ""),
                            "snippet": str(result.get("snippet") or "")[:1000],
                        }
                    )

        if item.get("type") != "message":
            continue

        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            annotations = part.get("annotations")
            if not isinstance(annotations, list):
                continue
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    continue
                citation = annotation
                if annotation.get("type") != "url_citation":
                    nested = annotation.get("url_citation")
                    if not isinstance(nested, dict):
                        continue
                    citation = nested
                if not citation.get("url"):
                    continue
                record = {
                    "url": str(citation["url"]),
                    "title": str(citation.get("title") or ""),
                    "start_index": citation.get("start_index"),
                    "end_index": citation.get("end_index"),
                }
                citation_occurrences.append(record)
                source_candidates.append(
                    {"url": record["url"], "title": record["title"]}
                )

    sources: list[dict[str, Any]] = []
    source_index_by_url: dict[str, int] = {}
    for candidate in source_candidates:
        url = candidate.get("url", "").strip()
        if not url:
            continue
        existing_index = source_index_by_url.get(url)
        if existing_index is not None:
            existing = sources[existing_index - 1]
            if not existing.get("title") and candidate.get("title"):
                existing["title"] = candidate["title"]
            if not existing.get("snippet") and candidate.get("snippet"):
                existing["snippet"] = candidate["snippet"]
            continue

        source_index = len(sources) + 1
        source_index_by_url[url] = source_index
        source: dict[str, Any] = {
            "index": source_index,
            "url": url,
            "title": candidate.get("title") or "",
        }
        if candidate.get("snippet"):
            source["snippet"] = candidate["snippet"]
        sources.append(source)

    citations: list[dict[str, Any]] = []
    for citation in citation_occurrences:
        citations.append(
            {
                **citation,
                "source_index": source_index_by_url.get(citation["url"]),
            }
        )

    return {
        "used": bool(actions or citations or sources),
        "actions": actions,
        "citations": citations,
        "sources": sources,
    }


def web_activity_labels(web_research: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for action in web_research.get("actions", []):
        action_type = action.get("type")
        if action_type == "search" and action.get("query"):
            label = f'Searched: {action["query"]}'
        elif action_type == "open_page" and action.get("url"):
            label = f'Opened page: {action["url"]}'
        elif action_type == "find_in_page" and action.get("pattern"):
            label = f'Found in page: {action["pattern"]}'
        else:
            continue
        if label not in labels:
            labels.append(label)
    return labels


def download_generated_file(
    endpoint: EndpointConfig,
    container_id: str,
    file_id: str,
) -> httpx.Response:
    with make_client(endpoint).with_options(timeout=60.0, max_retries=0) as client:
        upstream = client.containers.files.content.retrieve(file_id=file_id, container_id=container_id)
        headers = getattr(upstream, "headers", {}) or {}
        return httpx.Response(200, content=upstream.read(), headers={
            "content-type": headers.get("content-type", "application/octet-stream")})
