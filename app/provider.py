from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import httpx
from .config import EndpointConfig, ModelConfig


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
    with file_path.open("rb") as handle:
        uploaded = client.files.create(
            file=(original_name, handle),
            purpose="assistants",
        )
    return uploaded.id


def build_history_input(messages: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content", ""))
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
    # The UI exposes one research mode: thorough. Keep the parameter for
    # compatibility with older clients, but always use Azure's high-context search.
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
    return client.responses.create(**arguments)


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


class CombinedResponse:
    """Expose multiple chained Responses as one final response to the app."""

    def __init__(self, responses: list[Any], final_response: Any) -> None:
        self._responses = [*responses, final_response]
        self._final = final_response
        self.id = getattr(final_response, "id", "")
        self.output_text = "".join(
            str(getattr(response, "output_text", "") or "")
            for response in self._responses
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._final, name)

    def model_dump(self) -> dict[str, Any]:
        final_raw = self._final.model_dump()
        combined_output: list[Any] = []
        for response in self._responses:
            raw = response.model_dump()
            output = raw.get("output") if isinstance(raw, dict) else None
            if isinstance(output, list):
                combined_output.extend(output)
        final_raw["output"] = combined_output
        final_raw["id"] = self.id
        return final_raw


class ResumableResponseStream:
    """Durable Azure Responses stream with transparent reconnects.

    Azure background streams can be resumed from a response ID and event
    sequence number. This wrapper keeps the existing iterator interface used by
    main.py while recovering from dropped/chunk-truncated HTTP connections.

    A single GPT-5.6 / GPT-6 Astra response is capped at 128k output tokens. If
    Azure ends a response with ``max_output_tokens``, transparently chain a
    continuation response so the UI can receive a longer logical answer.
    """

    def __init__(self, client: Any, arguments: dict[str, Any]) -> None:
        self.client = client
        self.base_arguments = dict(arguments)
        self.current_stream: Any | None = None
        self.current_iterator: Any | None = None
        self.response_id: str | None = None
        self.sequence_number: int | None = None
        self.terminal = False
        self.closed = False
        self.resume_failures = 0
        self.continuations = 0
        self.prior_responses: list[Any] = []
        self.max_resume_attempts = _positive_int_env(
            "APP_STREAM_RESUME_ATTEMPTS", 8
        )
        self.max_continuations = _positive_int_env(
            "APP_MAX_AUTO_CONTINUATIONS", 3
        )
        self._start(self.base_arguments)

    def __iter__(self) -> "ResumableResponseStream":
        return self

    def _close_current(self) -> None:
        if self.current_stream is None:
            return
        close = getattr(self.current_stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        self.current_stream = None
        self.current_iterator = None

    def _start(self, arguments: dict[str, Any]) -> None:
        self._close_current()
        self.current_stream = self.client.responses.create(**arguments)
        self.current_iterator = iter(self.current_stream)
        self.response_id = None
        self.sequence_number = None
        self.terminal = False
        self.resume_failures = 0

    def _resume(self) -> None:
        if not self.response_id:
            raise RuntimeError("Cannot resume a response before Azure returned an ID")
        self._close_current()
        starting_after = self.sequence_number if self.sequence_number is not None else 0
        self.current_stream = self.client.responses.retrieve(
            response_id=self.response_id,
            stream=True,
            starting_after=starting_after,
        )
        self.current_iterator = iter(self.current_stream)

    @staticmethod
    def _incomplete_reason(response: Any) -> str:
        details = getattr(response, "incomplete_details", None)
        return str(getattr(details, "reason", "") or "")

    def _start_continuation(self, previous_response_id: str) -> None:
        self.continuations += 1
        arguments = dict(self.base_arguments)
        arguments["input"] = (
            "Continue exactly where the previous response stopped. "
            "Do not repeat material that was already completed. Finish the "
            "original user request, preserving the same format and level of detail."
        )
        arguments["previous_response_id"] = previous_response_id
        self._start(arguments)

    def _recover(self, error: Exception | None = None) -> bool:
        if not self.response_id or self.resume_failures >= self.max_resume_attempts:
            return False
        self.resume_failures += 1
        # Short bounded backoff avoids hammering Azure when a proxy or network
        # path is briefly unhealthy. The background response keeps running.
        time.sleep(min(0.4 * (2 ** (self.resume_failures - 1)), 5.0))
        try:
            self._resume()
            return True
        except Exception:
            if self.resume_failures >= self.max_resume_attempts:
                if error is not None:
                    raise error
                raise
            return self._recover(error)

    def __next__(self) -> Any:
        while not self.closed:
            try:
                if self.current_iterator is None:
                    raise StopIteration
                event = next(self.current_iterator)
            except StopIteration:
                if self.terminal:
                    raise
                if self._recover():
                    continue
                raise RuntimeError(
                    "The Azure response stream ended before a terminal event "
                    "and could not be resumed"
                )
            except Exception as exc:
                if self._recover(exc):
                    continue
                raise

            event_type = str(getattr(event, "type", ""))
            sequence_number = getattr(event, "sequence_number", None)
            if isinstance(sequence_number, int):
                self.sequence_number = sequence_number
                self.resume_failures = 0

            response = getattr(event, "response", None)
            response_id = getattr(response, "id", None)
            if isinstance(response_id, str) and response_id:
                self.response_id = response_id

            if event_type == "response.created":
                self.terminal = False
                return event

            if event_type == "response.incomplete":
                reason = self._incomplete_reason(response)
                if (
                    reason == "max_output_tokens"
                    and self.response_id
                    and self.continuations < self.max_continuations
                ):
                    if response is not None:
                        self.prior_responses.append(response)
                    self._start_continuation(self.response_id)
                    continue

                # main.py currently understands response.completed as the
                # successful terminal event. Preserve all text already streamed
                # instead of throwing it away when Azure returns a terminal
                # incomplete response (for example content filtering or after
                # the configured continuation ceiling).
                self.terminal = True
                return SimpleNamespace(
                    type="response.completed",
                    response=response,
                    sequence_number=self.sequence_number,
                )

            if event_type == "response.completed":
                self.terminal = True
                if self.prior_responses and response is not None:
                    return SimpleNamespace(
                        type="response.completed",
                        response=CombinedResponse(self.prior_responses, response),
                        sequence_number=self.sequence_number,
                    )

            if event_type == "response.failed":
                self.terminal = True

            return event

        raise StopIteration

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._close_current()

        # Because streamed requests run in background mode, closing the local
        # stream does not itself stop Azure. Cancel a non-terminal response so a
        # user pressing Stop, navigating away, or losing the browser connection
        # does not leave an orphaned billable generation running.
        if self.response_id and not self.terminal:
            try:
                self.client.responses.cancel(self.response_id)
            except Exception:
                pass


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
    return ResumableResponseStream(client, arguments)

def extract_generated_files(response: Any) -> list[dict[str, str]]:
    raw = response.model_dump()
    found: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            item_type = value.get("type")
            file_id = value.get("file_id")
            container_id = value.get("container_id")
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
    client = make_client(endpoint)
    upstream = client.containers.files.content.retrieve(
        file_id=file_id,
        container_id=container_id,
    )
    return httpx.Response(
        status_code=200,
        content=upstream.read(),
        headers={"content-type": "application/octet-stream"},
    )
