"""Recover the same Azure background response without replaying generation POSTs."""
from __future__ import annotations

import copy
import logging
import os
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Any, Callable

import httpx

from .diagnostics import aggregate_usage, error_details, field, response_details

logger = logging.getLogger("chat.stream")
TERMINAL_STATUSES = {"completed", "failed", "incomplete", "cancelled"}
DURABLE_START_MODES = {"auto", "always", "never"}


def _nonnegative_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _choice_env(name: str, default: str, allowed: set[str]) -> str:
    value = os.environ.get(name, default).strip().lower()
    return value if value in allowed else default


class CombinedResponse:
    def __init__(self, responses: list[Any], final_response: Any) -> None:
        self._responses = [*responses, final_response]
        self._final = final_response
        self.id = field(final_response, "id", "")
        self.output_text = "".join(str(field(r, "output_text", "") or "") for r in self._responses)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._final, name)

    def model_dump(self) -> dict[str, Any]:
        raw = dict(self._final.model_dump())
        raw["output"] = [item for r in self._responses for item in r.model_dump().get("output", [])]
        raw["usage"] = aggregate_usage([response_details(r) for r in self._responses])
        return raw


class ResumableResponseStream:
    """Resume SSE or poll a stored response after a transport failure.

    A provider-terminal failure stays failed. A new paid response is only
    created for an explicitly configured token-limit continuation.

    Long-running requests can use a durable polling start. Instead of holding
    the initial POST open waiting for the first SSE event, the app creates a
    background response without streaming so Azure returns the response ID
    immediately, then polls that saved response. This closes the pre-ID failure
    window where an Azure gateway timeout cannot be recovered safely.
    """

    def __init__(self, client: Any, arguments: dict[str, Any],
                 on_diagnostics: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.client = client
        self.base_arguments = dict(arguments)
        self._pending_arguments: dict[str, Any] | None = dict(arguments)
        self.current_stream: Any | None = None
        self.current_iterator: Any | None = None
        self.response_id: str | None = None
        self.sequence_number: int | None = None
        self.last_response: Any | None = None
        self.terminal = False
        self.closed = False
        self._stop = threading.Event()
        self._cancel_requested = threading.Event()
        self._lock = threading.RLock()
        self._polling = False
        self.resume_failures = 0
        self.continuations = 0
        self.prior_responses: list[Any] = []
        self.max_resume_attempts = _nonnegative_env("APP_STREAM_RESUME_ATTEMPTS", 8)
        self.max_continuations = _nonnegative_env("APP_MAX_AUTO_CONTINUATIONS", 0)
        self.durable_start_mode = _choice_env(
            "APP_DURABLE_START_MODE", "auto", DURABLE_START_MODES
        )
        self.durable_start_max_output_tokens = _nonnegative_env(
            "APP_DURABLE_START_MAX_OUTPUT_TOKENS", 65536
        )
        self._on_diagnostics = on_diagnostics
        self._diagnostics: dict[str, Any] = {
            "responses": [], "requests": [], "transport_errors": [],
            "continuations": 0, "reconnects": 0, "mode": "stream",
            "start_mode": None,
            "durable_start_mode": self.durable_start_mode,
            "durable_start_max_output_tokens": self.durable_start_max_output_tokens,
            "max_auto_continuations": self.max_continuations,
            "create_retries": 0,
        }

    @classmethod
    def resume_existing(cls, client, response_id, *, on_diagnostics=None, diagnostics=None):
        stream = cls(client, {}, on_diagnostics)
        stream._pending_arguments = None
        stream.response_id = response_id
        stream._polling = True
        stream.max_continuations = 0  # Attaching must never create paid work.
        if diagnostics:
            for key in stream._diagnostics:
                if key in diagnostics:
                    stream._diagnostics[key] = copy.deepcopy(diagnostics[key])
        stream._diagnostics["mode"] = "polling after restart"
        return stream

    def __iter__(self) -> "ResumableResponseStream":
        return self

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            result = copy.deepcopy(self._diagnostics)
            result.update(response_id=self.response_id, sequence_number=self.sequence_number,
                          continuations=self.continuations)
            result["usage"] = aggregate_usage(result["responses"])
            return result

    def _notify(self) -> None:
        if self._on_diagnostics:
            try:
                self._on_diagnostics(self.diagnostics())
            except Exception:
                # A diagnostic write failure must not trigger a second request.
                logger.warning("Could not persist request diagnostics", exc_info=True)

    def _request_client(self, *, short: bool = False) -> Any:
        # SDK retries can duplicate a POST whose outcome is unknown. Recovery
        # here is exclusively GET against the ID of the original response.
        options: dict[str, Any] = {"max_retries": 0}
        if short:
            options["timeout"] = httpx.Timeout(60.0, connect=15.0)
        return self.client.with_options(**options)

    def _record_http(self, value: Any, operation: str, client_request_id: str | None = None) -> None:
        response = field(value, "response")
        headers = field(response, "headers", {}) or {}
        row: dict[str, Any] = {"operation": operation, "at": time.time()}
        if client_request_id:
            row["client_request_id"] = client_request_id
        for key in ("x-request-id", "apim-request-id", "x-ms-request-id", "x-ms-region"):
            if headers.get(key):
                row[key] = headers[key]
        if field(value, "_request_id"):
            row["x-request-id"] = field(value, "_request_id")
        if field(response, "status_code"):
            row["http_status"] = field(response, "status_code")
        self._diagnostics["requests"].append(row)
        requests = self._diagnostics["requests"]
        self._diagnostics["requests"] = (
            [r for r in requests if r["operation"].startswith("create")]
            + [r for r in requests if not r["operation"].startswith("create")][-100:]
        )
        self._notify()

    def _remember(self, response: Any) -> None:
        if response is None:
            return
        self.last_response = response
        details = response_details(response)
        if details.get("id"):
            self.response_id = details["id"]
            rows = self._diagnostics["responses"]
            previous = next((r for r in rows if r.get("id") == self.response_id), None)
            if previous is None:
                rows.append(details)
            elif previous != details:
                previous.update(details)
            else:
                return
        self._notify()

    def _record_error(self, error: Any) -> None:
        self._diagnostics["transport_errors"].append({"at": time.time(), **error_details(error)})
        self._diagnostics["transport_errors"] = self._diagnostics["transport_errors"][-30:]
        self._notify()

    def _close_current(self) -> None:
        with self._lock:
            stream = self.current_stream
            self.current_stream = None
            self.current_iterator = None
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass

    def _attach(self, stream: Any) -> None:
        with self._lock:
            if self.closed:
                stream.close()
                return
            self.current_stream = stream
            self.current_iterator = iter(stream)

    def _should_use_durable_start(self, arguments: dict[str, Any]) -> bool:
        """Choose polling-first background creation for requests most likely to run long."""
        if not arguments.get("background"):
            return False
        if self.durable_start_mode == "always":
            return True
        if self.durable_start_mode == "never":
            return False

        reasoning = arguments.get("reasoning") or {}
        effort = str(field(reasoning, "effort", "") or "").strip().lower()
        if effort == "max":
            return True

        threshold = self.durable_start_max_output_tokens
        if threshold <= 0:
            return False
        try:
            return int(arguments.get("max_output_tokens") or 0) >= threshold
        except (TypeError, ValueError):
            return False

    def _start_streaming(self, arguments: dict[str, Any], client_request_id: str) -> None:
        self._diagnostics["start_mode"] = "stream"
        self._diagnostics["mode"] = "stream"
        self._notify()
        stream = self._request_client().responses.create(
            **arguments, extra_headers={"X-Client-Request-Id": client_request_id}
        )
        self._record_http(stream, "create", client_request_id)
        self._attach(stream)

    def _start_durable_polling(self, arguments: dict[str, Any], client_request_id: str) -> None:
        # Azure documents background non-streaming creation as returning a
        # response ID immediately. Polling that ID avoids depending on the
        # first SSE event for recoverability on long-running requests.
        create_arguments = dict(arguments)
        create_arguments.pop("stream", None)
        create_arguments["background"] = True
        create_arguments["store"] = True

        self._diagnostics["start_mode"] = "durable_background_polling"
        self._diagnostics["mode"] = "background polling"
        self._notify()
        response = self._request_client().responses.create(
            **create_arguments,
            extra_headers={"X-Client-Request-Id": client_request_id},
        )
        self._record_http(response, "create_background", client_request_id)
        self._remember(response)

        status = field(response, "status")
        if status in TERMINAL_STATUSES:
            self.current_iterator = iter([self._terminal_event(response)])
            self._polling = False
        elif status in {"queued", "in_progress"}:
            # We already have the durable response ID, so attach an SSE stream to
            # that saved response instead of issuing a blocking status GET. This
            # preserves live deltas/activity while retaining full recoverability:
            # if the stream drops, _recover() can resume or poll by response_id.
            try:
                self._resume()
                self._polling = False
                self._diagnostics["mode"] = "durable background streaming"
                self._diagnostics["start_mode"] = "durable_background_streaming"
                self._diagnostics["reconnects"] += 1
                self._notify()
            except Exception as exc:
                self._record_error(exc)
                if not self._recoverable(exc):
                    raise
                # Streaming attachment is an optimization, not a prerequisite.
                # The response ID is already safe, so fall back to polling the
                # same job without ever submitting a second generation.
                self._polling = True
                self._diagnostics["mode"] = "background polling fallback"
                self._diagnostics["start_mode"] = "durable_background_polling_fallback"
                self._notify()
        else:
            raise RuntimeError(f"Unexpected Azure response status after background create: {status}")

        # Stop can race with a background create just like it can race with the
        # first response.created SSE event. Once the ID is known, cancel it.
        if self._cancel_requested.is_set():
            self.close()
            self._cancel_remote()

    def _start(self, arguments: dict[str, Any]) -> None:
        self._close_current()
        self.response_id = None
        self.sequence_number = None
        self.last_response = None
        self.terminal = False
        self.resume_failures = 0
        self._polling = False
        client_request_id = str(uuid.uuid4())
        self._record_http(None, "create_started", client_request_id)
        if self._should_use_durable_start(arguments):
            self._start_durable_polling(arguments, client_request_id)
        else:
            self._start_streaming(arguments, client_request_id)

    def _resume(self) -> None:
        arguments: dict[str, Any] = {"response_id": self.response_id, "stream": True}
        if self.sequence_number is not None:
            arguments["starting_after"] = self.sequence_number
        stream = self._request_client().responses.retrieve(**arguments)
        self._record_http(stream, "resume")
        self._attach(stream)

    def _retrieve(self) -> Any:
        response = self._request_client(short=True).responses.retrieve(response_id=self.response_id)
        self._record_http(response, "retrieve")
        self._remember(response)
        return response

    @staticmethod
    def _terminal_event(response: Any) -> Any:
        return SimpleNamespace(type=f"response.{field(response, 'status')}", response=response)

    @staticmethod
    def _recoverable(error: Any) -> bool:
        status = field(error, "status_code")
        if isinstance(status, int):
            return status in {408, 409, 429} or status >= 500
        code = error_details(error).get("code")
        return code in {None, "server_error", "internal_error", "internal_server_error",
                        "rate_limit_exceeded", "timeout", "request_timeout"}

    def _recover(self, error: Any = None, *, prefer_polling: bool = False) -> bool:
        if self.closed or not self.response_id or (error is not None and not self._recoverable(error)):
            return False
        self._close_current()
        while not self.closed and self.resume_failures < self.max_resume_attempts:
            self.resume_failures += 1
            if self._stop.wait(min(0.4 * 2 ** min(self.resume_failures - 1, 4), 5.0)):
                return False
            try:
                response = self._retrieve()
                status = field(response, "status")
                if status in TERMINAL_STATUSES:
                    self.current_iterator = iter([self._terminal_event(response)])
                    self._polling = False
                    return True
                if status not in {"queued", "in_progress"}:
                    return False
                if not (prefer_polling or self._polling):
                    try:
                        self._resume()
                        self._diagnostics["reconnects"] += 1
                        self._notify()
                        return True
                    except Exception as exc:
                        self._record_error(exc)
                        if not self._recoverable(exc):
                            return False
                # A healthy status GET confirms the job is still alive. Poll
                # for its final result if its event stream is unavailable.
                self._polling = True
                self.resume_failures = 0
                self._diagnostics["mode"] = "polling"
                self._notify()
                return True
            except Exception as exc:
                self._record_error(exc)
                if not self._recoverable(exc):
                    return False
        return False

    def _poll_event(self) -> Any:
        if self._stop.wait(5.0):
            raise StopIteration
        response = self._retrieve()
        status = field(response, "status")
        if status in TERMINAL_STATUSES:
            return self._terminal_event(response)
        if status not in {"queued", "in_progress"}:
            raise RuntimeError(f"Unexpected Azure response status: {status}")
        self.resume_failures = 0
        return SimpleNamespace(type="app.snapshot" if field(response, "output_text") else "app.status", response=response, label="Waiting for Azure, checking saved response")

    def __next__(self) -> Any:
        while not self.closed and not self.terminal:
            if self._pending_arguments is not None:
                arguments, self._pending_arguments = self._pending_arguments, None
                try:
                    self._start(arguments)
                except Exception as exc:
                    self._record_error(exc)
                    # The creation outcome is unknown, so do not POST again.
                    raise
            try:
                event = self._poll_event() if self._polling else next(self.current_iterator)
            except StopIteration:
                if self.closed:
                    break
                if self._recover():
                    continue
                raise RuntimeError("The Azure connection ended and the saved response could not be retrieved.")
            except Exception as exc:
                self._record_error(exc)
                if self._recover(exc):
                    continue
                if self.closed:
                    break
                raise

            event_type = str(field(event, "type", ""))
            response = field(event, "response")
            event_id = field(response, "id")
            sequence = field(event, "sequence_number")
            if isinstance(sequence, int):
                if (self.sequence_number is not None and sequence <= self.sequence_number
                        and (not event_id or event_id == self.response_id)):
                    continue
                self.sequence_number = sequence
                if event_type not in {"error", "response.created"}:
                    self.resume_failures = 0
            self._remember(response)

            if self._cancel_requested.is_set():
                self.close()
                self._cancel_remote()
                break
            if self.closed:
                # Stop can race with the event that first reveals Azure's ID.
                self._cancel_remote()
                break
            if event_type == "error":
                self._record_error(event)
                if self._recover(event, prefer_polling=True):
                    continue
                return event

            if event_type == "response.incomplete" and (
                field(field(response, "incomplete_details"), "reason") == "max_output_tokens"
                and self.response_id and self.continuations < self.max_continuations
            ):
                self.prior_responses.append(response)
                self.continuations += 1
                self._pending_arguments = {
                    **self.base_arguments,
                    "previous_response_id": self.response_id,
                    "input": "Continue where the previous response stopped. Finish the original request without repeating completed material.",
                }
                self._notify()
                return SimpleNamespace(type="app.status", label=f"Continuing in an additional request ({self.continuations}/{self.max_continuations})")

            if event_type in {f"response.{s}" for s in TERMINAL_STATUSES}:
                self.terminal = True
                if self.prior_responses and response is not None:
                    response = CombinedResponse(self.prior_responses, response)
                    self.last_response = response
                    return SimpleNamespace(type=event_type, response=response)
            return event
        raise StopIteration

    def close(self) -> None:
        """Release local resources. A network failure is not a user cancellation."""
        self.closed = True
        self._stop.set()
        self._close_current()

    def dispose(self) -> None:
        self.close()
        close_client = getattr(self.client, "close", None)
        if callable(close_client):
            close_client()

    def _cancel_remote(self) -> None:
        if self.response_id and not self.terminal:
            try:
                result = self._request_client(short=True).responses.cancel(self.response_id)
                self._remember(result)
                self._diagnostics["cancel_status"] = field(result, "status", "unknown")
            except Exception as exc:
                self._diagnostics["cancel_error"] = error_details(exc)
            self._notify()

    def cancel(self) -> None:
        self._cancel_requested.set()
        self._stop.set()
        if self.response_id or self._pending_arguments is not None:
            self.close()
            self._cancel_remote()
        # If creation is in flight, wait for the first provider result to learn
        # the durable response ID, then cancel that saved response.
