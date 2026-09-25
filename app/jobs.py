from __future__ import annotations

import json
import threading
import time
import logging
from collections import deque
from collections.abc import Iterator
from typing import Any

ACTIVE_GENERATION_STATUSES = {"queued", "running"}
TERMINAL_GENERATION_STATUSES = {"completed", "failed", "cancelled"}


class GenerationJob:
    """In-process event log for one server-owned model response.

    The provider request runs on its own worker thread. Browser streams only
    subscribe to this event log, so closing or replacing a browser view never
    owns or cancels the provider request.
    """

    def __init__(
        self,
        row: dict[str, Any],
        user_message: dict[str, Any],
        on_checkpoint=None,
    ) -> None:
        self.id = str(row["id"])
        self.conversation_id = str(row["conversation_id"])
        self.user_message = user_message
        self.status = str(row.get("status") or "queued")
        self.status_label = str(row.get("status_label") or "Queued")
        self.created_at = float(row.get("created_at") or time.time())
        self.started_at = row.get("started_at")
        self.completed_at = row.get("completed_at")
        self.assistant_message_id = row.get("assistant_message_id")
        self.error = row.get("error")

        saved = row.get("snapshot") or {}
        self.assistant_text = saved.get("assistant_text", "")
        self.reasoning_summary = saved.get("reasoning_summary", "")
        self.activities = list(saved.get("activities", []))
        self._sequence = int(saved.get("last_sequence", 0))
        self._events = deque(maxlen=2048)
        self._on_checkpoint = on_checkpoint
        self._last_checkpoint = 0.0
        self._condition = threading.Condition(threading.RLock())
        self.cancel_event = threading.Event()
        self._cancel_callback: Any | None = None

    def snapshot(self, *, include_output: bool = True) -> dict[str, Any]:
        with self._condition:
            result: dict[str, Any] = {
                "id": self.id,
                "conversation_id": self.conversation_id,
                "status": self.status,
                "status_label": self.status_label,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "assistant_message_id": self.assistant_message_id,
                "error": self.error,
                "last_sequence": self._sequence,
            }
            if include_output:
                result.update(
                    {
                        "assistant_text": self.assistant_text,
                        "reasoning_summary": self.reasoning_summary,
                        "activities": list(self.activities),
                    }
                )
            return result

    def publish(self, payload: dict[str, Any]) -> dict[str, Any]:
        event = dict(payload)
        event_type = str(event.get("type") or "")

        with self._condition:
            if event_type == "started":
                self.status = "running"
                self.status_label = str(event.get("label") or "Thinking")
                self.started_at = float(event.get("started_at") or time.time())
            elif event_type == "status":
                self.status_label = str(event.get("label") or "Thinking")
            elif event_type == "reasoning_delta":
                self.reasoning_summary += str(event.get("delta") or "")
            elif event_type == "reasoning_done":
                self.reasoning_summary = str(event.get("text") or "")
            elif event_type == "activity":
                label = str(event.get("label") or "")
                if label and label not in self.activities:
                    self.activities.append(label)
                if label:
                    self.status_label = label
            elif event_type == "output_snapshot":
                self.assistant_text = str(event.get("text") or "")
            elif event_type == "output_delta":
                self.assistant_text += str(event.get("delta") or "")
                self.status_label = "Writing response"
            elif event_type == "done":
                self.status = "completed"
                self.status_label = "Completed"
                self.completed_at = float(event.get("completed_at") or time.time())
                assistant = event.get("assistant_message") or {}
                self.assistant_message_id = assistant.get("id")
            elif event_type == "cancelled":
                self.status = "cancelled"
                self.status_label = "Stopped"
                self.completed_at = float(event.get("completed_at") or time.time())
                assistant = event.get("assistant_message") or {}
                self.assistant_message_id = assistant.get("id")
            elif event_type == "error":
                self.status = "failed"
                self.status_label = "Failed"
                self.error = str(event.get("message") or "The request failed")
                self.completed_at = float(event.get("completed_at") or time.time())
                assistant = event.get("assistant_message") or {}
                self.assistant_message_id = assistant.get("id")

            event["generation_id"] = self.id
            event["conversation_id"] = self.conversation_id
            self._sequence += 1
            event["sequence"] = self._sequence
            self._events.append(event)
            self._condition.notify_all()
            self.checkpoint(force=self.status in TERMINAL_GENERATION_STATUSES)
            return dict(event)

    def checkpoint(self, *, force=False):
        if self._on_checkpoint and (force or time.monotonic() - self._last_checkpoint >= 1.0):
            try:
                self._on_checkpoint(self.snapshot())
                self._last_checkpoint = time.monotonic()
            except Exception:
                logging.getLogger(__name__).warning("Could not checkpoint generation %s", self.id, exc_info=True)

    def request_cancel(self) -> None:
        self.cancel_event.set()
        with self._condition:
            if self.status in ACTIVE_GENERATION_STATUSES:
                self.status_label = "Stopping"
            self._condition.notify_all()
            callback = self._cancel_callback
        if callable(callback):
            try:
                callback()
            except Exception:
                pass

    def set_cancel_callback(self, callback: Any | None) -> None:
        with self._condition:
            self._cancel_callback = callback
            already_cancelled = self.cancel_event.is_set()
        if already_cancelled and callable(callback):
            callback()

    def event_stream(self, after: int = 0) -> Iterator[str]:
        cursor = max(0, int(after))
        # New processes and bounded event logs need a full replacement snapshot.
        # Sending it even when the client's cursor is ahead prevents stale cursors
        # from hiding updates after a process restart.
        with self._condition:
            needs_snapshot = cursor > self._sequence or (self._events and cursor < self._events[0]["sequence"] - 1)
            if needs_snapshot:
                snapshot = self.snapshot()
                cursor = snapshot["last_sequence"]
        if needs_snapshot:
            yield f"data: {json.dumps({'type': 'snapshot', 'snapshot': snapshot})}\n\n"

        while True:
            with self._condition:
                while (
                    self._sequence <= cursor
                    and self.status in ACTIVE_GENERATION_STATUSES
                ):
                    notified = self._condition.wait(timeout=15.0)
                    if not notified:
                        break

                snapshot = None
                if self._events and cursor < self._events[0]["sequence"] - 1:
                    snapshot = self.snapshot()
                    cursor = snapshot["last_sequence"]
                events = [
                    dict(event)
                    for event in self._events
                    if int(event["sequence"]) > cursor
                ]
                terminal = self.status in TERMINAL_GENERATION_STATUSES

            if snapshot is not None:
                yield f"data: {json.dumps({'type': 'snapshot', 'snapshot': snapshot})}\n\n"
            if not events:
                if terminal:
                    return
                yield ": keep-alive\n\n"
                continue

            for event in events:
                cursor = int(event["sequence"])
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

            if terminal and cursor >= self._sequence:
                return

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self.status in ACTIVE_GENERATION_STATUSES:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True
