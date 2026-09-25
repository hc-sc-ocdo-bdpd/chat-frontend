"""Allowlisted request metadata, without prompts, files, or reasoning text."""
from __future__ import annotations

import re
from typing import Any


def field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def error_details(value: Any) -> dict[str, Any]:
    """Handle SDK exceptions, top-level SSE errors, and response.error."""
    response = field(value, "response")
    error = field(response, "error") or field(value, "error")
    body = field(value, "body")
    if not error and isinstance(body, dict):
        error = body.get("error", body)
    error = error or value
    code = field(error, "code")
    message = field(error, "message")
    if not message and isinstance(error, Exception):
        message = str(error)
    result = {
        "type": type(value).__name__,
        "code": str(code) if code is not None else None,
        "message": str(message or "The Azure response stream failed")[:4000],
        "http_status": field(value, "status_code"),
        "request_id": field(value, "request_id") or field(value, "_request_id"),
    }
    headers = field(response, "headers", {}) or {}
    for name in ("x-request-id", "apim-request-id", "x-ms-request-id"):
        if headers.get(name):
            result[name] = headers[name]
    # Azure sometimes only supplies the support ID inside the error message.
    match = re.search(r"request ID\s+([a-zA-Z0-9-]+)", result["message"], re.I)
    if match and not result["request_id"]:
        result["request_id"] = match.group(1)
    return {key: val for key, val in result.items() if val is not None}


def response_details(response: Any) -> dict[str, Any]:
    result = {name: field(response, name) for name in ("id", "status", "created_at", "model")}
    reason = field(field(response, "incomplete_details"), "reason")
    if reason:
        result["incomplete_reason"] = reason
    if field(response, "error"):
        result["error"] = error_details(response)
    usage = field(response, "usage")
    result["usage"] = None
    if usage is not None:
        result["usage"] = {
            key: field(usage, key) for key in ("input_tokens", "output_tokens", "total_tokens")
        }
        for key, detail in (("input_tokens_details", "cached_tokens"),
                            ("output_tokens_details", "reasoning_tokens")):
            value = field(field(usage, key), detail)
            if value is not None:
                result["usage"][key] = {detail: value}
    return result


def aggregate_usage(responses: list[dict[str, Any]]) -> dict[str, Any] | None:
    known = [row["usage"] for row in responses if isinstance(row.get("usage"), dict)]
    if not known:
        return None
    result: dict[str, Any] = {
        key: sum(row.get(key) or 0 for row in known)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    for key, detail in (("input_tokens_details", "cached_tokens"),
                        ("output_tokens_details", "reasoning_tokens")):
        result[key] = {detail: sum((row.get(key) or {}).get(detail, 0) for row in known)}
    result["responses_with_usage"] = len(known)
    result["responses_total"] = len(responses)
    result["complete"] = len(known) == len(responses)
    return result


def failure_message(response: Any) -> str:
    status = field(response, "status")
    if status == "incomplete":
        reason = field(field(response, "incomplete_details"), "reason", "unknown")
        if reason == "max_output_tokens":
            return ("Azure reached the output token limit, which also includes reasoning tokens. "
                    "The response is incomplete. Ask to continue if you want another request.")
        return f"Azure returned an incomplete response ({reason})."
    if status == "cancelled":
        return "Azure cancelled the response."
    return error_details(response)["message"]
