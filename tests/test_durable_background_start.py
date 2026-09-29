from types import SimpleNamespace

import pytest

from app.response_stream import ResumableResponseStream


def response(status="completed", text="Finished", *, rid="resp-1"):
    raw = {
        "id": rid,
        "status": status,
        "model": "test-model",
        "output": (
            [{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": text,
                    "annotations": [],
                }],
            }]
            if text else []
        ),
        "usage": {
            "input_tokens": 5,
            "output_tokens": 10,
            "total_tokens": 15,
            "output_tokens_details": {"reasoning_tokens": 8},
        },
        "error": None,
        "incomplete_details": None,
    }
    return SimpleNamespace(
        **raw,
        output_text=text,
        model_dump=lambda: raw,
    )


def event(kind, seq=None, **kwargs):
    return SimpleNamespace(type=kind, sequence_number=seq, **kwargs)


class Stream:
    def __init__(self, events):
        self.events = iter(events)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.events)

    def close(self):
        self.closed = True


class Client:
    def __init__(self, creates, retrievals=()):
        self.create_results = iter(creates)
        self.retrieval_results = iter(retrievals)
        self.responses = self
        self.creates = []
        self.gets = []
        self.cancels = []
        self.options = []

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    def create(self, **kwargs):
        self.creates.append(kwargs)
        result = next(self.create_results)
        if isinstance(result, Exception):
            raise result
        return result

    def retrieve(self, **kwargs):
        self.gets.append(kwargs)
        result = next(self.retrieval_results)
        if isinstance(result, Exception):
            raise result
        return result

    def cancel(self, response_id):
        self.cancels.append(response_id)
        return response("cancelled", "", rid=response_id)


def make_stream(monkeypatch, client, arguments):
    monkeypatch.setenv("APP_DURABLE_START_MODE", "auto")
    monkeypatch.setenv("APP_DURABLE_START_MAX_OUTPUT_TOKENS", "65536")
    stream = ResumableResponseStream(client, arguments)
    monkeypatch.setattr(stream._stop, "wait", lambda seconds: stream._stop.is_set())
    return stream


def test_max_reasoning_gets_response_id_before_long_wait(monkeypatch):
    completed = event("response.completed", 1, response=response("completed", "Finished"))
    client = Client(
        [response("queued", "")],
        [Stream([completed])],
    )
    stream = make_stream(monkeypatch, client, {
        "model": "test-model",
        "input": "Test",
        "reasoning": {"summary": "auto", "effort": "max"},
        "max_output_tokens": 128000,
        "background": True,
        "store": True,
        "stream": True,
    })

    events = list(stream)

    assert len(client.creates) == 1
    assert client.creates[0]["background"] is True
    assert client.creates[0]["store"] is True
    assert "stream" not in client.creates[0]
    assert client.gets == [
        {"response_id": "resp-1", "stream": True},
    ]
    assert events[-1].type == "response.completed"
    assert stream.diagnostics()["response_id"] == "resp-1"
    assert stream.diagnostics()["start_mode"] == "durable_background_streaming"


def test_large_output_budget_uses_durable_start_even_without_max_reasoning(monkeypatch):
    completed = event("response.completed", 1, response=response("completed", "Done"))
    client = Client([response("queued", "")], [Stream([completed])])
    stream = make_stream(monkeypatch, client, {
        "model": "test-model",
        "input": "Test",
        "reasoning": {"summary": "auto", "effort": "high"},
        "max_output_tokens": 65536,
        "background": True,
        "store": True,
        "stream": True,
    })

    assert list(stream)[-1].type == "response.completed"
    assert "stream" not in client.creates[0]


def test_normal_request_keeps_streaming_start(monkeypatch):
    created = event("response.created", 0, response=response("in_progress", ""))
    completed = event("response.completed", 1, response=response("completed", "Done"))
    client = Client([Stream([created, completed])])
    stream = make_stream(monkeypatch, client, {
        "model": "test-model",
        "input": "Test",
        "reasoning": {"summary": "auto", "effort": "medium"},
        "max_output_tokens": 16384,
        "background": True,
        "store": True,
        "stream": True,
    })

    assert list(stream)[-1].type == "response.completed"
    assert client.creates[0]["stream"] is True
    assert client.gets == []
    assert stream.diagnostics()["start_mode"] == "stream"


def test_durable_create_failure_is_not_reposted(monkeypatch):
    failure = RuntimeError("Azure gateway returned 408 before response ID")
    client = Client([failure])
    stream = make_stream(monkeypatch, client, {
        "model": "test-model",
        "input": "Test",
        "reasoning": {"summary": "auto", "effort": "max"},
        "max_output_tokens": 128000,
        "background": True,
        "store": True,
        "stream": True,
    })

    with pytest.raises(RuntimeError, match="408"):
        list(stream)

    assert len(client.creates) == 1
    assert client.gets == []
    assert stream.diagnostics()["create_retries"] == 0


def test_durable_stream_attach_failure_falls_back_to_polling_same_response(monkeypatch):
    attach_failure = RuntimeError("stream attach failed")
    attach_failure.status_code = 500
    client = Client(
        [response("queued", "")],
        [attach_failure, response("completed", "Done")],
    )
    stream = make_stream(monkeypatch, client, {
        "model": "test-model",
        "input": "Test",
        "reasoning": {"summary": "auto", "effort": "max"},
        "max_output_tokens": 128000,
        "background": True,
        "store": True,
        "stream": True,
    })

    assert list(stream)[-1].type == "response.completed"
    assert len(client.creates) == 1
    assert client.gets == [
        {"response_id": "resp-1", "stream": True},
        {"response_id": "resp-1"},
    ]
    assert stream.diagnostics()["start_mode"] == "durable_background_polling_fallback"
