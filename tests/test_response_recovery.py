import importlib
import json
import sys
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from app.diagnostics import error_details
from app.provider import build_history_input
from app.response_stream import ResumableResponseStream


def response(status="completed", text="Finished", *, rid="resp-1", reason=None, error=None, tokens=10):
    raw = {
        "id": rid, "status": status, "model": "test-model",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": text, "annotations": []}]}] if text else [],
        "usage": {"input_tokens": 5, "output_tokens": tokens, "total_tokens": 5 + tokens,
                  "output_tokens_details": {"reasoning_tokens": tokens - 2}},
        "error": error, "incomplete_details": {"reason": reason} if reason else None,
    }
    return SimpleNamespace(**raw, output_text=text, model_dump=lambda: raw)


def event(kind, seq=None, **kwargs):
    return SimpleNamespace(type=kind, sequence_number=seq, **kwargs)


class Stream:
    def __init__(self, events):
        self.events = iter(events)
        self.response = httpx.Response(200, headers={"apim-request-id": "azure-http-id"})
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.events)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.closed = True


class Client:
    def __init__(self, streams, retrievals=()):
        self.streams = iter(streams)
        self.retrievals = iter(retrievals)
        self.responses = self
        self.creates, self.gets, self.cancels, self.options = [], [], [], []

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    def create(self, **kwargs):
        self.creates.append(kwargs)
        return next(self.streams)

    def retrieve(self, **kwargs):
        self.gets.append(kwargs)
        result = next(self.retrievals)
        if isinstance(result, Exception):
            raise result
        return result

    def cancel(self, rid):
        self.cancels.append(rid)
        return response("cancelled", "", rid=rid)


@pytest.fixture
def make_stream(monkeypatch):
    monkeypatch.delenv("APP_MAX_AUTO_CONTINUATIONS", raising=False)
    def build(client, callback=None):
        stream = ResumableResponseStream(client, {"model": "test", "stream": True, "background": True, "store": True}, callback)
        monkeypatch.setattr(stream._stop, "wait", lambda seconds: stream._stop.is_set())
        return stream
    return build


def created():
    return event("response.created", 0, response=response("in_progress", ""))


def test_missing_terminal_event_recovers_complete_answer_without_new_post(make_stream):
    client = Client([Stream([created(), event("response.output_text.delta", 1, delta="First")])],
                    [response(text="First and the rest")])
    stream = make_stream(client)
    events = list(stream)
    assert events[-1].response.output_text == "First and the rest"
    assert len(client.creates) == 1
    assert client.gets == [{"response_id": "resp-1"}]
    stream.close()
    assert client.cancels == []
    assert all(options["max_retries"] == 0 for options in client.options)
    assert stream.diagnostics()["requests"][1]["apim-request-id"] == "azure-http-id"


def test_resume_discards_replayed_deltas(make_stream):
    client = Client([Stream([created(), event("response.output_text.delta", 1, delta="A"), httpx.ReadError("lost")])],
                    [response("in_progress", ""), Stream([
                        event("response.output_text.delta", 1, delta="A"),
                        event("response.output_text.delta", 2, delta="B"),
                        event("response.completed", 3, response=response(text="AB")),
                    ])])
    stream = make_stream(client)
    events = list(stream)
    assert "".join(e.delta for e in events if hasattr(e, "delta")) == "AB"
    assert client.gets[1] == {"response_id": "resp-1", "stream": True, "starting_after": 1}
    assert len(client.creates) == 1


def test_error_event_checks_saved_status_and_polls_same_job(make_stream):
    client = Client([Stream([created(), event("error", 1, code="server_error", message="The server had an error")])],
                    [response("in_progress", ""), response("queued", ""), response(text="Recovered")])
    stream = make_stream(client)
    events = list(stream)
    assert events[-1].response.output_text == "Recovered"
    assert any(e.type == "app.status" for e in events)
    assert stream.diagnostics()["mode"] == "polling"
    assert len(client.creates) == 1
    assert all(not row.get("stream") for row in client.gets)


def test_failed_stream_resume_falls_back_to_polling(make_stream):
    client = Client([Stream([created(), httpx.ReadError("lost")])],
                    [response("in_progress", ""), httpx.ConnectError("proxy unavailable"), response()])
    stream = make_stream(client)
    assert list(stream)[-1].type == "response.completed"
    assert stream.diagnostics()["mode"] == "polling"


def test_provider_failed_is_not_retried_as_new_generation(make_stream):
    failed = response("failed", "Partial", error={"code": "server_error", "message": "Backend failure"})
    client = Client([Stream([created(), event("response.failed", 2, response=failed)])])
    stream = make_stream(client)
    assert list(stream)[-1].type == "response.failed"
    assert error_details(failed)["code"] == "server_error"
    assert client.gets == []
    assert len(client.creates) == 1


def test_error_event_get_confirms_provider_failure(make_stream):
    failed = response("failed", "Partial", error={"code": "server_error", "message": "Backend failure"})
    client = Client([Stream([created(), event("error", 1, code="server_error", message="Connection error")])], [failed])
    assert list(make_stream(client))[-1].type == "response.failed"
    assert len(client.creates) == 1


@pytest.mark.parametrize("reason", ["max_output_tokens", "content_filter"])
def test_incomplete_keeps_real_status_and_does_not_auto_continue(make_stream, reason):
    client = Client([Stream([created(), event("response.incomplete", 1,
                    response=response("incomplete", "Partial", reason=reason))])])
    result = list(make_stream(client))[-1]
    assert result.type == "response.incomplete"
    assert result.response.status == "incomplete"
    assert len(client.creates) == 1


def test_opt_in_continuation_is_visible_and_usage_is_summed(make_stream, monkeypatch):
    monkeypatch.setenv("APP_MAX_AUTO_CONTINUATIONS", "1")
    client = Client([
        Stream([created(), event("response.incomplete", 1, response=response("incomplete", "A", reason="max_output_tokens", tokens=100))]),
        Stream([event("response.created", 0, response=response("in_progress", "", rid="resp-2")),
                event("response.completed", 2, response=response(text="B", rid="resp-2", tokens=20))]),
    ])
    stream = make_stream(client)
    events = list(stream)
    assert any(e.type == "app.status" and "additional request" in e.label for e in events)
    assert events[-1].response.output_text == "AB"
    assert stream.diagnostics()["usage"]["output_tokens"] == 120
    assert client.creates[1]["previous_response_id"] == "resp-1"


def test_failed_retrieval_is_bounded_and_does_not_cancel_or_repost(make_stream):
    client = Client([Stream([created(), httpx.ReadError("lost")])], [httpx.ConnectError("offline")] * 8)
    stream = make_stream(client)
    with pytest.raises(httpx.ReadError):
        list(stream)
    stream.close()
    assert len(client.gets) == 8
    assert len(client.creates) == 1
    assert client.cancels == []


def test_stop_before_iteration_never_starts_paid_request(make_stream):
    client = Client([])
    stream = make_stream(client)
    stream.cancel()
    assert list(stream) == []
    assert client.creates == []


def test_stop_known_response_cancels_azure(make_stream):
    client = Client([Stream([created()])])
    stream = make_stream(client)
    next(stream)
    stream.cancel()
    assert client.cancels == ["resp-1"]


def test_stop_while_creation_in_flight_cancels_when_id_arrives(make_stream):
    entered, release = threading.Event(), threading.Event()
    client = Client([])
    def create(**kwargs):
        client.creates.append(kwargs)
        entered.set()
        assert release.wait(2)
        return Stream([created()])
    client.create = create
    stream = make_stream(client)
    worker = threading.Thread(target=lambda: list(stream))
    worker.start()
    assert entered.wait(2)
    stream.cancel()
    release.set()
    worker.join(2)
    assert not worker.is_alive()
    assert client.cancels == ["resp-1"]


def test_real_sdk_does_not_retry_ambiguous_create_error(make_stream):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(500, json={"error": {"code": "server_error", "message": "Provider failed"}},
                              headers={"x-request-id": "failed-post-id"})
    client = OpenAI(api_key="fake", base_url="https://example.test/openai/v1/",
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=2)
    stream = make_stream(client)
    with pytest.raises(Exception, match="Provider failed"):
        list(stream)
    assert len(requests) == 1
    assert requests[0].headers["X-Client-Request-Id"]
    assert stream.diagnostics()["transport_errors"][-1]["request_id"] == "failed-post-id"
    client.close()


def test_diagnostics_exclude_prompt_and_output(make_stream):
    client = Client([Stream([created(), event("response.completed", 1, response=response(text="CONFIDENTIAL OUTPUT"))])])
    stream = make_stream(client)
    stream.base_arguments["input"] = "CONFIDENTIAL PROMPT"
    list(stream)
    assert "CONFIDENTIAL" not in json.dumps(stream.diagnostics())


def test_real_sdk_azure_sse_error_recovers_terminal_response(make_stream):
    requests = []
    def handler(request):
        requests.append(request)
        if request.method == "POST":
            created_data = {"type": "response.created", "sequence_number": 0,
                            "response": response("in_progress", "").model_dump()}
            failed_data = {"error": {"code": "server_error", "message": "The server had an error processing your request. (Please include the request ID 00000000-0000-4000-8000-000000000001 in your email.)"}}
            wire = f"data: {json.dumps(created_data)}\n\ndata: {json.dumps(failed_data)}\n\n"
            return httpx.Response(200, content=wire.encode(), headers={"content-type": "text/event-stream", "apim-request-id": "initial-request-id"})
        return httpx.Response(200, json=response(text="Saved answer").model_dump())
    client = OpenAI(api_key="fake", base_url="https://example.test/openai/v1/",
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    stream = make_stream(client)
    events = list(stream)
    assert events[-1].response.output_text == "Saved answer"
    assert [r.method for r in requests] == ["POST", "GET"]
    assert stream.diagnostics()["transport_errors"][0]["request_id"] == "00000000-0000-4000-8000-000000000001"
    client.close()


def test_cancel_registered_after_stop_is_applied():
    from app.jobs import GenerationJob
    job = GenerationJob({"id": "one", "conversation_id": "chat"}, {})
    cancelled = []
    job.request_cancel()
    job.set_cancel_callback(lambda: cancelled.append(True))
    assert cancelled == [True]


def test_null_usage_stays_unknown(make_stream):
    final = response()
    final.usage = None
    stream = make_stream(Client([Stream([event("response.completed", 0, response=final)])]))
    list(stream)
    assert stream.diagnostics()["usage"] is None


def test_error_text_does_not_enter_model_history():
    assert build_history_input([
        {"role": "assistant", "content": "Request failed: server error", "metadata": {"error": True}},
        {"role": "assistant", "content": "Partial plus diagnostic error", "metadata": {"error": True, "partial_output": "Partial"}},
    ]) == [{"role": "assistant", "content": "Partial"}]


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://example-resource.openai.azure.com/openai/v1/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_1", "test-model")
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "data"))
    sys.modules.pop("app.main", None)
    main = importlib.import_module("app.main")
    return main, TestClient(main.app)


@pytest.mark.parametrize("status,reason", [("failed", None), ("incomplete", "max_output_tokens")])
def test_failure_keeps_partial_output_and_durable_diagnostics(app_client, monkeypatch, make_stream, status, reason):
    main, client = app_client
    terminal = response(status, "Useful partial answer", reason=reason,
                        error={"code": "server_error", "message": "Backend failure"} if status == "failed" else None)
    provider = Client([Stream([created(), event("response.output_text.delta", 1, delta="Useful"),
                              event(f"response.{status}", 2, response=terminal)])])
    monkeypatch.setattr(main, "stream_response", lambda **kwargs: make_stream(provider, kwargs["on_diagnostics"]))
    conversation = client.post("/api/conversations", json={}).json()
    result = client.post(f"/api/conversations/{conversation['id']}/messages/start", json={"content": "Test", "model_id": "test-model"})
    job = main._generations[result.json()["generation"]["id"]]
    assert job.wait(3)
    chat = client.get(f"/api/conversations/{conversation['id']}").json()
    message = chat["messages"][-1]
    assert message["metadata"]["partial_output"] == "Useful partial answer"
    assert "Useful partial answer" in message["content"]
    assert message["metadata"]["provider_status"] == status
    diagnostic = client.get(message["metadata"]["diagnostics_url"])
    assert diagnostic.status_code == 200
    assert diagnostic.json()["response_id"] == "resp-1"
    assert diagnostic.json()["terminal_response"]["status"] == status
    assert diagnostic.json()["usage"]["output_tokens"] == 10
    # Diagnostics are on disk, and can be read after the live job is evicted.
    main._generations.pop(job.id)
    assert client.get(message["metadata"]["diagnostics_url"]).json() == diagnostic.json()
    assert bool(chat.get("previous_response_id")) == (status == "incomplete")


def test_completed_file_citation_survives_later_provider_failure(app_client, monkeypatch, make_stream):
    main, client = app_client
    item = {"type": "message", "content": [{"type": "output_text", "text": "Report",
            "annotations": [{"type": "container_file_citation", "container_id": "cntr-1",
                             "file_id": "cfile-1", "filename": "report.csv"}]}]}
    provider = Client([Stream([created(), event("response.output_item.done", 1,
                    item=SimpleNamespace(model_dump=lambda: item)),
                    event("response.failed", 2, response=response("failed", "", error={"code": "server_error", "message": "Failed"}))])])
    monkeypatch.setattr(main, "stream_response", lambda **kwargs: make_stream(provider, kwargs["on_diagnostics"]))
    monkeypatch.setattr(main, "download_generated_file", lambda *args: httpx.Response(200, content=b"value\n42\n"))
    conversation = client.post("/api/conversations", json={}).json()
    started = client.post(f"/api/conversations/{conversation['id']}/messages/start", json={"content": "Test", "model_id": "test-model"}).json()
    job = main._generations[started["generation"]["id"]]
    assert job.wait(3)
    message = client.get(f"/api/conversations/{conversation['id']}").json()["messages"][-1]
    url = message["metadata"]["generated_files"][0]["url"]
    assert client.get(url).content == b"value\n42\n"

