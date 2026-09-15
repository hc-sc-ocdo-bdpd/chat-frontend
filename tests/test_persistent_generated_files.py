from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from app.db import Database
from app.jobs import GenerationJob


def generated_record(file_id: str = "retained-1") -> dict[str, object]:
    return {
        "id": file_id,
        "original_name": "report.csv",
        "stored_name": f"conversation/{file_id}_report.csv",
        "content_type": "text/csv",
        "size_bytes": 12,
        "source_endpoint_id": "azure-openai",
        "source_container_id": "cntr-1",
        "source_file_id": "cfile-1",
        "provider_files": {},
        "created_at": 123.0,
    }


def test_generated_files_are_transactionally_linked_to_messages(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "app.db")
    conversation = database.create_conversation("azure-openai", "model")
    message = database.add_message(
        conversation["id"],
        "assistant",
        "Download the report.",
        {"generated_files": [{"id": "retained-1"}]},
        generated_files=[generated_record()],
    )

    files = database.list_generated_files(conversation["id"])
    assert len(files) == 1
    assert files[0]["message_id"] == message["id"]
    assert files[0]["original_name"] == "report.csv"

    database.set_generated_provider_file(
        files[0]["id"], "azure-openai", "file-uploaded"
    )
    assert database.get_generated_file(files[0]["id"])["provider_files"] == {
        "azure-openai": "file-uploaded"
    }


def test_truncating_messages_cascades_generated_file_records(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "app.db")
    conversation = database.create_conversation("azure-openai", "model")
    first = database.add_message(conversation["id"], "user", "Create it")
    database.add_message(
        conversation["id"],
        "assistant",
        "Created.",
        generated_files=[generated_record()],
    )

    pending_cleanup = database.list_generated_files_from_message(
        conversation["id"], first["id"]
    )
    assert [item["id"] for item in pending_cleanup] == ["retained-1"]

    database.truncate_messages_from(conversation["id"], first["id"])
    assert database.list_generated_files(conversation["id"]) == []


def test_generation_jobs_and_unread_state_are_persistent(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    conversation = database.create_conversation("azure-openai", "model")
    user_message = database.add_message(conversation["id"], "user", "Hello")
    job = database.create_generation_job(conversation["id"], user_message["id"])

    summary = database.list_conversations()[0]
    assert summary["generation_id"] == job["id"]
    assert summary["generation_status"] == "queued"

    assistant = database.add_message(conversation["id"], "assistant", "Hi")
    database.update_generation_job(
        job["id"],
        status="completed",
        status_label="Completed",
        assistant_message_id=assistant["id"],
        completed_at=456.0,
    )

    summary = database.list_conversations()[0]
    assert summary["generation_id"] is None
    assert summary["unread"] is True

    database.mark_conversation_read(conversation["id"])
    assert database.list_conversations()[0]["unread"] is False


def test_generation_event_log_can_be_replayed_after_navigation() -> None:
    row = {
        "id": "job-1",
        "conversation_id": "conversation-1",
        "status": "queued",
        "status_label": "Queued",
        "created_at": 100.0,
    }
    job = GenerationJob(row, {"id": "message-1"})
    job.publish({"type": "started", "started_at": 101.0})
    job.publish({"type": "output_delta", "delta": "Hello"})
    job.publish(
        {
            "type": "done",
            "assistant_message": {"id": "message-2"},
            "completed_at": 102.0,
        }
    )

    events = [
        json.loads(block.removeprefix("data: ").strip())
        for block in job.event_stream(after=0)
    ]
    assert [event["sequence"] for event in events] == [1, 2, 3]
    assert job.snapshot()["assistant_text"] == "Hello"
    assert job.snapshot()["status"] == "completed"


def test_generated_file_is_saved_and_reattached_on_later_turn(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AZURE_OPENAI_BASE_URL",
        "https://example-resource.openai.azure.com/openai/v1/",
    )
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_1", "test-model")
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "data"))
    sys.modules.pop("app.main", None)

    from app import main

    first_response = SimpleNamespace(
        id="resp-1",
        output_text="The report is ready.",
        model_dump=lambda: {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "The report is ready.",
                            "annotations": [
                                {
                                    "type": "container_file_citation",
                                    "container_id": "cntr-1",
                                    "file_id": "cfile-1",
                                    "filename": "report.csv",
                                }
                            ],
                        }
                    ],
                }
            ]
        },
    )
    second_response = SimpleNamespace(
        id="resp-2",
        output_text="I reopened the retained report.",
        model_dump=lambda: {"output": []},
    )
    provider_calls: list[dict[str, object]] = []

    def fake_stream_response(**kwargs):
        provider_calls.append(kwargs)
        response = first_response if len(provider_calls) == 1 else second_response
        return iter(
            [
                SimpleNamespace(
                    type="response.output_text.delta",
                    delta=response.output_text,
                ),
                SimpleNamespace(type="response.completed", response=response),
            ]
        )

    monkeypatch.setattr(main, "stream_response", fake_stream_response)
    monkeypatch.setattr(
        main,
        "download_generated_file",
        lambda *_args: httpx.Response(
            200,
            content=b"name,value\nanswer,42\n",
            headers={"content-type": "text/csv"},
        ),
    )
    monkeypatch.setattr(
        main,
        "upload_file",
        lambda _endpoint, _path, _name: "file-retained-upload",
    )

    client = TestClient(main.app)
    conversation = client.post("/api/conversations", json={}).json()
    started = client.post(
        f"/api/conversations/{conversation['id']}/messages/start",
        json={"content": "Create a CSV", "model_id": "test-model"},
    )
    assert started.status_code == 202
    first_job = main._generations[started.json()["generation"]["id"]]
    assert first_job.wait(timeout=3)

    stored_chat = client.get(
        f"/api/conversations/{conversation['id']}"
    ).json()
    generated = stored_chat["messages"][-1]["metadata"]["generated_files"][0]
    assert generated["url"].startswith(
        f"/api/conversations/{conversation['id']}/generated-files/"
    )
    downloaded = client.get(generated["url"])
    assert downloaded.status_code == 200
    assert downloaded.content == b"name,value\nanswer,42\n"

    source_assistant = stored_chat["messages"][-1]
    branch_response = client.post(
        f"/api/conversations/{conversation['id']}/branch/{source_assistant['id']}"
    )
    assert branch_response.status_code == 200
    branch = branch_response.json()
    branch_generated = branch["messages"][-1]["metadata"]["generated_files"][0]
    assert branch_generated["url"].startswith(
        f"/api/conversations/{branch['id']}/generated-files/"
    )
    assert client.get(branch_generated["url"]).content == downloaded.content

    branch_file = main.db.get_generated_file(branch_generated["id"])
    branch_path = main.GENERATED_DIR / branch_file["stored_name"]
    assert branch_path.is_file()
    assert client.delete(f"/api/conversations/{branch['id']}").status_code == 200
    assert not branch_path.exists()

    second = client.post(
        f"/api/conversations/{conversation['id']}/messages/start",
        json={"content": "Read that report", "model_id": "test-model"},
    )
    assert second.status_code == 202
    second_job = main._generations[second.json()["generation"]["id"]]
    assert second_job.wait(timeout=3)

    assert provider_calls[1]["provider_file_ids"] == ["file-retained-upload"]
    assert provider_calls[1]["use_code_interpreter"] is True

    barrier = threading.Barrier(2)

    def concurrent_stream_response(**_kwargs):
        barrier.wait(timeout=3)
        response = SimpleNamespace(
            id=f"resp-{threading.current_thread().name}",
            output_text="Finished independently.",
            model_dump=lambda: {"output": []},
        )
        return iter(
            [
                SimpleNamespace(
                    type="response.output_text.delta",
                    delta=response.output_text,
                ),
                SimpleNamespace(type="response.completed", response=response),
            ]
        )

    monkeypatch.setattr(main, "stream_response", concurrent_stream_response)
    parallel_conversations = [
        client.post("/api/conversations", json={}).json() for _ in range(2)
    ]
    parallel_starts = [
        client.post(
            f"/api/conversations/{item['id']}/messages/start",
            json={"content": "Run independently", "model_id": "test-model"},
        )
        for item in parallel_conversations
    ]
    assert [response.status_code for response in parallel_starts] == [202, 202]
    parallel_jobs = [
        main._generations[response.json()["generation"]["id"]]
        for response in parallel_starts
    ]
    assert all(job.wait(timeout=3) for job in parallel_jobs)
    assert all(job.snapshot()["status"] == "completed" for job in parallel_jobs)

    stream_entered = threading.Event()
    stream_closed = threading.Event()

    class CancellableStream:
        def __iter__(self):
            return self

        def __next__(self):
            stream_entered.set()
            stream_closed.wait(timeout=3)
            raise StopIteration

        def close(self):
            stream_closed.set()

    monkeypatch.setattr(
        main,
        "stream_response",
        lambda **_kwargs: CancellableStream(),
    )
    cancellable_conversation = client.post(
        "/api/conversations", json={}
    ).json()
    cancellable_start = client.post(
        f"/api/conversations/{cancellable_conversation['id']}/messages/start",
        json={"content": "Wait here", "model_id": "test-model"},
    )
    assert cancellable_start.status_code == 202
    assert stream_entered.wait(timeout=3)

    duplicate = client.post(
        f"/api/conversations/{cancellable_conversation['id']}/messages/start",
        json={"content": "Do not overlap", "model_id": "test-model"},
    )
    assert duplicate.status_code == 409

    cancelled = client.post(
        f"/api/conversations/{cancellable_conversation['id']}/cancel"
    )
    assert cancelled.json() == {"cancelled": True}
    cancelled_job = main._generations[
        cancellable_start.json()["generation"]["id"]
    ]
    assert cancelled_job.wait(timeout=3)
    assert cancelled_job.snapshot()["status"] == "cancelled"


def test_frontend_uses_per_chat_jobs_and_sidebar_indicators() -> None:
    javascript = Path("app/static/app.js").read_text(encoding="utf-8")
    reliability = Path("app/static/reliability.js").read_text(encoding="utf-8")
    styles = Path("app/static/styles.css").read_text(encoding="utf-8")

    assert "/messages/start" in javascript
    assert "generationConnections" in javascript
    assert "state.busy" not in javascript
    assert "conversation-status-dot" in javascript
    assert "sendMessageWithContent =" not in reliability
    assert "/messages/stream" not in reliability
    assert ".generation-running" in styles
    assert ".generation-unread" in styles
