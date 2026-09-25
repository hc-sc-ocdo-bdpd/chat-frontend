import hashlib
import io
import json
import zipfile
from email.parser import BytesParser
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from app.file_bundles import SourceFile, plan_files, upload_bundle
from app.provider import build_response_arguments
from test_response_recovery import app_client, response


def source_files(directory, count, *, kind="generated", content=b"example", name="report.csv"):
    sources = []
    directory.mkdir(parents=True, exist_ok=True)
    for number in range(count):
        identity = f"{kind}-{number}"
        path = directory / identity
        path.write_bytes(content + str(number).encode())
        row = {"id": identity, "original_name": name, "stored_name": identity,
               "size_bytes": path.stat().st_size, "provider_files": {},
               "sandbox_path": f"/mnt/data/project-{number}/report.csv",
               "created_at": 1000 + number}
        sources.append(SourceFile(kind, row, path))
    return sources


@pytest.mark.parametrize("explicit_count", [0, 1, 49, 50, 60])
def test_plan_preserves_every_source_and_never_exceeds_limit(tmp_path, explicit_count):
    explicit = source_files(tmp_path / "input", explicit_count, kind="attachment")
    generated = source_files(tmp_path / "output", 99)
    planned = plan_files(explicit + generated)
    assert len(planned) <= 50
    flattened = [source for item in planned for source in (item if isinstance(item, list) else [item])]
    assert flattened == explicit + generated
    if explicit_count < 50:
        assert planned[:explicit_count] == explicit


def test_fifty_files_stay_individual_and_duplicate_records_take_no_extra_slots(tmp_path):
    sources = source_files(tmp_path, 50)
    assert plan_files(sources + [sources[0]]) == sources
    assert all(isinstance(item, SourceFile) for item in plan_files(sources))


def test_size_limited_archives_split_without_omitting_large_files(tmp_path):
    sources = source_files(tmp_path, 99, content=b"data" * 100)
    sources[0].path.write_bytes(b"large" * 2000)
    plan = plan_files(sources, max_bytes=8000)
    assert 1 < len(plan) <= 50
    assert plan[0] == sources[0]
    assert [source for item in plan for source in (item if isinstance(item, list) else [item])] == sources
    with pytest.raises(ValueError, match="Saved files have not been removed"):
        plan_files(sources, max_bytes=1100)
    assert all(source.path.is_file() for source in sources)


def test_bundle_cache_tracks_content_and_endpoint_and_preserves_colliding_names(tmp_path):
    sources = source_files(tmp_path / "sources", 3, name="../same.csv")
    endpoint = SimpleNamespace(id="azure-openai", base_url="https://first.invalid/")
    uploaded = []

    def upload(_endpoint, path, name):
        uploaded.append(path.read_bytes())
        return f"file-{len(uploaded)}"

    cache_dir = tmp_path / "cache"
    first = upload_bundle(endpoint, sources, cache_dir, upload)
    assert upload_bundle(endpoint, sources, cache_dir, upload) == first
    assert len(uploaded) == 1
    with zipfile.ZipFile(io.BytesIO(uploaded[0])) as archive:
        entries = json.loads(archive.read("index.json"))["files"]
        assert len({entry["path"] for entry in entries}) == 3
        for source, entry in zip(sources, entries):
            assert ".." not in Path(entry["path"]).parts
            assert entry["sandbox_path"] == source.row["sandbox_path"]
            assert archive.read(entry["path"]) == source.path.read_bytes()
    assert all(path.suffix == ".json" for path in cache_dir.iterdir())
    assert all(source.row["provider_files"] == {} for source in sources)
    sources[0].path.write_bytes(b"changed")
    assert upload_bundle(endpoint, sources, cache_dir, upload)[1] == "file-2"
    endpoint.base_url = "https://second.invalid/"
    assert upload_bundle(endpoint, sources, cache_dir, upload)[1] == "file-3"


def test_failed_bundle_upload_keeps_originals_and_does_not_cache_failure(tmp_path):
    sources = source_files(tmp_path / "sources", 2)
    endpoint = SimpleNamespace(id="azure-openai", base_url="https://example.invalid/")
    def fail(*_args):
        raise OSError("connection lost")
    with pytest.raises(OSError, match="connection lost"):
        upload_bundle(endpoint, sources, tmp_path / "cache", fail)
    assert list((tmp_path / "cache").iterdir()) == []
    assert all(source.path.is_file() for source in sources)


def test_provider_deduplicates_ids_and_rejects_overflow_before_request():
    kwargs = dict(model=SimpleNamespace(deployment="test-model"), input_payload="hello",
                  instructions="", reasoning_effort="medium", verbosity="medium",
                  max_output_tokens=1024, use_code_interpreter=True, use_web_search=False,
                  research_depth="quick", web_allowed_domains=[], web_blocked_domains=[],
                  previous_response_id=None)
    ids = [f"file-{number}" for number in range(50)]
    args = build_response_arguments(**kwargs, provider_file_ids=ids + ids)
    assert args["tools"][0]["container"]["file_ids"] == ids
    with pytest.raises(ValueError, match="at most 50"):
        build_response_arguments(**kwargs, provider_file_ids=ids + ["overflow"])


@pytest.mark.parametrize("keep_response_link", [True, False])
def test_chat_with_99_saved_files_continues_and_every_download_survives(app_client, monkeypatch, keep_response_link):
    main, client = app_client
    conversation = client.post("/api/conversations", json={}).json()
    sources = source_files(main.GENERATED_DIR, 99)
    for source in sources:
        source.row["provider_files"] = {"azure-openai": f"file-individual-{source.row['id']}"}
        source.row["sha256"] = hashlib.sha256(source.path.read_bytes()).hexdigest()
    downloads = [main.generated_file_metadata(conversation["id"], source.row) for source in sources]
    main.db.add_message(conversation["id"], "assistant", "Saved reports.",
                        {"response_id": "resp-previous", "generated_files": downloads},
                        generated_files=[source.row for source in sources])
    if keep_response_link:
        main.db.update_conversation(conversation["id"], previous_response_id="resp-previous",
                                    previous_endpoint_id="azure-openai", previous_model_id="test-model")
    else:
        main.db.add_message(conversation["id"], "user", "Read the saved reports.")
        main.db.add_message(conversation["id"], "assistant", "Request failed: too many file IDs",
                            {"error": True, "response_id": None})
    uploaded, requests = [], []

    def handle(request):
        if request.url.path.endswith("/files"):
            multipart = BytesParser().parsebytes(
                f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content)
            file_part = next(part for part in multipart.walk()
                             if part.get_param("name", header="content-disposition") == "file")
            uploaded.append(file_part.get_payload(decode=True))
            return httpx.Response(200, json={"id": "file-bundle", "object": "file",
                "filename": file_part.get_filename(), "bytes": len(uploaded[-1]),
                "created_at": 1, "purpose": "assistants"})
        assert request.method == "POST" and request.url.path.endswith("/responses")
        body = json.loads(request.content)
        requests.append(body)
        assert body["tools"][0]["container"]["file_ids"] == ["file-bundle"]
        assert not any(item["type"] == "input_file" for item in body["input"][-1]["content"])
        assert "index.json" in body["input"][-1]["content"][1]["text"]
        final = response(text="The saved reports are available.", rid=f"resp-{len(requests)}")
        data = {"type": "response.completed", "sequence_number": 0, "response": final.model_dump()}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=("data: " + json.dumps(data) + "\n\n").encode())

    def make_client(_endpoint):
        return OpenAI(api_key="test-key", base_url="https://example.invalid/v1/",
                      http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    monkeypatch.setattr("app.provider.make_client", make_client)
    before = main.db.list_generated_files(conversation["id"])
    for turn in range(2):
        result = client.post(f"/api/conversations/{conversation['id']}/messages/start",
                             json={"content": "Read the saved reports.", "model_id": "test-model"})
        assert result.status_code == 202
        job = main._generations[result.json()["generation"]["id"]]
        assert job.wait(10)
        assert job.snapshot()["status"] == "completed"
        chat = client.get(f"/api/conversations/{conversation['id']}").json()
        diagnostic = client.get(chat["messages"][-1]["metadata"]["diagnostics_url"]).json()
        assert diagnostic["files"] == {"source_count": 99, "provider_count": 1, "bundles": 1}
        previous = "resp-1" if turn else ("resp-previous" if keep_response_link else None)
        assert requests[-1].get("previous_response_id") == previous
        assert "Request failed" not in json.dumps(requests[-1]["input"])
    assert len(uploaded) == 1
    assert len(requests) == 2
    assert main.db.list_generated_files(conversation["id"]) == before
    with zipfile.ZipFile(io.BytesIO(uploaded[0])) as archive:
        entries = json.loads(archive.read("index.json"))["files"]
        assert len(entries) == 99
        for source, entry, download in zip(sources, entries, downloads):
            expected = source.path.read_bytes()
            assert archive.read(entry["path"]) == expected
            result = client.get(download["url"])
            assert result.status_code == 200
            assert result.content == expected
            assert "attachment" in result.headers["content-disposition"]
