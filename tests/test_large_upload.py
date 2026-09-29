import io
import zipfile
from types import SimpleNamespace

import pytest

import app.provider as provider


class FakeFiles:
    def __init__(self):
        self.created = []
        self.deleted = []

    def create(self, *, file, purpose):
        filename, handle = file
        self.created.append((filename, handle.read(), purpose))
        return SimpleNamespace(id=f"file-{len(self.created)}")

    def delete(self, file_id):
        self.deleted.append(file_id)


class FakeClient:
    def __init__(self):
        self.files = FakeFiles()
        self.options = []
        self.closed = False

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    def close(self):
        self.closed = True


def test_large_zip_uses_standard_file_shards_and_reconstructs_exactly(tmp_path, monkeypatch):
    source = tmp_path / "redaction_lab.zip"
    source.write_bytes(bytes(range(256)) * 100)
    client = FakeClient()

    monkeypatch.setattr(provider, "LARGE_FILE_SHARD_THRESHOLD_BYTES", 1000)
    monkeypatch.setattr(provider, "LARGE_FILE_SHARD_BYTES", 4096)
    monkeypatch.setattr(provider, "make_client", lambda _endpoint: client)

    token = provider.upload_file(SimpleNamespace(), source, "redaction_lab.zip")
    shard = provider._decode_sharded_file(token)

    assert shard is not None
    assert len(shard["file_ids"]) == 7
    assert client.options == [{"max_retries": 0}]
    assert len(client.files.created) == 7
    assert client.files.created[0][0] == "redaction_lab.zip"

    reconstructed = b""
    for filename, data, purpose in client.files.created:
        assert purpose == "assistants"
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            assert provider.SHARD_MANIFEST_NAME in archive.namelist()
            payload = next(name for name in archive.namelist() if name.startswith("payload-"))
            reconstructed += archive.read(payload)

    assert reconstructed == source.read_bytes()
    assert client.files.deleted == []
    assert client.closed is True


def test_sharded_cached_value_expands_into_container_file_ids(monkeypatch):
    token = provider._encode_sharded_file(
        original_name="redaction_lab.zip",
        file_ids=["file-a", "file-b", "file-c"],
        shard_names=["redaction_lab.zip", "redaction_lab.zip.part002.zip", "redaction_lab.zip.part003.zip"],
    )
    model = SimpleNamespace(deployment="test-model")

    arguments = provider.build_response_arguments(
        model=model,
        input_payload="analyze it",
        instructions="base instructions",
        reasoning_effort="max",
        verbosity="high",
        max_output_tokens=128000,
        use_code_interpreter=True,
        provider_file_ids=["file-normal", token],
        use_web_search=False,
        research_depth="quick",
        web_allowed_domains=[],
        web_blocked_domains=[],
        previous_response_id=None,
    )

    assert arguments["tools"][0]["container"]["file_ids"] == [
        "file-normal", "file-a", "file-b", "file-c"
    ]
    assert "transport-sharded" in arguments["instructions"]
    assert "redaction_lab.zip" in arguments["instructions"]


def test_partial_shard_failure_cleans_up_uploaded_remote_parts(tmp_path, monkeypatch):
    source = tmp_path / "large.zip"
    source.write_bytes(b"a" * 100)
    client = FakeClient()
    original_create = client.files.create

    def fail_third(**kwargs):
        if len(client.files.created) == 2:
            raise RuntimeError("shard upload failed")
        return original_create(**kwargs)

    client.files.create = fail_third
    monkeypatch.setattr(provider, "LARGE_FILE_SHARD_THRESHOLD_BYTES", 1)
    monkeypatch.setattr(provider, "LARGE_FILE_SHARD_BYTES", 25)
    monkeypatch.setattr(provider, "make_client", lambda _endpoint: client)

    with pytest.raises(RuntimeError, match="shard upload failed"):
        provider.upload_file(SimpleNamespace(), source, "large.zip")

    assert client.files.deleted == ["file-1", "file-2"]


def test_small_file_still_uses_one_standard_upload_without_retries(tmp_path, monkeypatch):
    source = tmp_path / "small.txt"
    source.write_text("hello", encoding="utf-8")
    client = FakeClient()

    monkeypatch.setattr(provider, "LARGE_FILE_SHARD_THRESHOLD_BYTES", 1024)
    monkeypatch.setattr(provider, "make_client", lambda _endpoint: client)

    result = provider.upload_file(SimpleNamespace(), source, "small.txt")

    assert result == "file-1"
    assert client.options == [{"max_retries": 0}]
    assert len(client.files.created) == 1
