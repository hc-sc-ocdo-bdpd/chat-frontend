"""Fit local file collections within the provider's per-request file limit."""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

MAX_CONTAINER_FILES = 50
MAX_BUNDLE_BYTES = 500 * 1024 * 1024


@dataclass(frozen=True)
class SourceFile:
    kind: str
    row: dict[str, Any]
    path: Path


def _entry(source: SourceFile) -> dict[str, Any]:
    # Each record has its own directory so equal basenames never overwrite.
    name = str(source.row["original_name"]).replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "_", name)
    if name in {"", ".", ".."}:
        name = "file"
    identity = hashlib.sha256(str(source.row["id"]).encode()).hexdigest()[:24]
    return {
        "path": f"files/{source.kind}/{identity}/{name}",
        "name": source.row["original_name"],
        "source": source.kind,
        "sandbox_path": source.row.get("sandbox_path"),
        "created_at": source.row.get("created_at"),
    }


def _encoded_index(sources: list[SourceFile]) -> bytes:
    return json.dumps({"files": [_entry(item) for item in sources]},
                      ensure_ascii=True, sort_keys=True).encode("utf-8")


def _pack(sources: list[SourceFile], max_bytes: int) -> list[SourceFile | list[SourceFile]]:
    result: list[SourceFile | list[SourceFile]] = []
    pending: list[SourceFile] = []
    size = 1024

    def flush() -> None:
        nonlocal pending, size
        if pending:
            result.append(pending[0] if len(pending) == 1 else pending)
        pending, size = [], 1024

    for source in sources:
        length = source.path.stat().st_size
        # Bound deflate expansion, ZIP headers, UTF-8 names and the index.
        overhead = len(_encoded_index([source])) * 3 + 1024
        estimate = length + (length >> 12) + (length >> 14) + (length >> 25) + 13 + overhead
        if size + estimate > max_bytes:
            flush()
        if 1024 + estimate > max_bytes:
            result.append(source)
        else:
            pending.append(source)
            size += estimate
    flush()
    return result


def plan_files(sources: list[SourceFile], *, max_bytes: int | None = None
               ) -> list[SourceFile | list[SourceFile]]:
    """Keep explicit inputs separate when space permits; never drop files."""
    unique = {(source.kind, source.row["id"]): source for source in sources}
    sources = list(unique.values())
    if len(sources) <= MAX_CONTAINER_FILES:
        return sources
    max_bytes = MAX_BUNDLE_BYTES if max_bytes is None else max_bytes
    explicit = [source for source in sources if source.kind != "generated"]
    retained = [source for source in sources if source.kind == "generated"]
    planned = explicit + _pack(retained, max_bytes)
    if len(planned) > MAX_CONTAINER_FILES:
        planned = _pack(sources, max_bytes)
    if len(planned) > MAX_CONTAINER_FILES:
        raise ValueError(
            "The files still require more than 50 uploads after ZIP bundling. "
            "Use fewer attachments or active project files, or a separate chat. "
            "Saved files have not been removed."
        )
    return planned


def upload_bundle(endpoint: Any, sources: list[SourceFile], cache_dir: Path,
                  upload: Callable[..., str]) -> tuple[dict[str, Any], str]:
    """Cache upload IDs separately from individual files and discard temporary ZIPs."""
    index = _encoded_index(sources)
    digest = hashlib.sha256(b"chat-file-bundle-v1\0" + index)
    source_hashes = []
    for source in sources:
        content_hash = hashlib.sha256()
        with source.path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                content_hash.update(block)
        source_hashes.append(content_hash.hexdigest())
        digest.update(content_hash.digest())
    signature = digest.hexdigest()
    filename = f"context-{signature[:24]}.zip"
    row = {"original_name": filename, "bundle_count": len(sources)}
    # A provider ID belongs to an endpoint, even if its display ID is unchanged.
    cache_key = hashlib.sha256(
        f"{endpoint.id}\0{endpoint.base_url}\0{signature}".encode()
    ).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.json"
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(cached.get("file_id"), str) and cached["file_id"]:
            return row, cached["file_id"]
    except (OSError, ValueError, AttributeError):
        pass

    with tempfile.TemporaryDirectory(prefix="bundle-", dir=cache_dir) as temporary:
        archive_path = Path(temporary) / filename
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED,
                             allowZip64=True) as archive:
            archive.writestr("index.json", index)
            for source, expected_hash in zip(sources, source_hashes):
                content_hash = hashlib.sha256()
                with source.path.open("rb") as incoming, archive.open(
                    _entry(source)["path"], "w", force_zip64=True
                ) as outgoing:
                    for block in iter(lambda: incoming.read(1024 * 1024), b""):
                        content_hash.update(block)
                        outgoing.write(block)
                if content_hash.hexdigest() != expected_hash:
                    raise ValueError("A file changed while preparing its ZIP. Please retry.")
        if archive_path.stat().st_size > MAX_BUNDLE_BYTES:
            raise ValueError("A file bundle exceeds the upload size limit. Saved files are intact.")
        provider_id = upload(endpoint, archive_path, filename)
        cache_temp = Path(temporary) / "cache.json"
        cache_temp.write_text(json.dumps({"file_id": provider_id}), encoding="utf-8")
        cache_temp.replace(cache_path)
    return row, provider_id


def bundle_prompt(named_files: list[tuple[dict[str, Any], str]]) -> str:
    bundles = [{"file_id": provider_id, "filename": row["original_name"],
                "file_count": row["bundle_count"]}
               for row, provider_id in named_files if row.get("bundle_count")]
    if not bundles:
        return ""
    return (
        "Attached file collections are available in Code Interpreter as ZIP archives: "
        + json.dumps(bundles)
        + ". Use Python zipfile to read index.json in each archive. It maps every file's "
        "archive path to its original name, source, saved time and original sandbox path. "
        "Read the entries needed for the task directly from the ZIP, or restore their "
        "original directory structure when running code. Files with the same name are "
        "stored separately; use their paths and saved times to select the intended version."
    )
