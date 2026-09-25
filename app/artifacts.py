"""Conversation-scoped discovery and durable retention of Azure sandbox files."""
from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import re
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from .diagnostics import field
from .provider import _container_path, make_client

log = logging.getLogger(__name__)


def raw(value):
    return value if isinstance(value, dict) else value.model_dump()


def references(value):
    containers, citations, paths = [], [], set()
    def visit(item):
        if isinstance(item, dict):
            kind = item.get('type')
            cid = item.get('container_id')
            if kind in {'code_interpreter_call', 'container_file_citation'} and cid:
                if cid not in containers:
                    containers.append(cid)
                if kind == 'container_file_citation' and item.get('file_id'):
                    name = item.get('filename') or item['file_id']
                    citations.append({'container_id': cid, 'file_id': item['file_id'],
                                      'filename': name, 'sandbox_path': _container_path(name) or None})
            if kind == 'output_text':
                paths.update(linked_paths(item.get('text', '')))
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(raw(value) if not isinstance(value, (dict, list)) else value)
    return containers, citations, paths


def linked_paths(text):
    # Includes the model's escaped Markdown parentheses; filenames are case-sensitive.
    return {path for match in re.finditer(r'\[[^\]\n]+\]\\?\((sandbox:/+[^)\n]+?)\\?\)', text, re.I)
            if (path := _container_path(match[1]))}


def list_sources(client, containers, paths=None):
    found = []
    for cid in containers:
        for item in client.containers.files.list(cid, limit=100):
            path = _container_path(field(item, 'path', ''))
            if not path or (paths is not None and path not in paths):
                continue
            if paths is None and field(item, 'source') != 'assistant':
                continue  # Do not duplicate uploaded inputs during early capture.
            found.append({'container_id': cid, 'file_id': field(item, 'id'),
                          'filename': Path(path).name, 'sandbox_path': path,
                          'expected_bytes': field(item, 'bytes')})
    return found


def find_in_history(endpoint, response_ids, target, known_containers=()):
    """GET only. Never searches unrelated chats or creates a model response."""
    pending = deque(dict.fromkeys(response_ids))
    seen_responses, seen_containers = set(), set()
    transient_error = None
    deadline = time.monotonic() + 90
    with make_client(endpoint).with_options(timeout=15.0, max_retries=0) as client:
        def inspect(containers):
            nonlocal transient_error
            candidates = {}
            for cid in containers:
                if cid in seen_containers or time.monotonic() > deadline:
                    continue
                seen_containers.add(cid)
                try:
                    for source in list_sources(client, [cid], {target}):
                        candidates[(cid, source['file_id'])] = source
                except Exception as exc:
                    if getattr(exc, 'status_code', None) not in {404, 410}:
                        transient_error = exc
            if len(candidates) > 1:
                raise ValueError('Several containers contain this exact path. Recovery cannot safely choose one.')
            return next(iter(candidates.values()), None)

        while pending and len(seen_responses) < 30 and time.monotonic() < deadline:
            rid = pending.popleft()
            if not rid or rid in seen_responses:
                continue
            seen_responses.add(rid)
            try:
                response = raw(client.responses.retrieve(rid))
            except Exception as exc:
                if getattr(exc, 'status_code', None) not in {404, 410}:
                    transient_error = exc
                continue
            match = inspect(references(response)[0])
            if match:
                return match
            previous = response.get('previous_response_id')
            if previous and previous not in seen_responses:
                pending.appendleft(previous)
            # Azure can expose prior code calls only through the input-items API.
            try:
                for page in client.responses.input_items.list(rid, limit=100, order='desc').iter_pages():
                    match = inspect(references([raw(item) for item in page.data])[0])
                    if match:
                        return match
                    if time.monotonic() > deadline:
                        break
            except ValueError:
                raise
            except Exception as exc:
                if getattr(exc, 'status_code', None) not in {404, 410}:
                    transient_error = exc
        match = inspect(known_containers)
        if match:
            return match
    if transient_error is not None:
        raise RuntimeError('Azure file lookup is temporarily unavailable. Your locally saved files are unaffected.') from transient_error
    raise FileNotFoundError('This file was not found in the saved Azure responses. Its sandbox may have expired, or the model may not have created it.')


def retain_sources(endpoint, sources, directory, conversation_id, download, *, database=None, owner_id=None, errors=None):
    """Save each successful file independently, with atomic writes and hashes."""
    records = []
    directory = Path(directory)
    (directory / conversation_id).mkdir(parents=True, exist_ok=True)
    seen = set()
    existing = database.list_generated_files(conversation_id) if database else []
    for source in sources:
        key = (source['container_id'], source['file_id'])
        if key in seen:
            continue
        seen.add(key)
        temporary = destination = None
        try:
            upstream = None
            for attempt in range(3):
                try:
                    upstream = download(endpoint, *key)
                    break
                except Exception as exc:
                    if getattr(exc, 'status_code', None) in {400, 401, 403, 404, 410} or attempt == 2:
                        raise
                    time.sleep(.25 * 2 ** attempt)
            content = bytes(upstream.content)
            expected = source.get('expected_bytes')
            if expected is not None and len(content) != int(expected):
                raise ValueError('Downloaded file size does not match Azure metadata')
            digest = hashlib.sha256(content).hexdigest()
            prior = next((f for f in existing if f.get('source_endpoint_id') == endpoint.id
                          and f.get('source_container_id') == key[0] and f.get('source_file_id') == key[1]
                          and f.get('sha256') == digest and (directory / f['stored_name']).is_file()), None)
            if prior:
                prior['sandbox_path'] = source.get('sandbox_path') or prior.get('sandbox_path')
                records.append(prior)
                continue
            filename = re.sub(r'[^\w.\- ()]', '_', Path(source['filename']).name)[:200] or 'download'
            retained_id = str(uuid.uuid4())
            stored_name = str(Path(conversation_id) / f'{retained_id}_{filename}')
            destination = directory / stored_name
            temporary = destination.with_suffix(destination.suffix + '.part')
            with temporary.open('wb') as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(destination)
            record = {'id': retained_id, 'original_name': filename, 'stored_name': stored_name,
                      'content_type': mimetypes.guess_type(filename)[0] or upstream.headers.get('content-type') or 'application/octet-stream',
                      'size_bytes': len(content), 'sha256': digest, 'sandbox_path': source.get('sandbox_path'),
                      'source_endpoint_id': endpoint.id, 'source_container_id': key[0], 'source_file_id': key[1],
                      'provider_files': {}, 'created_at': time.time()}
            if database and owner_id:
                database.stage_generated_file(conversation_id, owner_id, record)
                record['message_id'] = owner_id
            records.append(record)
        except Exception as exc:
            if temporary:
                temporary.unlink(missing_ok=True)
            if destination:
                destination.unlink(missing_ok=True)
            log.warning('Could not retain file %s: %s', source.get('filename'), type(exc).__name__)
            if errors is not None:
                errors.append({'filename': source.get('filename'), 'error': type(exc).__name__})
    return records
