import importlib
import json
import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from app import artifacts
from app.db import Database
from app.jobs import GenerationJob
from app.response_stream import ResumableResponseStream
from test_response_recovery import app_client, response, event


def message(main, cid, path='/mnt/data/result.zip', rid='resp-new'):
    return main.db.add_message(cid, 'assistant', f'**[Download]\\(sandbox:{path})**',
        {'response_id': rid, 'endpoint_id': 'azure-openai'})


def sdk_client(handler):
    return OpenAI(api_key='test', base_url='https://azure.test/openai/v1/',
                  http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0)


def test_history_recovery_paginates_downloads_and_survives_restart_without_azure(app_client, monkeypatch):
    main, client = app_client
    cid = client.post('/api/conversations', json={}).json()['id']
    msg = message(main, cid)
    calls = []
    content = b'PK\x03\x04real ZIP bytes'
    def handler(req):
        calls.append((req.method, req.url.path, dict(req.url.params)))
        assert req.method == 'GET', 'Recovery must never submit paid generation'
        path = req.url.path
        if path.endswith('/responses/resp-new'):
            return httpx.Response(200, json={'id': 'resp-new', 'status': 'completed', 'output': [], 'previous_response_id': 'resp-old'})
        if path.endswith('/responses/resp-old'):
            return httpx.Response(200, json={'id': 'resp-old', 'status': 'completed', 'output': [
                {'id': 'code', 'type': 'code_interpreter_call', 'container_id': 'cntr-old', 'status': 'completed', 'code': '', 'outputs': []}]})
        if path.endswith('/input_items'):
            return httpx.Response(200, json={'object': 'list', 'data': [], 'has_more': False})
        if path.endswith('/containers/cntr-old/files'):
            if req.url.params.get('after') == 'file-other':
                return httpx.Response(200, json={'object': 'list', 'data': [{'id': 'file-zip', 'path': '/mnt/data/result.zip',
                    'source': 'assistant', 'bytes': len(content)}], 'has_more': False, 'last_id': 'file-zip'})
            return httpx.Response(200, json={'object': 'list', 'data': [{'id': 'file-other', 'path': '/mnt/data/other.txt'}],
                                           'has_more': True, 'last_id': 'file-other'})
        if path.endswith('/files/file-zip/content'):
            return httpx.Response(200, content=content)
        return httpx.Response(404, json={'error': {'message': path}})
    monkeypatch.setattr(artifacts, 'make_client', lambda endpoint: sdk_client(handler))
    import app.provider as provider
    monkeypatch.setattr(provider, 'make_client', lambda endpoint: sdk_client(handler))
    url = f'/api/conversations/{cid}/messages/{msg["id"]}/files/recover'
    result = client.post(url, json={'sandbox_path': 'sandbox:/mnt/data/result.zip'})
    assert result.status_code == 200, result.text
    download = result.json()
    assert client.get(download['url']).content == content
    assert any(params.get('after') == 'file-other' for _, _, params in calls)
    row = main.db.get_generated_file(download['id'])
    assert row['sha256'] and row['sandbox_path'] == '/mnt/data/result.zip'
    # New app/database instance; no Azure connection is necessary for retained files.
    monkeypatch.setattr(artifacts, 'make_client', lambda endpoint: pytest.fail('Local download must not contact Azure'))
    main.db = Database(main.DATA_DIR / 'app.db')
    main._generations.clear()
    fresh = TestClient(main.app)
    assert fresh.get(download['url']).content == content
    assert fresh.post(url, json={'sandbox_path': '/mnt/data/result.zip'}).json()['id'] == download['id']
    assert len(main.db.list_generated_files(cid)) == 1


def test_file_recovery_rejects_other_chats_and_unlinked_paths_before_network(app_client, monkeypatch):
    main, client = app_client
    one = client.post('/api/conversations', json={}).json()['id']
    two = client.post('/api/conversations', json={}).json()['id']
    msg = message(main, one)
    monkeypatch.setattr(artifacts, 'make_client', lambda endpoint: pytest.fail('Invalid request contacted Azure'))
    assert client.post(f'/api/conversations/{two}/messages/{msg["id"]}/files/recover', json={'sandbox_path': '/mnt/data/result.zip'}).status_code == 404
    for path in ['/etc/passwd', '/mnt/data/../../etc/passwd', '/mnt/data/private.zip']:
        assert client.post(f'/api/conversations/{one}/messages/{msg["id"]}/files/recover', json={'sandbox_path': path}).status_code == 400


def test_one_failed_file_does_not_discard_successful_answer_or_other_files(app_client, monkeypatch):
    main, client = app_client
    citations = [{'type': 'container_file_citation', 'container_id': 'cntr', 'file_id': name, 'filename': name + '.csv'} for name in ['good', 'bad']]
    raw = {'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Your results.', 'annotations': citations}]}]}
    final = SimpleNamespace(id='resp-files', output_text='Your results.', status='completed', model_dump=lambda: raw)
    monkeypatch.setattr(main, 'stream_response', lambda **kw: iter([event('response.completed', response=final)]))
    def download(endpoint, cid, fid):
        if fid == 'bad':
            raise httpx.HTTPStatusError('Not available', request=httpx.Request('GET', 'https://azure.test'), response=httpx.Response(404))
        return httpx.Response(200, content=b'value\n42\n')
    monkeypatch.setattr(main, 'download_generated_file', download)
    cid = client.post('/api/conversations', json={}).json()['id']
    started = client.post(f'/api/conversations/{cid}/messages/start', json={'content': 'Make files'}).json()
    job = main._generations[started['generation']['id']]
    assert job.wait(5)
    assert job.status == 'completed'
    msg = main.db.list_messages(cid)[-1]
    assert msg['content'] == 'Your results.'
    assert len(msg['metadata']['generated_files']) == 1
    assert msg['metadata']['file_errors'][0]['filename'] == 'bad.csv'
    assert client.get(msg['metadata']['generated_files'][0]['url']).content == b'value\n42\n'


@pytest.mark.parametrize("cancel", [False, True])
def test_code_files_are_checkpointed_before_generation_finishes(app_client, monkeypatch, cancel):
    main, client = app_client
    saved = threading.Event()
    release = threading.Event()
    ci = {'id': 'code', 'type': 'code_interpreter_call', 'container_id': 'cntr-early'}
    actual_stage = main.db.stage_generated_file
    def stage(*args):
        actual_stage(*args)
        saved.set()
    monkeypatch.setattr(main.db, 'stage_generated_file', stage)
    fake = SimpleNamespace(with_options=lambda **kw: nullcontext(SimpleNamespace(containers=SimpleNamespace(files=SimpleNamespace(
        list=lambda *a, **kw: [SimpleNamespace(id='file-early', path='/mnt/data/early.csv', source='assistant', bytes=3)])))))
    monkeypatch.setattr(main, 'make_client', lambda endpoint: fake)
    monkeypatch.setattr(main, 'download_generated_file', lambda *args: httpx.Response(200, content=b'123'))
    def stream(**kwargs):
        yield event('response.output_item.done', item=SimpleNamespace(model_dump=lambda: ci))
        release.wait(5)
        yield event('response.failed', response=response('failed', '', error={'code': 'server_error', 'message': 'Late Azure failure'}))
    monkeypatch.setattr(main, 'stream_response', stream)
    cid = client.post('/api/conversations', json={}).json()['id']
    started = client.post(f'/api/conversations/{cid}/messages/start', json={'content': 'Make early file'}).json()
    job = main._generations[started['generation']['id']]
    try:
        assert saved.wait(3), 'File should be on disk while Azure is still thinking'
        row = main.db.list_generated_files(cid)[0]
        assert (main.GENERATED_DIR / row['stored_name']).read_bytes() == b'123'
        assert job.status == 'running'
    finally:
        if cancel:
            job.request_cancel()
        release.set()
    assert job.wait(5)
    final = main.db.list_messages(cid)[-1]
    assert final['metadata']['stopped' if cancel else 'error'] is True
    assert client.get(final['metadata']['generated_files'][0]['url']).content == b'123'


def test_ambiguous_exact_paths_are_not_guessed():
    api = SimpleNamespace(containers=SimpleNamespace(files=SimpleNamespace(list=lambda cid, **kw: [
        SimpleNamespace(id=f'file-{cid}', path='/mnt/data/a.zip', bytes=4)])),
        responses=SimpleNamespace(retrieve=lambda rid: {'id': rid, 'output': [
            {'type': 'code_interpreter_call', 'container_id': cid} for cid in ['c1', 'c2']]}))
    from unittest.mock import patch
    with patch.object(artifacts, 'make_client', return_value=SimpleNamespace(with_options=lambda **kw: nullcontext(api))):
        with pytest.raises(ValueError, match='Several containers'):
            artifacts.find_in_history(None, ['resp'], '/mnt/data/a.zip')


def test_truncated_download_is_rejected_without_partial_file(tmp_path):
    errors = []
    records = artifacts.retain_sources(SimpleNamespace(id='e'), [{'container_id': 'c', 'file_id': 'f', 'filename': 'file.zip', 'expected_bytes': 100}],
        tmp_path, 'chat', lambda *args: httpx.Response(200, content=b'short'), errors=errors)
    assert records == [] and errors
    assert list((tmp_path / 'chat').iterdir()) == []


def test_bounded_event_log_resynchronizes_slow_or_restarted_clients():
    job = GenerationJob({'id': 'j', 'conversation_id': 'c', 'snapshot': {'assistant_text': 'saved', 'last_sequence': 42}}, {'id': 'u'})
    for _ in range(2500):
        job.publish({'type': 'output_delta', 'delta': 'x'})
    job.publish({'type': 'done', 'assistant_message': {'id': 'a'}})
    assert len(job._events) == 2048
    for cursor in (0, 10000):
        events = [json.loads(e[6:]) for e in job.event_stream(cursor)]
        assert events[0]['type'] == 'snapshot'
        assert events[0]['snapshot']['assistant_text'] == 'saved' + 'x' * 2500
        assert events[0]['snapshot']['status'] == 'completed'


@pytest.mark.parametrize("already_saved", [False, True])
def test_restart_reattaches_by_get_and_keeps_the_same_job(app_client, monkeypatch, already_saved):
    main, client = app_client
    cid = client.post('/api/conversations', json={}).json()['id']
    user = main.db.add_message(cid, 'user', 'Long task')
    row = main.db.create_generation_job(cid, user['id'])
    main.db.update_generation_job(row['id'], status='running', request={'payload': {'content': 'Long task'},
        'endpoint_id': 'azure-openai', 'model_id': 'test-model'},
        diagnostics={'response_id': 'resp-saved', 'endpoint_id': 'azure-openai', 'started_at': 1},
        snapshot={'assistant_text': 'Partial', 'reasoning_summary': 'Saved reasoning', 'last_sequence': 500})
    if already_saved:
        import uuid
        main.db.add_message(cid, 'assistant', 'Saved before the process died',
                            message_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"foundry-chat/{row['id']}")))
    requests = []
    def handler(req):
        requests.append((req.method, req.url.path))
        assert req.method == 'GET'
        final = response(text='Recovered full answer', rid='resp-saved').model_dump()
        return httpx.Response(200, json=final)
    monkeypatch.setattr(main, 'make_client', lambda endpoint: sdk_client(handler))
    original = ResumableResponseStream.resume_existing
    def fast_resume(*args, **kwargs):
        stream = original(*args, **kwargs)
        monkeypatch.setattr(stream._stop, 'wait', lambda seconds: stream._stop.is_set())
        return stream
    monkeypatch.setattr(main.ResumableResponseStream, 'resume_existing', fast_resume)
    with TestClient(main.app) as restarted:
        job = main._generations[row['id']]
        assert job.wait(5)
        assert job.status == 'completed'
        messages = restarted.get(f'/api/conversations/{cid}').json()['messages']
        assert len(messages) == 2 and messages[-1]['content'] == 'Recovered full answer'
        assert messages[-1]['metadata']['reasoning_summary'] == 'Saved reasoning'
        assert job.snapshot()['last_sequence'] > 500
        assert requests == [('GET', '/openai/v1/responses/resp-saved')]


def test_restart_with_unknown_create_outcome_never_submits_again(app_client, monkeypatch):
    main, client = app_client
    cid = client.post('/api/conversations', json={}).json()['id']
    user = main.db.add_message(cid, 'user', 'Work')
    row = main.db.create_generation_job(cid, user['id'])
    main.db.update_generation_job(row['id'], snapshot={'assistant_text': 'Saved partial output'})
    monkeypatch.setattr(main, 'make_client', lambda endpoint: pytest.fail('Must not recreate unknown request'))
    with TestClient(main.app):
        stored = main.db.get_generation_job(row['id'])
        assert stored['status'] == 'failed'
        msg = main.db.get_message(stored['assistant_message_id'])
        assert 'Saved partial output' in msg['content']
        assert 'No new generation was submitted' in msg['content']


def test_database_initialization_preserves_retained_file_records(tmp_path):
    # Initialize a database whose file records do not include optional path metadata.
    import sqlite3
    path = tmp_path / 'legacy.db'
    with sqlite3.connect(path) as connection:
        connection.executescript('''
            CREATE TABLE generated_files (id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
                message_id TEXT NOT NULL, original_name TEXT NOT NULL, stored_name TEXT NOT NULL,
                content_type TEXT, size_bytes INTEGER NOT NULL, source_endpoint_id TEXT,
                source_container_id TEXT, source_file_id TEXT, provider_files_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL);
            INSERT INTO generated_files VALUES ('retained', 'chat', 'message', 'old.zip', 'chat/old.zip',
                'application/zip', 3, 'azure-openai', 'cntr-old', 'file-old', '{}', 123);
        ''')
    database = Database(path)
    row = database.get_generated_file('retained')
    assert row['original_name'] == 'old.zip' and row['size_bytes'] == 3
    assert row['sandbox_path'] is None and row['source_file_id'] == 'file-old'
    assert Database(path).get_generated_file('retained') == row  # Repeated startup is safe.
