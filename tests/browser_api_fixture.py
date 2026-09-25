"""Local FastAPI + real OpenAI SDK fixture. All Azure traffic is simulated."""
import io
import json
import os
import sys
import threading
import zipfile
from pathlib import Path

import httpx
import uvicorn
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('AZURE_OPENAI_BASE_URL', 'https://azure.test/openai/v1/')
os.environ.setdefault('AZURE_OPENAI_API_KEY', 'fixture-only')
os.environ.setdefault('AZURE_OPENAI_DEPLOYMENT_1', 'test-model')
from app import main, artifacts, provider

buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w') as archive:
    archive.writestr(zipfile.ZipInfo('results.csv', date_time=(2020, 1, 1, 0, 0, 0)), 'prediction,actual\n1,1\n')
FILE_BYTES = buffer.getvalue()
PATH = '/mnt/data/report.zip'
TEXT = f'**[Download report]\\(sandbox:{PATH})**'
CODE = {'id': 'ci-1', 'type': 'code_interpreter_call', 'container_id': 'cntr-existing',
        'code': 'print("fixture")', 'outputs': [], 'status': 'completed'}
CITATION = {'type': 'container_file_citation', 'container_id': 'cntr-existing',
            'file_id': 'file-zip', 'filename': PATH, 'start_index': 0, 'end_index': len(TEXT)}
FINAL = {'id': 'resp-running', 'object': 'response', 'created_at': 1, 'status': 'completed',
         'model': 'test-model', 'output': [{'id': 'msg-1', 'type': 'message', 'role': 'assistant',
         'status': 'completed', 'content': [{'type': 'output_text', 'text': TEXT, 'annotations': [CITATION]}]}],
         'usage': {'input_tokens': 10, 'output_tokens': 20, 'total_tokens': 30}}
requests = []


class EventStream(httpx.SyncByteStream):
    def __iter__(self):
        events = [
            {'type': 'response.created', 'sequence_number': 0, 'response': {**FINAL, 'status': 'in_progress', 'output': []}},
            {'type': 'response.output_item.done', 'sequence_number': 1, 'output_index': 0, 'item': CODE},
            {'type': 'response.output_text.delta', 'sequence_number': 2, 'item_id': 'msg-1', 'output_index': 1,
             'content_index': 0, 'delta': 'Saved partial text'}]
        for event in events:
            yield ('data: ' + json.dumps(event) + '\n\n').encode()
        if os.environ.get('FIXTURE_STALL') == '1':
            threading.Event().wait(45)  # Test kills this fixture process after durable checkpoint.
        yield b'data: {"type":"error","sequence_number":3,"code":"server_error","message":"Simulated broken Azure stream"}\n\n'


def handler(request):
    requests.append({'method': request.method, 'path': request.url.path})
    path = request.url.path
    if request.method == 'POST' and path.endswith('/responses'):
        return httpx.Response(200, headers={'content-type': 'text/event-stream', 'apim-request-id': 'fixture-request'}, stream=EventStream())
    if path.endswith('/responses/resp-link'):
        return httpx.Response(200, json={**FINAL, 'id': 'resp-link', 'output': [], 'previous_response_id': 'resp-tools'})
    if path.endswith('/responses/resp-tools'):
        return httpx.Response(200, json={**FINAL, 'id': 'resp-tools', 'output': [CODE]})
    if path.endswith('/responses/resp-running'):
        return httpx.Response(200, json=FINAL)
    if path.endswith('/input_items'):
        return httpx.Response(200, json={'object': 'list', 'data': [], 'has_more': False})
    if path.endswith('/containers/cntr-existing/files'):
        return httpx.Response(200, json={'object': 'list', 'data': [{'id': 'file-zip', 'path': PATH,
            'container_id': 'cntr-existing', 'source': 'assistant', 'bytes': len(FILE_BYTES)}], 'has_more': False})
    if path.endswith('/files/file-zip/content'):
        return httpx.Response(200, content=FILE_BYTES)
    if path.endswith('/files') and request.method == 'POST':
        return httpx.Response(200, json={'id': 'uploaded-file', 'bytes': len(FILE_BYTES), 'created_at': 1,
            'filename': 'input.zip', 'object': 'file', 'purpose': 'assistants', 'status': 'processed'})
    return httpx.Response(404, json={'error': {'message': 'Fixture route not found'}})


def client(endpoint):
    return OpenAI(api_key='fixture-only', base_url='https://azure.test/openai/v1/',
                  http_client=httpx.Client(transport=httpx.MockTransport(handler)))


main.make_client = artifacts.make_client = provider.make_client = client
if not main.db.list_conversations():
    chat = main.db.create_conversation('azure-openai', 'test-model', title='Saved sandbox link')
    main.db.add_message(chat['id'], 'assistant', TEXT, {'endpoint_id': 'azure-openai', 'response_id': 'resp-link'})


@main.app.get('/test/requests')
def fixture_requests():
    return requests


@main.app.get('/test/checkpoint')
def fixture_checkpoint():
    active = main.db.list_active_generation_jobs()
    return [{'id': r['id'], 'response_id': r['diagnostics'].get('response_id'),
             'files': len(main.db.list_generated_files(r['conversation_id']))} for r in active]


if __name__ == '__main__':
    uvicorn.run(main.app, host='127.0.0.1', port=int(sys.argv[1]), log_level='warning')
