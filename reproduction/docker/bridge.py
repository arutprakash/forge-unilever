"""Minimal MCP stdio bridge. Commands go ONLY to the offline Unix-socket worker.

No command is executed in the networked/authenticated controller container.
"""
import json
import hashlib
from pathlib import Path
import socket
import sys
import time
from datetime import datetime, timezone

MAX_MESSAGE = 4 * 1024 * 1024


def execute(arguments):
    started = time.monotonic()
    audit_path = Path('/audit/tool-calls.jsonl')
    def log(event, **data):
        with audit_path.open('a') as stream:
            stream.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'event': event, **data}) + '\n')
    log('tool_started', arguments=arguments)
    request = json.dumps(arguments).encode() + b'\n'
    if len(request) > MAX_MESSAGE:
        raise ValueError('Command request exceeds size limit')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(660)
        connection.connect('/ipc/worker.sock')
        connection.sendall(request)
        stream = connection.makefile('rb')
        raw = stream.readline(MAX_MESSAGE + 1)
    if len(raw) > MAX_MESSAGE:
        raise ValueError('Worker response exceeds size limit')
    response = json.loads(raw)
    log('tool_finished', elapsed_seconds=time.monotonic()-started,
        response=response, response_sha256=hashlib.sha256(raw).hexdigest())
    return response


def dispatch(request):
    method = request['method']
    if method == 'initialize':
        return {'protocolVersion': '2024-11-05', 'capabilities': {'tools': {}},
                'serverInfo': {'name': 'offline-worker', 'version': '1.0.0'}}
    if method == 'ping':
        return {}
    if method == 'tools/list':
        return {'tools': [{'name': 'exec',
                          'description': 'Execute a shell command in the isolated, network-disabled worker. Use this for ALL file reads, writes, PDF extraction, training, and tests. Working directory /workspace; inputs /inputs are read-only. No internet or credentials are present.',
                          'inputSchema': {'type': 'object', 'properties': {
                              'command': {'type': 'string'},
                              'timeout': {'type': 'integer', 'minimum': 1, 'maximum': 600}},
                              'required': ['command'], 'additionalProperties': False}}]}
    if method == 'tools/call':
        if request['params']['name'] != 'exec':
            raise ValueError('Unknown tool')
        response = execute(request['params'].get('arguments', {}))
        return {'content': [{'type': 'text', 'text': json.dumps(response)}],
                'isError': response.get('exit_code', 1) != 0}
    raise ValueError('Unsupported MCP method')


def main():
    for line in sys.stdin:
        request = json.loads(line)
        if 'id' not in request:
            continue
        try:
            reply = {'jsonrpc': '2.0', 'id': request['id'], 'result': dispatch(request)}
        except Exception as exc:
            reply = {'jsonrpc': '2.0', 'id': request['id'],
                     'error': {'code': -32603, 'message': str(exc)}}
        print(json.dumps(reply), flush=True)


if __name__ == '__main__':
    main()
