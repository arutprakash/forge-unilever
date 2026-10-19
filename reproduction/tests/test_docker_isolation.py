"""Opt-in Docker integration test; requires built images, no Codex inference.

RUN_DOCKER_TESTS=1 python3 -m unittest discover -s tests -v
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline


@unittest.skipUnless(os.environ.get('RUN_DOCKER_TESTS') == '1', 'Docker integration is opt-in')
class DockerIsolation(unittest.TestCase):
    def test_worker_has_no_egress_credentials_or_author_repo(self):
        pipeline.configure_images(pipeline.load_config(Path(__file__).resolve().parents[1] / 'experiment.json'))
        token = uuid.uuid4().hex[:10]
        name, volume = 'p12-test-' + token, 'p12-test-ipc-' + token

        def docker(args, **kwargs):
            return subprocess.run(args, text=True, capture_output=True, check=True, timeout=40, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace, inputs = root / 'generated', root / 'inputs'
            workspace.mkdir()
            inputs.mkdir()
            (inputs / 'protocol.json').write_text('{}')
            (root / 'AUTHOR_SENTINEL').write_text('not mounted')
            try:
                docker(['docker', 'volume', 'create', volume])
                docker(pipeline.hardened(name+'-init', user='0:0') + ['--rm']
                       + pipeline.mount(volume, '/ipc', volume=True)
                       + [pipeline.IMAGES['runtime'], 'python', '-c', 'import os; os.chmod("/ipc",0o777)'])
                docker(pipeline.worker_command(name, workspace, inputs, volume))
                info = json.loads(docker(['docker', 'inspect', name]).stdout)[0]
                pipeline.assert_worker_policy(info, 'paper-only')
                for _ in range(30):
                    result = subprocess.run(['docker', 'exec', name, 'test', '-S', '/ipc/worker.sock'], capture_output=True)
                    if result.returncode == 0:
                        break
                    time.sleep(0.2)
                probe = '''import socket,os,json
assert all(name=='lo' or not (int(open('/sys/class/net/'+name+'/flags').read(),16)&1) for _,name in socket.if_nameindex())
assert not open('/proc/net/route').read().splitlines()[1:]
for path in ['/author-repo','/run/login.json','/login','/var/run/docker.sock','/AUTHOR_SENTINEL']:
    assert not os.path.exists(path), path
assert 'OPENAI_API_KEY' not in os.environ
try:
    connection=socket.create_connection(('1.1.1.1',443),timeout=2)
except OSError:
    pass
else:
    connection.close()
    raise AssertionError('External network reachable')
with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
    client.connect('/ipc/worker.sock')
    client.sendall(json.dumps({'command':'printf offline-bridge-ok'}).encode()+b'\\n')
    response=json.loads(client.makefile('rb').readline())
assert response['output']=='offline-bridge-ok',response
print('container isolation and worker RPC passed')
'''
                result = docker(['docker', 'exec', name, 'python', '-c', probe])
                self.assertIn('container isolation and worker RPC passed', result.stdout)
            finally:
                subprocess.run(['docker', 'rm', '-f', name], capture_output=True)
                subprocess.run(['docker', 'volume', 'rm', volume], capture_output=True)
