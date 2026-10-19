"""Offline command server. Docker, not a prompt, enforces its network boundary."""
import json
import os
from pathlib import Path
import signal
import socketserver
import subprocess
import tempfile
import time
import resource

MAX_COMMAND = 1024 * 1024
MAX_OUTPUT = 60000


def run_command(request):
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    command = request.get('command')
    timeout = request.get('timeout', 120)
    if not isinstance(command, str) or len(command.encode()) > MAX_COMMAND:
        raise ValueError('Expected a command smaller than 1 MiB')
    if not isinstance(timeout, int) or not 1 <= timeout <= 600:
        raise ValueError('timeout must be 1..600 seconds')
    # Never inherit the controller's environment or login. This server and its
    # children live in the offline worker, which has its own mounts and PID ns.
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(['/bin/bash', '-lc', command], cwd='/workspace',
                                   stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        output.seek(0, 2)
        size = output.tell()
        output.seek(max(0, size - MAX_OUTPUT))
        text = output.read(MAX_OUTPUT).decode(errors='replace')
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {'exit_code': process.returncode, 'timed_out': timed_out,
            'output_truncated': size > MAX_OUTPUT, 'output': text,
            'elapsed_seconds': time.monotonic()-started,
            'child_cpu_seconds': after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
            'worker_cumulative_child_peak_rss_kib': after.ru_maxrss}


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            raw = self.rfile.readline(MAX_COMMAND + 1)
            if len(raw) > MAX_COMMAND:
                raise ValueError('Oversized request')
            response = run_command(json.loads(raw))
        except Exception as exc:
            response = {'exit_code': 1, 'output': str(exc)}
        self.wfile.write(json.dumps(response).encode() + b'\n')


def main():
    Path('/tmp/home').mkdir(exist_ok=True)
    socket_path = Path('/ipc/worker.sock')
    socket_path.unlink(missing_ok=True)
    with socketserver.ThreadingUnixStreamServer(str(socket_path), Handler) as server:
        os.chmod(socket_path, 0o666)
        print('offline worker ready', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
