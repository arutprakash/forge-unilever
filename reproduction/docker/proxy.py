"""CONNECT-only egress proxy for the Codex client, NOT for offline workers.

Exact inference/auth host allowlist. No wildcard domains, HTTP forwarding,
redirect following, custom port, GitHub, package registries, or general web.
"""
import select
import socket
import socketserver

ALLOWED = frozenset({'chatgpt.com', 'auth.openai.com', 'api.openai.com'})


def allowed_target(method, target):
    return method == 'CONNECT' and target in {f'{name}:443' for name in ALLOWED}


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        first = self.rfile.readline(8193).decode('ascii', errors='replace').strip().split()
        if len(first) != 3:
            return
        method, target, _ = first
        # Read bounded headers, never log their contents.
        total = 0
        while True:
            line = self.rfile.readline(8193)
            total += len(line)
            if total > 32768:
                return
            if line in (b'\r\n', b'\n', b''):
                break
        if not allowed_target(method, target):
            self.wfile.write(b'HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n')
            print(f'deny {method} {target[:200]}', flush=True)
            return
        print(f'allow CONNECT {target}', flush=True)
        try:
            with socket.create_connection((target[:-4], 443), timeout=30) as upstream:
                self.wfile.write(b'HTTP/1.1 200 Connection established\r\n\r\n')
                self.wfile.flush()
                while True:
                    ready, _, _ = select.select([self.connection, upstream], [], [], 300)
                    if not ready:
                        return
                    for source in ready:
                        data = source.recv(65536)
                        if not data:
                            return
                        destination = upstream if source is self.connection else self.connection
                        destination.sendall(data)
        except OSError:
            return


if __name__ == '__main__':
    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
    with Server(('0.0.0.0', 3128), Handler) as server:
        server.serve_forever()
