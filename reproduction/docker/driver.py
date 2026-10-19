"""Start one fresh Codex process. Never run model code in this container."""
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    for directory in ('/tmp/home', '/tmp/session', '/tmp/codex'):
        Path(directory).mkdir(parents=True, exist_ok=True)
    # Copy only the login cache into disposable tmpfs so token refresh works.
    # No host config, memories, sessions, hooks, MCPs, skills, or plugins.
    shutil.copyfile('/login/cache/auth.json', '/tmp/codex/auth.json')
    os.chmod('/tmp/codex/auth.json', 0o600)
    shutil.copyfile('/opt/pipeline/config.toml', '/tmp/codex/config.toml')
    if os.environ.get('FORGE_JUDGE_ONLY') == '1':
        # Post-run judging receives text only. No MCP or execution tools, and
        # no author/generated workspace is mounted into this controller.
        config = Path('/tmp/codex/config.toml')
        config.write_text(config.read_text().split('[mcp_servers.offline]')[0])
    command = ['codex', 'exec', '--ephemeral', '--skip-git-repo-check',
               '--ignore-rules', '--strict-config', '--color', 'never', '--json',
               '-C', '/tmp/session']
    if len(sys.argv) > 1:
        command += ['--model', sys.argv[1]]
    command += ['-']
    try:
        return subprocess.call(command, stdin=sys.stdin.buffer)
    finally:
        # Carry only refreshed login state to the next stage, never sessions or
        # config. This disposable volume is not mounted in either worker.
        shutil.copyfile('/tmp/codex/auth.json', '/login/cache/auth.json')
        os.chmod('/login/cache/auth.json', 0o600)


if __name__ == '__main__':
    raise SystemExit(main())
