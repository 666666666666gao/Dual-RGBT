"""Create a separate conda CPU scorer without touching the active NN environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--protocol', required=True)
    args = ap.parse_args()
    root = Path(args.project).resolve()
    spec_path = root / '.aris/compute/cpu-scoring-env.json'
    spec = json.loads(spec_path.read_bytes())
    prefix = root / spec['prefix']
    assert not prefix.exists()
    directory = root / 'runs' / args.run
    directory.mkdir(parents=True, exist_ok=False)
    state = dict(state='RUNNING', pid=os.getpid(), started=time.time(),
                 command=[os.sys.executable] + os.sys.argv, steps=[],
                 spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                 prefix=str(prefix), scope=spec['scope'])
    state_path = directory / 'status.json'
    state_path.write_text(json.dumps(state, indent=2) + '\n')
    env = dict(os.environ, **spec['environment'])
    env['MAMBA_ROOT_PREFIX'] = str(root / spec['root_prefix'])
    env['MPLCONFIGDIR'] = str(root / '.cache/matplotlib-cpu')

    def run(command):
        started = time.time()
        code = subprocess.call(command, cwd=str(root), env=env)
        state['steps'].append(dict(command=command, exit_code=code, seconds=time.time()-started))
        if code:
            state.update(state='FAILED', exit_code=code, completed=time.time())
        state_path.write_text(json.dumps(state, indent=2) + '\n')
        if code:
            raise SystemExit(code)

    archive = directory / 'micromamba.tar.bz2'
    run(['curl', '--fail', '--location', '--max-time', '180', spec['bootstrap']['url'], '--output', str(archive)])
    state['bootstrap_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    binary = root / 'conda-tools/bin/micromamba'
    assert not binary.exists()
    with tarfile.open(str(archive), 'r:bz2') as package:
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(package.extractfile('bin/micromamba').read())
    binary.chmod(0o755)
    state['bootstrap_version'] = subprocess.check_output([str(binary), '--version'], env=env).decode().strip()
    run([str(binary), '--no-rc', 'create', '--yes', '--root-prefix', env['MAMBA_ROOT_PREFIX'],
         '--prefix', str(prefix), '--override-channels', '--channel', spec['channel'],
         '--strict-channel-priority'] + spec['packages'])
    explicit = subprocess.check_output([str(binary), '--no-rc', 'env', 'export', '--prefix', str(prefix), '--explicit'], env=env)
    (directory / 'explicit.txt').write_bytes(explicit)
    run([str(prefix / 'bin/python'), str(root / spec['witness']),
         '--project', str(root), '--protocol', args.protocol,
         '--output', str(directory / 'witness.json')])
    state.update(state='COMPLETE', exit_code=0, completed=time.time(),
                 explicit_sha256=hashlib.sha256(explicit).hexdigest())
    state_path.write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
