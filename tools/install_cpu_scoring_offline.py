"""Install the exact accepted CPU scoring packages from a staged offline bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    root = Path(args.project).resolve()
    bundle = root / args.bundle
    manifest = json.loads((bundle / 'manifest.json').read_bytes())
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == digest, name
    spec_path = root / '.aris/compute/cpu-scoring-env.json'
    assert hashlib.sha256(spec_path.read_bytes()).hexdigest() == manifest['spec_sha256']
    spec = json.loads(spec_path.read_bytes())
    prefix = root / spec['prefix']
    assert not prefix.exists()
    run = root / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, **spec['environment'])
    env['MAMBA_ROOT_PREFIX'] = str(root / spec['root_prefix'])
    env['MPLCONFIGDIR'] = str(root / '.cache/matplotlib-cpu')
    binary = bundle / 'micromamba'
    explicit = run / 'offline-explicit.txt'
    lines = ['@EXPLICIT']
    for record in manifest['packages']:
        package = bundle / 'packages' / record['filename']
        assert hashlib.md5(package.read_bytes()).hexdigest() == record['md5']
        lines.append(package.as_uri() + '#' + record['md5'])
    explicit.write_text('\n'.join(lines) + '\n')
    state = dict(state='RUNNING', pid=os.getpid(), started=time.time(), steps=[],
                 prefix=str(prefix), bundle_manifest=manifest,
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    status = run / 'status.json'
    status.write_text(json.dumps(state, indent=2) + '\n')
    commands = [
        [str(binary), '--no-rc', 'create', '--yes', '--offline', '--root-prefix',
         env['MAMBA_ROOT_PREFIX'], '--prefix', str(prefix), '--file', str(explicit)],
        [str(prefix / 'bin/python'), str(root / spec['witness']), '--project', str(root),
         '--protocol', args.protocol, '--output', str(run / 'witness.json')],
    ]
    for command in commands:
        start = time.time()
        code = subprocess.call(command, cwd=str(root), env=env)
        state['steps'].append(dict(command=command, exit_code=code, seconds=time.time()-start))
        if code:
            state.update(state='FAILED', exit_code=code, completed=time.time())
        status.write_text(json.dumps(state, indent=2) + '\n')
        if code:
            raise SystemExit(code)
    state.update(state='COMPLETE', exit_code=0, completed=time.time())
    status.write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps(dict(state=state['state'], seconds=state['completed']-state['started'])))


if __name__ == '__main__':
    main()
