"""Preserve the latest five weights from the already running legacy chain."""
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
NAME = 'size_tail_b576_formal_epochs_20261030'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    controller = ROOT / 'runs' / NAME
    output = controller / 'retained_last_five'
    output.mkdir(exist_ok=False)
    retained = []
    while True:
        state = json.loads((controller / 'status.json').read_text())
        for directory in sorted((ROOT / 'runs').glob(NAME + '_train_e*')):
            receipt_path = directory / 'COMPLETE.json'
            native_path = ROOT / 'runs/pipeline_tasks' / ('train_' + directory.name) / 'status.json'
            if not receipt_path.exists() or not native_path.exists():
                continue
            native = json.loads(native_path.read_text())
            if native['state'] != 'COMPLETE' or native['exit_code'] != 0:
                continue
            receipt = json.loads(receipt_path.read_text())
            epoch = receipt['epoch']
            if any(row['epoch'] == epoch for row in retained):
                continue
            source = directory / ('epoch_%03d.pth' % epoch)
            if not source.exists():
                continue
            assert receipt['strict_reload_bit_equal'] and sha(source) == receipt['checkpoint_sha256']
            target = output / source.name
            assert not target.exists()
            shutil.copyfile(str(source), str(target))
            assert sha(target) == receipt['checkpoint_sha256']
            retained.append(dict(epoch=epoch, checkpoint=target.relative_to(ROOT).as_posix(),
                                 checkpoint_sha256=receipt['checkpoint_sha256'],
                                 training_native_sha256=sha(native_path)))
            if len(retained) > 5:
                old = retained.pop(0)
                owned = ROOT / old['checkpoint']
                assert owned.parent == output and sha(owned) == old['checkpoint_sha256']
                owned.unlink()
        status = dict(controller_status=state['status'], retained=retained,
                      poll_seconds=180, updated=time.time())
        temporary = output / 'status.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(output / 'status.json')
        if state['status'] != 'RUNNING':
            return
        time.sleep(180)


if __name__ == '__main__':
    main()
