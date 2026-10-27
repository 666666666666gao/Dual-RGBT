"""CPU environment witness: real annotation identity and synthetic metric fixtures only."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    root = Path(args.project).resolve()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    import numpy as np
    import matplotlib
    assert np.__version__ == '1.26.4' and matplotlib.__version__ == '3.8.4'
    assert sys.version_info[:2] == (3, 10)
    toolkit = root / 'vendor/cadtrack/toolkit'
    sys.path.insert(0, str(toolkit))
    sys.path.insert(0, str(root / 'scripts'))
    from benchmark_inputs import groundtruth
    from rgbt import LasHeR, RGBT210, RGBT234
    from rgbt.metrics.metrics import PR, SR, PR_LasHeR, SR_LasHeR, NPR, MPR, MSR
    spec = json.loads((root / args.protocol).read_bytes())
    for name, digest in spec['toolkit_code_sha256'].items():
        assert hashlib.sha256((toolkit / name).read_bytes()).hexdigest() == digest
    rng = np.random.RandomState(2027)
    x = rng.randn(16, 16)
    product = x @ x.T
    assert np.isfinite(product).all()
    buffer = io.BytesIO()
    np.save(buffer, product)
    buffer.seek(0)
    assert np.array_equal(np.load(buffer), product)
    counts = {}
    for benchmark, cls in [('LasHeR', LasHeR), ('RGBT210', RGBT210), ('RGBT234', RGBT234)]:
        path = root / spec['datasets'][benchmark]['test']
        manifest = json.loads(path.read_bytes())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec['input_sha256'][path.relative_to(root).as_posix()]
        dataset = cls()
        assert set(dataset.seqs_name) == {s['name'] for s in manifest['sequences']}
        for sequence in manifest['sequences']:
            visible, infrared = groundtruth(sequence, benchmark)
            if benchmark == 'LasHeR':
                assert np.array_equal(visible.round(0), np.asarray(dataset[sequence['name']]))
            else:
                for modality, gt in [('visible', visible), ('infrared', infrared)]:
                    assert np.array_equal(gt, np.asarray(dataset[sequence['name']][modality], dtype=np.float32))
        for name, digest in manifest.get('annotation_file_sha256', {}).items():
            assert hashlib.sha256((toolkit / name.replace('\\', '/')).read_bytes()).hexdigest() == digest
        counts[benchmark] = dict(sequences=len(manifest['sequences']), frames=manifest['frame_count'])
    checks = []
    truth = np.tile([10., 20., 10., 20.], (4, 1))
    far = np.tile([1000., 2000., 10., 20.], (4, 1))
    for metric, expected in [(PR, .25), (PR_LasHeR, .25), (NPR, .25),
                             (SR, 5/21), (SR_LasHeR, 5/21)]:
        value, curve = metric()({'fixture': truth.copy()}, {'fixture': far.copy()}, ['fixture'])
        assert np.isclose(value, expected), (metric.__name__, value, expected)
        assert curve.shape == (1, 21 if metric in (SR, SR_LasHeR) else 51)
        checks.append(metric.__name__)
    for metric, expected in [(MPR, 1.), (MSR, 20/21)]:
        value, curve = metric()({'fixture': {'visible': truth.copy(), 'infrared': truth.copy()}},
                                {'fixture': truth.copy()}, ['fixture'])
        assert np.isclose(value, expected), (metric.__name__, value, expected)
        checks.append(metric.__name__)
    result = dict(status='CPU_SCORING_WITNESS_PASS', python=sys.version, numpy=np.__version__,
                  matplotlib=matplotlib.__version__, prefix=sys.prefix,
                  real_annotation_identity=counts, synthetic_fixture_checks=checks,
                  cpu_kernel_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  nn_runs=0, checkpoints_read=0, model_predictions_read=0,
                  scope=__doc__ + ' Synthetic values are not tracking accuracy.')
    output = Path(args.output)
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
