"""Export the provided toolkit's GT to a private input, not to Git."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--toolkit', required=True)
    parser.add_argument('--benchmark', choices=('RGBT210', 'RGBT234'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    base = Path(args.toolkit) / 'rgbt/gt_file' / args.benchmark
    name_file = base / ('SequencesName.txt' if args.benchmark == 'RGBT210' else 'attr_txt/SequencesName.txt')
    names = name_file.read_text(encoding='utf-8').split()
    assert len(names) == len(set(names)) == (210 if args.benchmark == 'RGBT210' else 234)
    rows = []
    hashes = {str(name_file.relative_to(Path(args.toolkit))): hashlib.sha256(name_file.read_bytes()).hexdigest()}
    for name in names:
        paths = ([base / 'groundtruth' / name / 'init.txt'] if args.benchmark == 'RGBT210' else
                 [base / 'rgbt234_gt' / name / (modality + '.txt') for modality in ('visible', 'infrared')])
        arrays = [np.loadtxt(path, delimiter=',', dtype=np.float32, ndmin=2) for path in paths]
        assert all(array.shape[1] == 4 and np.isfinite(array).all() for array in arrays)
        assert len({len(array) for array in arrays}) == 1
        row = dict(name=name)
        if args.benchmark == 'RGBT210':
            row['boxes'] = arrays[0].tolist()
        else:
            row.update(visible_boxes=arrays[0].tolist(), infrared_boxes=arrays[1].tolist())
        rows.append(row)
        hashes.update({str(path.relative_to(Path(args.toolkit))): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    output = Path(args.output)
    assert not output.exists()
    result = dict(benchmark=args.benchmark, annotation_source='Provided CADTrack RGBT toolkit, unrounded float32 GT',
                  annotation_file_sha256=hashes, sequences=rows)
    output.write_bytes((json.dumps(result, separators=(',', ':')) + '\n').encode())
    print(json.dumps(dict(benchmark=args.benchmark, sequences=len(rows),
                         output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), nn_calls=0)))


if __name__ == '__main__':
    main()
