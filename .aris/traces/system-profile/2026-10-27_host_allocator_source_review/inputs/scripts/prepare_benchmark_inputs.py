"""Prepare complete private image/GT inputs; keep original annotations untouched."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from benchmark_inputs import load_manifest, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', choices=('LasHeR', 'RGBT210', 'RGBT234'), required=True)
    parser.add_argument('--split', choices=('train_smoke', 'dev', 'test'), required=True)
    parser.add_argument('--dataset-root')
    parser.add_argument('--annotations')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows, source = [], {}
    if args.benchmark == 'LasHeR':
        assert args.dataset_root is None and args.annotations is None
        path = ROOT / 'data/manifest.json'
        original = json.loads(path.read_bytes())
        source = dict(annotation_source='Original LasHeR init.txt; existing 882/97/245 split',
                      original_manifest_sha256=sha(path))
        if args.split in ('train_smoke', 'dev'):
            selected = original['train'][:1] if args.split == 'train_smoke' else original['dev']
            rows = [dict(name=row['name'], rgb=row['rgb'], tir=row['tir'], boxes=row['boxes']) for row in selected]
        else:
            for name in original['test_sequences']:
                directory = ROOT / 'data/LasHeR/test' / name
                boxes = np.loadtxt(str(directory / 'init.txt'), delimiter=',', dtype=np.float32, ndmin=2)
                rows.append(dict(name=name, rgb=list(map(str, sorted((directory / 'visible').glob('*.jpg')))),
                                 tir=list(map(str, sorted((directory / 'infrared').glob('*.jpg')))), boxes=boxes.tolist()))
        assert len(rows) == {'train_smoke': 1, 'dev': 97, 'test': 245}[args.split]
    else:
        assert args.split == 'test' and args.dataset_root and args.annotations
        annotations = json.loads(Path(args.annotations).read_bytes())
        assert annotations['benchmark'] == args.benchmark
        source = dict(annotation_source=annotations['annotation_source'],
                      annotation_input_sha256=sha(args.annotations), annotation_file_sha256=annotations['annotation_file_sha256'],
                      image_root=str(Path(args.dataset_root)))
        assert len(annotations['sequences']) == (210 if args.benchmark == 'RGBT210' else 234)
        assert {row['name'] for row in annotations['sequences']} == {p.name for p in Path(args.dataset_root).iterdir() if p.is_dir()}
        for annotation in annotations['sequences']:
            directory = Path(args.dataset_root) / annotation['name']
            rows.append(dict(annotation,
                rgb=list(map(str, sorted((directory / 'visible').glob('*.jpg')))),
                tir=list(map(str, sorted((directory / 'infrared').glob('*.jpg'))))))
    endpoints = []
    for row in rows:
        boxes = row['visible_boxes'] if args.benchmark == 'RGBT234' else row['boxes']
        row['initial_box'] = boxes[0]
        assert len(row['rgb']) == len(row['tir']) == len(boxes), row['name']
        if args.benchmark == 'RGBT234':
            assert len(row['infrared_boxes']) == len(boxes), row['name']
        sizes = []
        for frame in (0, len(row['rgb']) - 1):
            with Image.open(row['rgb'][frame]) as rgb, Image.open(row['tir'][frame]) as tir:
                assert rgb.size == tir.size, row['name']
                sizes.append(list(rgb.size))
        endpoints.append(dict(sequence=row['name'], endpoint_sizes=sizes))
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists()
    result = dict(benchmark=args.benchmark, split=args.split, sequences=rows,
                  sequence_count=len(rows), frame_count=sum(len(row['rgb']) for row in rows),
                  initialization='Frame0 visible/common raw float32 GT only; subsequent frames use predictions.', **source)
    output.write_bytes((json.dumps(result, separators=(',', ':')) + '\n').encode())
    checked = load_manifest(output)
    report = dict(benchmark=args.benchmark, split=args.split, sequence_count=checked['sequence_count'],
                  frame_count=checked['frame_count'], manifest_sha256=sha(output), nn_calls=0,
                  annotation_source=checked['annotation_source'], endpoint_pairs_checked=2 * len(rows),
                  scope='All filenames/counts and unrounded GT serialized; first/last pairs decoded. Interior image pixels not exhaustively decoded. No tracking metrics.')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
