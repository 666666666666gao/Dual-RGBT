"""Sample first/middle/last registered frames for cross-server image comparison."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifests', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    samples = {}
    for group in ('lasher_train_smoke', 'lasher_dev', 'lasher_test', 'rgbt210_test', 'rgbt234_test'):
        manifest = json.loads((ROOT / args.manifests / (group + '.json')).read_bytes())
        for row in manifest['sequences']:
            for modality in ('rgb', 'tir'):
                paths = row[modality]
                for index in sorted({0, len(paths) // 2, len(paths) - 1}):
                    path = Path(paths[index])
                    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                    assert image is not None, str(path)
                    key = '%s/%s/%s/%d' % (group, row['name'], modality, index)
                    samples[key] = dict(file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                        decoded_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                                        shape=list(image.shape))
    report = dict(samples=samples, sampled_images=len(samples),
                  scope='First/middle/last frame per sequence/modality; image equality is sampled, not exhaustive.')
    output = ROOT / args.output
    assert not output.exists()
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(output=args.output, sampled_images=len(samples))))


if __name__ == '__main__':
    main()
