"""Inspect exact CADTrack-to-E0 tensor compatibility without writing weights."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cadtrack", required=True)
    parser.add_argument("--e0", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = torch.load(args.cadtrack, map_location="cpu")
    parent = torch.load(args.e0, map_location="cpu")["net"]
    assert source["net_type"] == "CADTrack"
    mapping = {}
    for key in source["net"]:
        assert key.startswith(("backbone.", "box_head.", "decode_fuse_search."))
        target = key.replace("decode_fuse_search.", "decode.", 1)
        mapping[key] = target
    assert len(set(mapping.values())) == len(mapping)
    missing = [target for target in mapping.values() if target not in parent]
    mismatched = [
        {"source": key, "target": target,
         "source_shape": list(source["net"][key].shape),
         "target_shape": list(parent[target].shape)}
        for key, target in mapping.items()
        if target in parent and (
            source["net"][key].shape != parent[target].shape
            or source["net"][key].dtype != parent[target].dtype)
    ]
    preserved = sorted(set(parent) - set(mapping.values()))
    unexpected = [key for key in preserved if not key.startswith(
        ("preview.", "quality.", "gain.")) and key != "token_cost"]
    print(json.dumps({
        "status": "TENSOR_COMPATIBLE" if not (
            missing or mismatched or unexpected) else "INCOMPATIBLE",
        "cadtrack": args.cadtrack, "cadtrack_sha256": digest(args.cadtrack),
        "e0": args.e0, "e0_sha256": digest(args.e0),
        "mapped_tensors": len(mapping), "preserved_tensors": preserved,
        "missing_targets": missing, "shape_or_dtype_mismatches": mismatched,
        "unexpected_preserved_tensors": unexpected,
        "scope": "CPU tensor schema only; no model reload, forward, training, "
                 "weight export or benchmark acceptance.",
    }, indent=2))


if __name__ == "__main__":
    main()
