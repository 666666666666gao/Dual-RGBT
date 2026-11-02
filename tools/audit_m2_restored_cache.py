"""Check restored M2 TRAIN caches against their original acceptance hashes."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4194304), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    relative = "reports/E0_M2_IDENTITY_FULL_COLLECTION_ACCEPTANCE_P25_20261101.json"
    acceptance_path = args.root / relative
    acceptance_sha = sha(acceptance_path)
    assert acceptance_sha == "74eb5104ee223c8fa4eec4992c47f1e466efc6f27a9679a95d372fdd521f2522"
    acceptance = json.loads(acceptance_path.read_bytes())
    assert acceptance["split"] == "train" and acceptance["test_eligible"] is False
    files = {name: digest for name, digest in acceptance["inputs_sha256"].items()
             if name.endswith(".npz")}
    assert len(files) == acceptance["sequences"] == 882
    size = 0
    shards = {}
    for relative, expected in files.items():
        path = args.root / relative
        assert sha(path) == expected, relative
        size += path.stat().st_size
        shard = path.parent.name
        shards[shard] = shards.get(shard, 0) + 1
    assert len(shards) == 8
    print(json.dumps(dict(
        status="RESTORED_M2_TRAIN_CACHE_BYTES_VERIFIED", root=str(args.root),
        acceptance_sha256=acceptance_sha, npz_files=len(files), bytes=size,
        shards=shards, candidate_sets=acceptance["candidate_sets"],
        frames=acceptance["frames"], test_eligible=False,
        scope="Every restored NPZ matches original acceptance. No collection, "
              "model forward, optimizer, GPU allocation or new full acceptance. "
              "Other acceptance inputs and current source admission are not "
              "revalidated by this byte-restoration check.")), flush=True)


if __name__ == "__main__":
    main()
