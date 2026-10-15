"""Download the fixed, hash-checked historical CLIP assets (no model execution)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT))
from t2methods.config import CLIP_SOURCE_SHA, CLIP_BPE_SHA, CLIP_WEIGHT_SHA

COOP_COMMIT = "ff61507c790454bce7c5052c3ac39e60772f1f89"
SOURCE_BASE = "https://raw.githubusercontent.com/KaiyangZhou/CoOp/" + COOP_COMMIT + "/"
WEIGHT_URL = "https://openaipublic.azureedge.net/clip/models/" + CLIP_WEIGHT_SHA + "/ViT-B-16.pt"


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def obtain(url, destination, expected, check_only=False):
    if destination.exists():
        if not destination.is_file() or file_sha(destination) != expected:
            raise ValueError("Existing asset SHA differs; inspect rather than overwrite: " + str(destination))
        return "verified"
    if check_only:
        raise FileNotFoundError("Required asset is absent: " + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise FileExistsError("Incomplete prior download; inspect/remove it before retry: " + str(temporary))
    request = urllib.request.Request(url, headers={"User-Agent": "SPCP-reproducibility-setup/0.1"})
    print("Downloading " + destination.name, flush=True)
    with urllib.request.urlopen(request, timeout=60) as source, temporary.open("xb") as output:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            output.write(block)
    if file_sha(temporary) != expected:
        raise ValueError("Downloaded SHA differs; incomplete file retained for inspection: " + str(temporary))
    temporary.rename(destination)
    return "downloaded_and_verified"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", default=str(ROOT / "external"))
    parser.add_argument("--weights", action="store_true", help="Also download the official approximately 335 MB ViT-B/16 checkpoint")
    parser.add_argument("--check-only", action="store_true", help="Verify existing files without network access")
    args = parser.parse_args(argv)
    destination = Path(args.destination).absolute()
    records = {}
    for name, digest in {**CLIP_SOURCE_SHA, "clip/bpe_simple_vocab_16e6.txt.gz": CLIP_BPE_SHA}.items():
        records[name] = obtain(SOURCE_BASE + name, destination / "original_clip" / name, digest, args.check_only)
    if args.weights:
        records["ViT-B-16.pt"] = obtain(WEIGHT_URL, destination / "ViT-B-16.pt", CLIP_WEIGHT_SHA, args.check_only)
    print(json.dumps({"valid": True, "source_commit": COOP_COMMIT, "assets": records}, indent=2))


if __name__ == "__main__":
    main()
