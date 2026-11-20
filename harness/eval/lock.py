"""Evaluator lock: SHA-256 of every evaluator file, recorded in harness/eval/LOCK.

    python -m harness.eval.lock --check     exit 1 if the evaluator differs from LOCK
    python -m harness.eval.lock --write     (humans only) record the current files after a reviewed change
The normalizer the evaluator imports from src/utils (text_norm.py) is locked too. Only git-tracked files are
hashed, so the digest is the same on every machine; the git-ignored chime-utils clone (English normalizer) is
pinned instead by its commit, which `score.py` records in summary.json.
"""
import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LOCK = HERE / "LOCK"
EXTRA = [ROOT / "src/utils/text_norm.py"]


def files():
    return sorted(HERE.glob("*.py")) + EXTRA


def chime_utils_commit():
    import subprocess

    try:
        return subprocess.run(["git", "-C", str(ROOT / "lib/chime-utils"), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files()}


def digest():
    h = hashlib.sha256()
    for k, v in sorted(hashes().items()):
        h.update(f"{k} {v}\n".encode())
    return h.hexdigest()[:16]


def recorded():
    if not LOCK.exists():
        return {}
    return dict(line.split()[::-1] for line in LOCK.read_text().splitlines() if line.strip())


def check(quiet=True):
    now, rec = hashes(), recorded()
    ok = now == rec
    if not ok and not quiet:
        changed = sorted(k for k in set(now) | set(rec) if now.get(k) != rec.get(k))
        print(f"WARNING evaluator differs from LOCK ({', '.join(changed)}): results are UNOFFICIAL",
              file=sys.stderr)
    return ok


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.write:
        LOCK.write_text("".join(f"{v}  {k}\n" for k, v in sorted(hashes().items())))
        print(f"LOCK written, digest {digest()}")
    else:
        sys.exit(0 if check(quiet=False) else 1)


if __name__ == "__main__":
    main()
