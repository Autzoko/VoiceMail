"""Oracle diarization for the M1 credit-assignment ladder: reference speaker activity written in the same
format as the Nemotron cache (T x 8 float32, 10 ms frames, one column per speaker in order of first
appearance), so the twin can replay it with `--diar data/outputs/harness/diar/oracle`.

    python -m harness.diar.oracle --manifest m1.jsonl m2.jsonl --split search
Frame count is taken from the Nemotron output of the same session (same audio); sessions with more than
8 reference speakers fold the extra speakers into the last column (reported).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "data/outputs/harness"
FRAME, S = 0.01, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--split", required=True, choices=["search", "select", "test", "all"])
    ap.add_argument("--splits", default=str(ROOT / "harness/splits.json"))
    ap.add_argument("--nemotron", default=str(H / "diar/nemotron3_low"))
    ap.add_argument("--out", default=str(H / "diar/oracle"))
    a = ap.parse_args()
    from harness.eval.score import split_of

    splits = json.loads(Path(a.splits).read_text())["splits"]
    segs = defaultdict(list)
    for m in a.manifest:
        for line in open(m, encoding="utf-8"):
            r = json.loads(line)
            segs[r["session"]].append(r)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n = folded = 0
    for session, rows in sorted(segs.items()):
        if a.split != "all" and split_of(session, splits) != a.split:
            continue
        ref = Path(a.nemotron) / f"{session}.npy"
        if not ref.exists():
            continue
        T = np.load(ref, mmap_mode="r").shape[-2]
        x = np.zeros((T, S), np.float32)
        col = {}
        for r in sorted(rows, key=lambda r: r["start"]):
            c = col.setdefault(r["speaker"], min(len(col), S - 1))
            x[int(r["start"] / FRAME):int(np.ceil(r["end"] / FRAME)), c] = 1.0
        folded += len(col) > S
        np.save(out / f"{session}.npy", x)
        n += 1
    print(f"oracle diarization: {n} sessions, {folded} with more than {S} speakers folded")


if __name__ == "__main__":
    main()
