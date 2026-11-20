"""Score one system run (a directory of hypothesis JSON files) on one split.

    python -m harness.eval.score --hyp RUN_DIR --ref data/datasets/manifests/notsofar__sdm.jsonl [...] \
        --split select --out RUN_DIR/score

Writes `sessions.jsonl` (per-session counts; the gate reads these) and `summary.json` (all metrics, overall and
per reference language). Sessions of the split that have no hypothesis file are scored as empty output: a run
cannot improve by skipping hard sessions. Without a matching LOCK the result is marked unofficial.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from . import lock
from .metrics import METRICS, session_counts, value
from .schema import load_hypotheses, load_references

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "harness/splits.json"


def split_of(session, splits):
    for ds, units in splits.items():
        if session in units:
            return units[session]
    meeting = session.split("_sc_")[0]
    return splits.get("notsofar", {}).get(meeting)


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups["all"].append(r)
        groups[r["lang"]].append(r)
    return {g: {"sessions": len(rs), **{m: (round(v, 3) if (v := value(rs, m)) is not None else None)
                                        for m in METRICS}} for g, rs in groups.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True, help="directory of per-session hypothesis JSON files")
    ap.add_argument("--ref", required=True, nargs="+", help="reference manifest JSONL file(s)")
    ap.add_argument("--split", required=True, choices=["search", "select", "test", "anchor"])
    ap.add_argument("--splits", default=str(SPLITS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    official = lock.check(quiet=False)
    refs = load_references(a.ref)
    if a.split != "anchor":  # the anchor set (macmini only) is its own split: all of its sessions count
        splits = json.loads(Path(a.splits).read_text())["splits"]
        refs = {s: r for s, r in refs.items() if split_of(s, splits) == a.split}
    if not refs:
        raise SystemExit(f"no reference session of split {a.split!r} in {a.ref}")
    hyps = load_hypotheses(a.hyp)
    extra = sorted(set(hyps) - set(refs))
    rows = [session_counts(s, refs[s], hyps.get(s)) for s in sorted(refs)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sessions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"official": official, "evaluator": lock.digest(), "chime_utils": lock.chime_utils_commit(),
               "split": a.split, "hyp": str(a.hyp),
               "missing_hypotheses": sorted(set(refs) - set(hyps)), "ignored_hypotheses": len(extra),
               "metrics": summarize(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary["metrics"], indent=1))


if __name__ == "__main__":
    main()
