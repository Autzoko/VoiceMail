"""Promotion gate (design doc §4.4): may a candidate replace the baseline?

    python -m harness.eval.gate BASE/score CAND/score [--out gate.json]    exit 0 = promote, 1 = reject

Both directories come from score.py on the same split. Paired bootstrap over sessions (resampled within each
reference language), 95% intervals of cand - base:
  1. improves   at least one PRIMARY metric is better overall (whole interval on the good side)
  2. no regress every GUARD metric, overall and per language: the interval's bad end is within the tolerance
                max(TOL x |base|, ABS_FLOOR[metric]). The floor exists because 2 % of a rate near zero (foreign
                script, words in silence) is smaller than one word; it is set to roughly one unit of resolution.
  3. official   both runs scored by the locked evaluator, with the same evaluator digest
Gates 4-6 of the design (agent change audit, Swift/Python parity, macmini replay) are checked outside this file.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .metrics import METRICS

TOL = 0.02
N_BOOT = 2000
PRIMARY = ["cp", "orc"]
GUARD = ["orc", "cp", "tcp", "orc_char", "del", "ins", "foreign", "silence_words/min", "spk_abs_err",
         "first_lag", "final_lag", "NE", "big_rewrites/min", "cost_rtf", "chrf"]
ABS_FLOOR = {"del": 0.1, "ins": 0.1, "foreign": 0.1, "silence_words/min": 0.1, "spk_abs_err": 0.05,
             "first_lag": 0.02, "final_lag": 0.02, "NE": 0.005, "big_rewrites/min": 0.05, "cost_rtf": 0.005,
             "chrf": 0.2}


def load(d):
    d = Path(d)
    rows = [json.loads(l) for l in open(d / "sessions.jsonl", encoding="utf-8") if l.strip()]
    return {r["session"]: r for r in rows}, json.loads((d / "summary.json").read_text())


def _ratio(rows, name, weights):
    num, den, scale, _ = METRICS[name]
    n = np.array([r.get(num, 0) for r in rows], float)
    d = np.array([r.get(den, 0) for r in rows], float)
    dd = weights @ d
    with np.errstate(invalid="ignore", divide="ignore"):
        return scale * (weights @ n) / dd, dd


def paired(base, cand, name, sessions, rng):
    """(base value, cand value, low, high) of cand - base with a session bootstrap; None if undefined."""
    b = [base[s] for s in sessions]
    c = [cand[s] for s in sessions]
    one = np.ones((1, len(sessions)))
    bv, bd = _ratio(b, name, one)
    cv, cd = _ratio(c, name, one)
    if not (bd[0] > 0 and cd[0] > 0):
        return None
    w = rng.multinomial(len(sessions), [1 / len(sessions)] * len(sessions), size=N_BOOT).astype(float)
    diff = _ratio(c, name, w)[0] - _ratio(b, name, w)[0]
    diff = diff[np.isfinite(diff)]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return float(bv[0]), float(cv[0]), float(lo), float(hi)


def decide(base, cand, base_summary, cand_summary, seed=0):
    rng = np.random.default_rng(seed)
    if set(base) != set(cand):
        raise SystemExit("baseline and candidate were scored on different sessions")
    groups = {"all": sorted(base)}
    for s in sorted(base):
        groups.setdefault(base[s]["lang"], []).append(s)
    checks, improved = [], []
    for g, sessions in groups.items():
        for m in GUARD:
            r = paired(base, cand, m, sessions, rng)
            if r is None:
                continue
            bv, cv, lo, hi = r
            lower_better = METRICS[m][3]
            tol = max(TOL * abs(bv), ABS_FLOOR.get(m, 0.0))
            bad_end = hi if lower_better else -lo
            ok = bad_end <= tol
            better = (hi < 0) if lower_better else (lo > 0)
            checks.append({"group": g, "metric": m, "base": round(bv, 4), "cand": round(cv, 4),
                           "ci": [round(lo, 4), round(hi, 4)], "tolerance": round(tol, 4), "ok": ok,
                           "better": better})
            if g == "all" and m in PRIMARY and better:
                improved.append(m)
    official = bool(base_summary.get("official") and cand_summary.get("official")
                    and base_summary.get("evaluator") == cand_summary.get("evaluator"))
    regressions = [c for c in checks if not c["ok"]]
    return {"promote": bool(improved) and not regressions and official, "improved": improved,
            "regressions": regressions, "official": official, "checks": checks}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("cand")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    base, bs = load(a.base)
    cand, cs = load(a.cand)
    if bs.get("split") != cs.get("split"):
        raise SystemExit(f"splits differ: {bs.get('split')} vs {cs.get('split')}")
    if bs.get("split") in ("test",):
        print("NOTE: gating on the test split; the test split is for the one final report only", file=sys.stderr)
    res = decide(base, cand, bs, cs)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("promote", "improved", "official")}, indent=1))
    for c in res["regressions"]:
        print(f"  REGRESSION {c['group']}/{c['metric']}: {c['base']} -> {c['cand']}, ci {c['ci']}, "
              f"tolerance {c['tolerance']}")
    sys.exit(0 if res["promote"] else 1)


if __name__ == "__main__":
    main()
