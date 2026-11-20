"""Twin vs real app on the same sessions (design doc §3.3: route agreement >= 95 %, WER difference <= 0.5 points).

    python -m harness.twin.calibrate --twin RUN_T --vocali RUN_V        (both scored with harness.eval.score)

Utterances are matched by time overlap (IoU >= 0.5). Reports: share of app utterances the twin reproduces, engine
agreement on matched utterances, and the difference of every evaluator metric (twin - app), overall and per
language. Calibration fails (exit 1) when a threshold is missed; the cost model and the twin are then fixed before
any twin result is trusted.
"""
import argparse
import json
from pathlib import Path

ROUTE_MIN = 0.95
WER_MAX = 0.5
LAG_MAX = 0.15


def _utts(h):
    eng = {}
    for ln in h["lines"]:
        eng.setdefault(str(ln.get("utterance")), ln.get("engine"))
    return [(a, b, eng.get(u)) for u, (a, b) in h["utterances"].items()]


def match(tw, vo):
    matched = agree = 0
    for a, b, e in vo:
        best, best_iou = None, 0.0
        for c, d, f in tw:
            inter = max(0.0, min(b, d) - max(a, c))
            iou = inter / max(max(b, d) - min(a, c), 1e-9)
            if iou > best_iou:
                best, best_iou = f, iou
        if best_iou >= 0.5:
            matched += 1
            agree += best == e
    return matched, agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--twin", required=True)
    ap.add_argument("--vocali", required=True)
    a = ap.parse_args()
    T, V = Path(a.twin), Path(a.vocali)
    n_v = m = ag = 0
    for p in sorted(V.glob("[!_]*.json")):
        q = T / p.name
        if not q.exists():
            continue
        vo, tw = _utts(json.loads(p.read_text())), _utts(json.loads(q.read_text()))
        mm, aa = match(tw, vo)
        n_v, m, ag = n_v + len(vo), m + mm, ag + aa
    st = json.loads((T / "score/summary.json").read_text())["metrics"]
    sv = json.loads((V / "score/summary.json").read_text())["metrics"]
    diffs = {g: {k: (round(st[g][k] - sv[g][k], 3) if st[g].get(k) is not None and sv[g].get(k) is not None
                     else None) for k in sv[g] if k != "sessions"} for g in sv if g in st}
    res = {"app_utterances": n_v, "matched": m / max(n_v, 1), "route_agreement": ag / max(m, 1), "diff": diffs}
    fails = []
    if res["route_agreement"] < ROUTE_MIN:
        fails.append(f"route agreement {res['route_agreement']:.3f} < {ROUTE_MIN}")
    for g, d in diffs.items():
        for k in ("orc", "cp"):
            if d.get(k) is not None and abs(d[k]) > WER_MAX:
                fails.append(f"{g}/{k} differs by {d[k]}")
        for k in ("first_lag", "final_lag"):
            if d.get(k) is not None and abs(d[k]) > LAG_MAX:
                fails.append(f"{g}/{k} differs by {d[k]} s")
    res["fails"] = fails
    print(json.dumps(res, indent=1))
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
