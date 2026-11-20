"""Fixed search / select / test splits for the harness RSI loop (docs/harness-rsi.md §6), by recording session.

Rules
- NOTSOFAR eval (129 meetings): 45 search / 40 select / rest test, by meeting (all devices + close-talk of a
  meeting go together).
- AliMeeting Test: half search, half select, by session.
- LibriCSS: OV10-OV30, 0S, 0L search; OV40 select.
- AMI, CHiME-6, DiPCo: test only (never searched).
- Remix: inherits the split of the meeting its target/interferer came from; remixes of test meetings are
  excluded, and remix never enters select or test (synthetic).
- Arabic sets: added when they are moved to JUBAIL (dataset-level hold-out: Casablanca-UAE and Mixat = test).
Assignment is a seeded hash of the unit name, so it is stable and independent of file order.

    python harness/make_splits.py   ->  harness/splits.json (committed; the evaluator reads only this file)
"""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "data/datasets/manifests"
SEED = "vocali-harness-v1"


def order(units):
    return sorted(units, key=lambda u: hashlib.sha1(f"{SEED}:{u}".encode()).hexdigest())


def sessions(name):
    with open(MAN / f"{name}.jsonl") as f:
        return {json.loads(line)["session"] for line in f}


def meeting(ds, session):
    return session.split("_sc_")[0] if ds == "notsofar" else session


def main():
    splits = defaultdict(dict)  # dataset -> unit -> split
    nsf = order({meeting("notsofar", s) for s in sessions("notsofar__sdm") | sessions("notsofar__ihm")})
    for i, m in enumerate(nsf):
        splits["notsofar"][m] = "search" if i < 45 else "select" if i < 85 else "test"
    ali = order(sessions("alimeeting__far") | sessions("alimeeting__near"))
    for i, s in enumerate(ali):
        splits["alimeeting"][s] = "search" if i < len(ali) // 2 else "select"
    for s in sorted(sessions("libricss__sdm")):
        splits["libricss"][s] = "select" if s.startswith("OV40") else "search"
    for ds in ("ami", "chime6", "dipco"):
        for s in sorted(sessions(f"{ds}__sdm") | sessions(f"{ds}__ihm")):
            splits[ds][s] = "test"
    for ds in ("notsofar", "alimeeting", "ami", "chime6", "dipco"):
        meta = ROOT / f"data/datasets/remix/{ds}.meta.jsonl"
        if not meta.exists():
            continue
        for line in open(meta):
            m = json.loads(line)
            src = m["cluster_id"].split("/")[1]       # "<ds>/<meeting>/<speaker>"
            parent = splits.get(ds, {}).get(meeting(ds, src), "test")
            splits[f"remix_{ds}"][m["id"]] = "search" if parent == "search" else "excluded"
    out = {"seed": SEED, "unit": "session (NOTSOFAR: meeting; remix: mixture id)", "splits": splits,
           "counts": {ds: dict(Counter(v.values())) for ds, v in splits.items()}}
    (ROOT / "harness/splits.json").write_text(json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps(out["counts"], indent=1))


if __name__ == "__main__":
    main()
