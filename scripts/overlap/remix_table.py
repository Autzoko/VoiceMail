"""Overlap study, controlled experiment analysis: same per-utterance attribution as overlap_table.py, on the
remixed mixtures (exact interferer words and gains known from data/datasets/remix/<ds>.meta.jsonl).

Output: data/outputs/analysis/overlap/<model>/remix__<ds>.parquet (design columns + attribution counts; the
        `clean` row of each target and the paired deltas are used by overlap_summary.py --tag remix)

    python scripts/overlap/remix_table.py [--models Qwen3-ASR-1.7B Qwen3-ASR-0.6B]
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "overlap")]
import overlap_table as ot  # noqa: E402
from src.utils.text_norm import normalize, tokenize  # noqa: E402

AN = ROOT / "data" / "outputs" / "analysis"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen3-ASR-1.7B", "Qwen3-ASR-0.6B"])
    ap.add_argument("--datasets", nargs="+", default=list(ot.PAIRS))
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    for ds in args.datasets:
        mf = ROOT / "data/datasets/remix" / f"{ds}.meta.jsonl"
        if not mf.exists():
            continue
        lang = "zh" if ds == "alimeeting" else "en"
        meta = [json.loads(l) for l in open(mf)]
        clock, idx, rows = {}, {}, {}
        for m in meta:
            ref_t = [w[0] for w in m["target_words"]]
            S = np.array([w[1] for w in m["target_words"]])
            E = np.array([w[2] for w in m["target_words"]])
            clock[m["id"]] = (ref_t, S, E, 1.0)
            toks = sorted((s, e, t, p) for t, s, e, p in m["interferers"]) + [(s, e, t, "T") for t, s, e in m["target_words"]]
            toks.sort()
            st = np.array([x[0] for x in toks])
            en = np.array([x[1] for x in toks])
            idx[m["id"]] = (st, en, [x[2] for x in toks], [x[3] for x in toks], float((en - st).max()))
            rows[m["id"]] = {"id": m["id"], "session": m["id"], "speaker": "T"}
        ot.G.update(lang=lang, clock=clock, idx=idx, own={}, leak={})
        for model in args.models:
            hf = ROOT / "data/outputs" / model / f"remix__{ds}__forced.jsonl"
            if not hf.exists():
                print(f"  missing {hf}", flush=True)
                continue
            recs = list({o["id"]: o for o in map(json.loads, open(hf))}.values())
            # `own` (same speaker's adjacent segments) is irrelevant inside a remix; target text check:
            jobs = [(o, rows[o["id"]], {"own": ""}) for o in recs if o["id"] in clock]
            with Pool(args.workers) as pool:
                res = pool.map(ot.analyze, jobs, chunksize=128)
            df = pd.DataFrame(res)
            design = pd.DataFrame([{k: m[k] for k in ("id", "target_id", "cluster_id", "variant", "type", "sir_set",
                                                      "ratio_set", "position_set", "n_int", "noise_twin")} |
                                   {"sir_act": m.get("sir_act", np.nan), "dur": m["span"][1] - m["span"][0]}
                                   for m in meta])
            df = df.drop(columns=["sir_db"]).merge(design, on="id", how="left")
            df["sir_db"] = df.sir_act
            df["session"], df["speaker"] = df.target_id, "T"
            for k, v in (("model", model), ("dataset", f"remix_{ds}"), ("condition", "far"), ("lang_mode", "forced")):
                df.insert(0, k, v)
            dst = AN / "overlap" / model / f"remix__{ds}.parquet"
            dst.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst, index=False)
            Et = df[["S", "D", "I"]].sum().sum()
            print(f"[remix] {model} {ds}: {len(df)} mixes err={100 * Et / df.n_ref.sum():.2f} " +
                  " ".join(f"{c}={100 * df[c].sum() / df.n_ref.sum():.2f}" for c in ot.CATS), flush=True)


if __name__ == "__main__":
    main()
