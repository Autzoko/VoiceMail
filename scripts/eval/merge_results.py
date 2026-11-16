"""Merge per-job result CSVs (data/results/parts/*.csv) into data/results/qwen3asr_baseline.csv.
Keeps the latest row per (model, dataset, condition); sorted for readability.

    python scripts/eval/merge_results.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["librispeech", "ami", "alimeeting", "notsofar", "chime6", "dipco"]
SPLIT = {"librispeech": "test-clean (sanity; official WER 1.7B=1.63 / 0.6B=2.11)",
         "ami": "test full-corpus-asr (ES2004 IS1009 TS3003 EN2002)", "alimeeting": "M2MeT Test",
         "notsofar": "eval 240825.1_eval_full_with_GT (129 mtgs)", "chime6": "eval S01+S21",
         "dipco": "eval S01+S03+S06+S07+S08"}


def main():
    parts = sorted((ROOT / "data" / "results" / "parts").glob("*.csv"))
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df["lang_mode"] = df.get("lang_mode", pd.Series(index=df.index, dtype=object)).fillna("auto")
    df["split"] = df["dataset"].map(SPLIT).fillna(df["split"])
    df = df.drop_duplicates(["model", "dataset", "condition", "lang_mode"], keep="last")
    df["_d"] = df["dataset"].map({d: i for i, d in enumerate(ORDER)})
    df["_c"] = df["condition"].map({"clean": 0, "ihm": 0, "near": 0, "sdm": 1, "far": 1})
    df = df.sort_values(["model", "lang_mode", "_d", "_c"]).drop(columns=["_d", "_c"])
    out = ROOT / "data" / "results" / "qwen3asr_baseline.csv"
    cols = ["model", "dataset", "condition", "lang_mode"] + [c for c in df.columns if c not in
                                                           ("model", "dataset", "condition", "lang_mode")]
    df = df[cols]
    df.to_csv(out, index=False)
    print(df[["model", "lang_mode", "dataset", "condition", "metric", "error_rate", "sub_rate", "del_rate", "ins_rate",
              "n_utts", "hours"]].to_string(index=False))
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
