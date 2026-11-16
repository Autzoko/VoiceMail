"""Merge per-job result CSVs (data/results/parts/*.csv) into data/results/qwen3asr_baseline.csv.
Keeps the latest row per (model, dataset, condition); sorted for readability.

    python scripts/eval/merge_results.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["librispeech", "ami", "alimeeting", "notsofar", "chime6", "dipco"]


def main():
    parts = sorted((ROOT / "data" / "results" / "parts").glob("*.csv"))
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates(["model", "dataset", "condition"], keep="last")
    df["_d"] = df["dataset"].map({d: i for i, d in enumerate(ORDER)})
    df["_c"] = df["condition"].map({"clean": 0, "ihm": 0, "near": 0, "sdm": 1, "far": 1})
    df = df.sort_values(["model", "_d", "_c"]).drop(columns=["_d", "_c"])
    out = ROOT / "data" / "results" / "qwen3asr_baseline.csv"
    df.to_csv(out, index=False)
    print(df[["model", "dataset", "condition", "metric", "error_rate", "sub_rate", "del_rate", "ins_rate",
              "n_utts", "hours"]].to_string(index=False))
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
