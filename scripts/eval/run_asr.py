"""Oracle-segmented (utterance-level) ASR evaluation of Qwen3-ASR on meeting corpora.

For each manifest (data/datasets/manifests/<dataset>__<condition>.jsonl):
  1. cut every reference segment from the recording (given channel), 16 kHz mono
  2. transcribe with Qwen3-ASR (vLLM backend, bf16, greedy, max_new_tokens=1024, no language hint —
     the official Qwen3-ASR evaluation protocol)
  3. normalize ref & hyp identically (src/utils/text_norm.py), score corpus-level WER (en) / CER (zh)
  4. write per-utterance outputs to data/outputs/<model>/<dataset>__<condition>.jsonl and append a
     summary row to data/results/<csv>

Usage:
  python scripts/eval/run_asr.py --model Qwen/Qwen3-ASR-1.7B --manifests ami__ihm alimeeting__far ...
"""
import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src" / "models")]  # vendored qwen_asr takes precedence

from src.utils.manifest import SR, load_segment, read_manifest  # noqa: E402
from src.utils.metrics import ErrorCounts, count_errors  # noqa: E402
from src.utils.text_norm import normalize, tokenize  # noqa: E402

CSV_FIELDS = [
    "model", "dataset", "condition", "split", "metric", "error_rate", "sub_rate", "del_rate", "ins_rate",
    "n_ref_tokens", "n_utts", "n_empty_hyp", "hours", "rtf", "lang_acc", "normalizer", "segmentation",
    "decoding", "git_commit", "date",
]

SPLIT = {"ami": "test (full-corpus-asr)", "alimeeting": "Test", "notsofar": "eval (240825.1 full)",
         "chime6": "eval (S01,S21)", "dipco": "eval (S01,S03,S06,S07,S08)"}


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--manifests", nargs="+", required=True, help="names under data/datasets/manifests")
    ap.add_argument("--csv", default="qwen3asr_baseline.csv")
    ap.add_argument("--batch", type=int, default=256, help="segments per transcribe() call")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N segments only")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from qwen_asr import Qwen3ASRModel

    model_tag = args.model.rstrip("/").split("/")[-1]
    out_dir = ROOT / "data" / "outputs" / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    asr = Qwen3ASRModel.LLM(model=args.model, gpu_memory_utilization=args.gpu_mem,
                            max_inference_batch_size=args.batch, max_new_tokens=args.max_new_tokens)
    commit = git_commit()

    for name in args.manifests:
        dataset, condition = name.split("__")
        out_path = out_dir / f"{name}.jsonl"
        rows = read_manifest(ROOT / "data" / "datasets" / "manifests" / f"{name}.jsonl")
        if args.limit:
            rows = rows[: args.limit]
        lang = rows[0]["lang"]

        done = {}
        if out_path.exists() and not args.overwrite:
            done = {r["id"]: r for r in map(json.loads, open(out_path))}
        todo = [r for r in rows if r["id"] not in done]
        print(f"[{name}] {len(rows)} segments, {len(todo)} to decode", flush=True)

        t0, audio_sec = time.time(), 0.0
        with open(out_path, "a") as fout, ThreadPoolExecutor(16) as pool:
            for i in range(0, len(todo), args.batch):
                chunk = todo[i: i + args.batch]
                wavs = list(pool.map(load_segment, chunk))
                audio_sec += sum(len(w) for w in wavs) / SR
                res = asr.transcribe(audio=[(w, SR) for w in wavs], language=None)
                for r, o in zip(chunk, res):
                    rec = {"id": r["id"], "session": r["session"], "speaker": r["speaker"],
                           "ref": r["text"], "hyp": o.text, "pred_lang": o.language,
                           "dur": round(r["end"] - r["start"], 3)}
                    done[r["id"]] = rec
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"  [{name}] {min(i + args.batch, len(todo))}/{len(todo)} "
                      f"{time.time() - t0:.0f}s", flush=True)
        rtf = (time.time() - t0) / audio_sec if audio_sec else float("nan")

        # ---- score (corpus level) ----
        tot, n_empty, n_lang_ok, n_scored = ErrorCounts(), 0, 0, 0
        want_lang = "Chinese" if lang == "zh" else "English"
        for r in rows:
            rec = done[r["id"]]
            ref_t = tokenize(normalize(rec["ref"], lang), lang)
            if not ref_t:  # reference empty after normalization (pure noise/filler) -> not scored
                continue
            hyp_t = tokenize(normalize(rec["hyp"] or "", lang), lang)
            tot += count_errors(ref_t, hyp_t)
            n_scored += 1
            n_empty += not hyp_t
            n_lang_ok += want_lang.lower() in (rec.get("pred_lang") or "").lower()
        hours = sum(r["end"] - r["start"] for r in rows) / 3600
        row = {"model": model_tag, "dataset": dataset, "condition": condition, "split": SPLIT.get(dataset, ""),
               "metric": "CER" if lang == "zh" else "WER", **tot.as_dict(), "n_empty_hyp": n_empty,
               "hours": round(hours, 2), "rtf": round(rtf, 4) if todo else "",
               "lang_acc": round(100 * n_lang_ok / max(n_scored, 1), 2),
               "normalizer": "zh: t2s+NFKC+no-punct (char)" if lang == "zh" else "chime8 whisper-en",
               "segmentation": "oracle", "decoding": f"vLLM bf16 greedy max_new_tokens={args.max_new_tokens} lang=auto",
               "git_commit": commit, "date": dt.date.today().isoformat()}
        print(json.dumps(row), flush=True)
        csv_path = ROOT / "data" / "results" / args.csv
        new = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
