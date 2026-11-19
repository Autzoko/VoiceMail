"""Overlap study, step 1: word-level timestamps of every reference segment (Qwen3-ForcedAligner-0.6B).

Aligns the raw reference text (corpus markup stripped) to the CLOSE-TALK audio of the segment, so that target
and interfering speakers' words get absolute times. Far-field runs reuse these times (same session clock;
DiPCo far-field times are shifted by the per-segment array offset in overlap_table.py).

Output: data/outputs/analysis/words/<manifest>.jsonl  {"id", "words": [[word, start_abs, end_abs], ...]}
Resumable (already aligned ids are skipped).

    python scripts/overlap/align_words.py --manifests ami__ihm alimeeting__near notsofar__ihm chime6__ihm dipco__ihm
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src" / "models")]

from src.utils.manifest import SR, load_segment, read_manifest  # noqa: E402
from src.utils.text_norm import strip_markup  # noqa: E402

OUT = ROOT / "data" / "outputs" / "analysis" / "words"
MAX_SEC = 290.0  # aligner input limit (~300 s); longer segments are skipped (none in the test sets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--model", default=str(ROOT / "data/checkpoints/Qwen3-ForcedAligner-0.6B"))
    ap.add_argument("--batch-sec", type=float, default=600.0, help="total audio seconds per forward pass")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16", "fp16"])
    args = ap.parse_args()

    import torch
    from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner

    cuda = torch.cuda.is_available()
    if args.dtype == "auto":
        dt = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float32
    else:
        dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    al = Qwen3ForcedAligner.from_pretrained(args.model, dtype=dt, device_map="cuda:0" if cuda else "cpu")
    print(f"aligner on {'cuda' if cuda else 'cpu'} dtype={dt}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for name in args.manifests:
        rows = read_manifest(ROOT / "data/datasets/manifests" / f"{name}.jsonl")
        lang = "Chinese" if rows[0]["lang"] == "zh" else "English"
        out = OUT / f"{name}.jsonl"
        done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
        todo = []
        for r in rows:
            txt = strip_markup(r["text"])
            if r["id"] in done:
                continue
            if not txt or r["end"] - r["start"] > MAX_SEC:
                done.add(r["id"])
                continue
            todo.append((r, txt))
        todo.sort(key=lambda x: x[0]["start"] - x[0]["end"])  # longest first
        print(f"[{name}] {len(rows)} segments, {len(todo)} to align", flush=True)
        t0, i = time.time(), 0
        with open(out, "a") as f, ThreadPoolExecutor(8) as pool:
            while i < len(todo):
                j, sec = i, 0.0
                while j < len(todo) and (j == i or sec + todo[j][0]["end"] - todo[j][0]["start"] <= args.batch_sec):
                    sec += todo[j][0]["end"] - todo[j][0]["start"]
                    j += 1
                chunk = todo[i:j]
                wavs = list(pool.map(lambda x: load_segment(x[0]), chunk))
                try:
                    res = al.align(audio=[(w, SR) for w in wavs], text=[t for _, t in chunk], language=lang)
                except Exception as e:  # one bad item must not kill the batch: retry singly
                    print(f"  batch failed ({e}); retrying singly", flush=True)
                    res = []
                    for w, (_, t) in zip(wavs, chunk):
                        try:
                            res.append(al.align(audio=(w, SR), text=t, language=lang)[0])
                        except Exception as e2:
                            print(f"    skip: {e2}", flush=True)
                            res.append(None)
                for (r, _), o in zip(chunk, res):
                    words = [] if o is None else [[it.text, round(r["start"] + it.start_time, 3),
                                                    round(r["start"] + it.end_time, 3)] for it in o]
                    f.write(json.dumps({"id": r["id"], "words": words}, ensure_ascii=False) + "\n")
                f.flush()
                i = j
                if (i // max(len(chunk), 1)) % 20 == 0 or i == len(todo):
                    print(f"  [{name}] {i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
