"""Export paired near/far listening samples with ground truth (and model outputs if available).

For each dataset, pick N utterances that exist in BOTH conditions (same session, same time span, same text),
cut them from the near-field and far-field recordings, and write
    data/datasets/samples/<dataset>/{near,far}/<k>_<session>_<speaker>.wav   (16 kHz mono PCM16)
    data/datasets/samples/<dataset>/index.tsv   (GT raw / normalized, hyps of --model, file names)
    data/datasets/samples/README.md             (all datasets in one readable page)

    python scripts/data/export_samples.py [--n 5] [--model Qwen3-ASR-1.7B]
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.manifest import SR, load_segment, read_manifest  # noqa: E402
from src.utils.text_norm import normalize  # noqa: E402

PAIRS = {  # dataset: (near condition, far condition, what the far mic is)
    "ami": ("ihm", "sdm", "Array1-01 (table array, mic 1)"),
    "alimeeting": ("near", "far", "8-ch table array, channel 0"),
    "notsofar": ("ihm", "sdm", "single-channel device sc_*/ch0 (first device of the meeting)"),
    "chime6": ("ihm", "sdm", "reference Kinect array of the utterance, CH1"),
    "dipco": ("ihm", "sdm", "far-field array U01, CH1"),
}


def key(r):
    return (r["session"].split("_sc_")[0], r["text"])


def load_hyps(model, name):
    p = ROOT / "data" / "outputs" / model / f"{name}.jsonl"
    return {r["id"]: r for r in map(json.loads, open(p))} if p.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--model", default="Qwen3-ASR-1.7B")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_root = ROOT / "data" / "datasets" / "samples"
    md = ["# Near/far listening samples", "",
          "Same utterance, same time span, different microphone (all real recordings, nothing synthesized).",
          f"Hypotheses: {args.model}, official protocol (no language hint). GT_norm = text actually scored.", ""]

    for ds, (cn, cf, far_desc) in PAIRS.items():
        near = read_manifest(ROOT / "data/datasets/manifests" / f"{ds}__{cn}.jsonl")
        far = read_manifest(ROOT / "data/datasets/manifests" / f"{ds}__{cf}.jsonl")
        if ds == "notsofar":  # far has one row per SC device: keep the first device of each meeting
            first = {}
            for r in far:
                first.setdefault(r["session"].split("_sc_")[0], r["session"])
            far = [r for r in far if r["session"] == first[r["session"].split("_sc_")[0]]]
        lang = near[0]["lang"]
        far_by = {}
        for r in far:
            far_by.setdefault(key(r), r)
        cand = []
        for r in near:
            f = far_by.get(key(r))
            nt = normalize(r["text"], lang)
            long_enough = len(nt) >= 10 if lang == "zh" else len(nt.split()) >= 6
            if f and 3.0 <= r["end"] - r["start"] <= 10.0 and long_enough:
                cand.append((r, f))
        rng = random.Random(args.seed)
        rng.shuffle(cand)
        picked, seen = [], set()  # spread over sessions
        for r, f in cand:
            if r["session"] not in seen or len(seen) >= len({c[0]["session"] for c in cand}):
                picked.append((r, f))
                seen.add(r["session"])
            if len(picked) == args.n:
                break

        hn, hf = load_hyps(args.model, f"{ds}__{cn}"), load_hyps(args.model, f"{ds}__{cf}")
        d = out_root / ds
        (d / "near").mkdir(parents=True, exist_ok=True)
        (d / "far").mkdir(parents=True, exist_ok=True)
        rows = []
        md += [f"## {ds}", f"near = `{cn}` (speaker's own headset/close-talk mic), far = `{cf}`: {far_desc}", ""]
        for k, (r, f) in enumerate(picked, 1):
            fn = f"{k}_{r['session']}_{r['speaker']}.wav".replace("/", "-")
            sf.write(d / "near" / fn, load_segment(r), SR, subtype="PCM_16")
            sf.write(d / "far" / fn, load_segment(f), SR, subtype="PCM_16")
            row = {"k": k, "file": fn, "session": r["session"], "speaker": r["speaker"],
                   "start": r["start"], "end": r["end"], "dur": round(r["end"] - r["start"], 2),
                   "near_src": f"{Path(r['audio']).name} ch{r['channel']}",
                   "far_src": f"{Path(f['audio']).parent.name}/{Path(f['audio']).name} ch{f['channel']}",
                   "far_start": f["start"], "far_end": f["end"],
                   "GT_raw": r["text"], "GT_norm": normalize(r["text"], lang),
                   "hyp_near": hn.get(r["id"], {}).get("hyp", ""), "hyp_far": hf.get(f["id"], {}).get("hyp", "")}
            rows.append(row)
            md += [f"**{k}. `{fn}`** — {row['session']} / {row['speaker']}, {row['start']:.2f}–{row['end']:.2f} s "
                   f"({row['dur']} s); near: `{row['near_src']}`, far: `{row['far_src']}` "
                   f"({row['far_start']:.2f}–{row['far_end']:.2f} s)", "",
                   f"- GT (raw): {row['GT_raw']}", f"- GT (normalized, scored): {row['GT_norm']}",
                   f"- hyp near: {row['hyp_near']}", f"- hyp far: {row['hyp_far']}", ""]
        with open(d / "index.tsv", "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"{ds}: {len(rows)} pairs -> {d.relative_to(ROOT)}")
    (out_root / "README.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
