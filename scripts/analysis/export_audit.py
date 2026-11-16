"""Phase 1 error analysis, step 3: stratified listening package to check the automatic error-type labels.

For every dataset x near/far x error type, sample N utterances whose DOMINANT error type is that type and write
    data/datasets/samples/audit/<model>_<mode>/<dataset>/<cond>/<TYPE>_<k>_<id>.wav
    data/datasets/samples/audit/<model>_<mode>/audit.tsv   (fill in `human_label`, `note`)
    data/datasets/samples/audit/<model>_<mode>/README.md   (ref / hyp / alignment / other speakers' text)
Alignment notation: [ref->hyp] substitution, [-ref] deletion, [+hyp] insertion.

    python scripts/analysis/export_audit.py [--model Qwen3-ASR-1.7B] [--mode auto] [--n 5]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.error_taxonomy import alignment  # noqa: E402
from src.utils.manifest import SR, load_segment, read_manifest  # noqa: E402
from src.utils.text_norm import normalize, tokenize  # noqa: E402

TYPES = ["LID", "HALL", "EMPTY", "XTALK", "BOUND", "PSEUDO", "ACOUS"]


def render(ref_t, hyp_t, zh):
    sep, out = ("" if zh else " "), []
    for t, r0, r1, h0, h1 in alignment(ref_t, hyp_t):
        R, H = ref_t[r0:r1], hyp_t[h0:h1]
        if t == "equal":
            out.append(sep.join(R))
        elif t == "substitute":
            out.append(f"[{sep.join(R)}->{sep.join(H)}]")
        elif t == "delete":
            out.append(f"[-{sep.join(R)}]")
        else:
            out.append(f"[+{sep.join(H)}]")
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-ASR-1.7B")
    ap.add_argument("--mode", default="auto")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    suffix = "" if args.mode == "auto" else "__forced"
    out_root = ROOT / "data/datasets/samples/audit" / f"{args.model}_{args.mode}"
    rows, md = [], [f"# Error-type audit: {args.model} / {args.mode}", "",
                    "Listen, then fill `human_label` in audit.tsv with the type you think is right "
                    f"({', '.join(TYPES)}, or OTHER:<name>).",
                    "Alignment: [ref->hyp] substitution, [-ref] deletion, [+hyp] insertion.", ""]
    utt_dir = ROOT / "data/outputs/analysis/utt" / args.model
    for p in sorted(utt_dir.glob(f"*{suffix}.parquet")):
        if args.mode == "auto" and p.stem.endswith("__forced"):
            continue
        ds, cond = p.stem.split("__")[:2]
        df = pd.read_parquet(p)
        df["ACOUS"] = df[["ACOUS_S", "ACOUS_D", "ACOUS_I"]].sum(axis=1)
        df["dominant"] = df[TYPES].idxmax(axis=1).where(df[TYPES].max(axis=1) > 0, "")
        if ds == "notsofar" and cond == "sdm":  # one device per meeting is enough for listening
            first = df.groupby(df.session.str.split("_sc_").str[0]).session.transform("min")
            df = df[df.session == first]
        man = {r["id"]: r for r in read_manifest(ROOT / "data/datasets/manifests" / f"{ds}__{cond}.jsonl")}
        outs = {}
        out_file = ROOT / "data/outputs" / args.model / f"{ds}__{cond}{suffix}.jsonl"
        for line in open(out_file):
            r = json.loads(line)
            outs[r["id"]] = r
        lang = "zh" if ds == "alimeeting" else "en"
        md += [f"## {ds} / {cond}", ""]
        for t in TYPES:
            cand = df[df.dominant == t]
            if cand.empty:
                continue
            pick = cand.sample(min(args.n, len(cand)), random_state=args.seed)
            for k, (_, u) in enumerate(pick.iterrows(), 1):
                m, o = man[u.id], outs[u.id]
                fn = f"{t}_{k}_{u.id}.wav".replace("/", "-")
                d = out_root / ds / cond
                d.mkdir(parents=True, exist_ok=True)
                sf.write(d / fn, load_segment(m), SR, subtype="PCM_16")
                ref_t = tokenize(normalize(o["ref"], lang), lang)
                hyp_t = tokenize(normalize(o["hyp"] or "", lang), lang)
                ali = render(ref_t, hyp_t, lang == "zh")
                counts = " ".join(f"{c}={int(u[c])}" for c in TYPES if u[c])
                rows.append({"file": f"{ds}/{cond}/{fn}", "dataset": ds, "condition": cond, "auto_type": t,
                             "counts": counts, "dur": u.dur, "overlap": u.overlap, "snr_db": round(u.snr_db, 1),
                             "pred_lang": u.pred_lang, "ref_raw": o["ref"], "hyp_raw": o["hyp"], "alignment": ali,
                             "human_label": "", "note": ""})
                md += [f"**{t} {k}** `{ds}/{cond}/{fn}` — {u.dur:.1f} s, overlap {u.overlap:.0%}, "
                       f"SNR {u.snr_db:.1f} dB, pred_lang={u.pred_lang or '-'}; {counts}", "",
                       f"- ref: {o['ref']}", f"- hyp: {o['hyp']}", f"- alignment: {ali}", ""]
        print(f"{ds}/{cond}: {sum(r['dataset'] == ds and r['condition'] == cond for r in rows)} samples")
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "audit.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    (out_root / "README.md").write_text("\n".join(md), encoding="utf-8")
    print(f"{len(rows)} samples -> {out_root.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
