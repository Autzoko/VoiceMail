"""Phase 1 error analysis, step 1: one row per scored utterance per run, with error-type counts and conditions.

Step A (per manifest, cached; reads audio):  data/outputs/analysis/features/<ds>__<cond>.parquet
    dur, overlap (fraction of the segment covered by other speakers), n_ovl_spk, snr_db (WADA-SNR, blind),
    rms_db, native (AMI participants.xml), ctx / ctrl / own (normalized text of overlapping other speakers,
    of a time-shifted control window, and of the same speaker's adjacent segments).
Step B (per run = model x manifest x lang_mode):   data/outputs/analysis/utt/<model>/<run>.parquet
    S, D, I, n_ref, error-type counts (src/utils/error_taxonomy.py), xtalk_ctrl, hall_kind, pred_lang + features.

    python scripts/analysis/build_utt_table.py [--models Qwen3-ASR-0.6B Qwen3-ASR-1.7B] [--workers 32]
"""
import argparse
import bisect
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.error_taxonomy import CATS, tag_utterance  # noqa: E402
from src.utils.manifest import load_segment, read_manifest  # noqa: E402
from src.utils.text_norm import normalize, tokenize  # noqa: E402

AN = ROOT / "data" / "outputs" / "analysis"
PAD, OWN_GAP = 0.5, 1.0
SHIFTS = [60, -60, 120, -120, 180, -180, 300, -300]


# ---------------- WADA-SNR (Kim & Stern 2008), table built by simulation ----------------
def _wada_table():
    p = AN / "wada_table.npy"
    if p.exists():
        return np.load(p)
    rng = np.random.default_rng(0)
    s = rng.gamma(0.4, 1.0, 2_000_000) * rng.choice([-1, 1], 2_000_000)
    s /= np.sqrt(np.mean(s ** 2))
    n = rng.standard_normal(2_000_000)
    snrs = np.arange(-20, 101, 1.0)
    g = []
    for db in snrs:
        x = np.abs(s + n * 10 ** (-db / 20))
        x = np.maximum(x, 1e-20)
        g.append(np.log(np.mean(x)) - np.mean(np.log(x)))
    tab = np.stack([snrs, np.array(g)])
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, tab)
    return tab


_TAB = None


def _snr_rms(row):
    global _TAB
    if _TAB is None:
        _TAB = _wada_table()
    try:
        x = load_segment(row).astype(np.float64)
    except Exception:
        return np.nan, np.nan
    if len(x) < 1600 or not np.any(x):
        return np.nan, np.nan
    rms_db = 10 * np.log10(np.mean(x ** 2) + 1e-20)
    a = np.maximum(np.abs(x / np.max(np.abs(x))), 1e-10)
    g = np.log(np.mean(a)) - np.mean(np.log(a))
    snrs, gs = _TAB
    o = np.argsort(gs)  # G is monotonic in SNR; np.interp needs increasing x
    snr = float(np.interp(g, gs[o], snrs[o]))
    return snr, rms_db


# ---------------- per-manifest features ----------------
def ami_native():
    p = ROOT / "data/datasets/ami/corpusResources/participants.xml"
    if not p.exists():
        return {}
    out = {}
    for e in ET.parse(p).getroot().iter("participant"):
        pid = e.get("{http://nite.sourceforge.net/}id")
        out[pid] = "native" if (e.get("native_language") or "").lower() == "english" else "non-native"
    return out


def _window(segs, starts, maxdur, s, e):
    """Indices of segments in a session overlapping [s, e)."""
    hi = bisect.bisect_left(starts, e)
    lo = bisect.bisect_left(starts, s - maxdur)
    return [j for j in range(lo, hi) if segs[j]["end"] > s]


def build_features(name, workers):
    out = AN / "features" / f"{name}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    rows = read_manifest(ROOT / "data/datasets/manifests" / f"{name}.jsonl")
    lang = rows[0]["lang"]
    native = ami_native() if name.startswith("ami") else {}
    norm = {r["id"]: normalize(r["text"], lang) for r in rows}
    by_sess = defaultdict(list)
    for r in rows:
        by_sess[r["session"]].append(r)
    feats = []
    for sess, segs in by_sess.items():
        segs.sort(key=lambda r: r["start"])
        starts = [r["start"] for r in segs]
        maxdur = max(r["end"] - r["start"] for r in segs) + 2 * PAD + 2 * OWN_GAP
        for r in segs:
            s, e, spk = r["start"], r["end"], r["speaker"]
            dur = max(e - s, 1e-3)
            idx = _window(segs, starts, maxdur, s - PAD, e + PAD)
            others = [segs[j] for j in idx if segs[j]["speaker"] != spk]
            ivs = sorted((max(o["start"], s), min(o["end"], e)) for o in others if o["start"] < e and o["end"] > s)
            cov, cur_s, cur_e = 0.0, None, None
            for a, b in ivs:
                if cur_e is None or a > cur_e:
                    if cur_e is not None:
                        cov += cur_e - cur_s
                    cur_s, cur_e = a, b
                else:
                    cur_e = max(cur_e, b)
            if cur_e is not None:
                cov += cur_e - cur_s
            n_ovl = len({o["speaker"] for o in others if o["start"] < e and o["end"] > s})
            ctrl = []
            for sh in SHIFTS:
                if abs(sh) <= dur + 2 * PAD:
                    continue
                cj = _window(segs, starts, maxdur, s + sh - PAD, e + sh + PAD)
                ctrl = [segs[j] for j in cj if segs[j]["speaker"] != spk]
                if ctrl:
                    break
            own = [segs[j] for j in _window(segs, starts, maxdur, s - OWN_GAP, e + OWN_GAP)
                   if segs[j]["speaker"] == spk and segs[j]["id"] != r["id"]]
            sep = "" if lang == "zh" else " "
            feats.append({"id": r["id"], "session": sess, "speaker": spk, "start": s, "end": e,
                          "dur": round(dur, 3), "overlap": round(min(cov / dur, 1.0), 3), "n_ovl_spk": n_ovl,
                          "native": native.get(spk, "NA"),
                          "ctx": sep.join(norm[o["id"]] for o in others),
                          "ctrl": sep.join(norm[o["id"]] for o in ctrl),
                          "own": sep.join(norm[o["id"]] for o in own)})
    order = {f["id"]: i for i, f in enumerate(feats)}
    rows_sorted = sorted(rows, key=lambda r: order[r["id"]])
    with Pool(workers) as pool:
        sr = pool.map(_snr_rms, rows_sorted, chunksize=64)
    for f, (snr, rms) in zip(feats, sr):
        f["snr_db"], f["rms_db"] = snr, rms
    df = pd.DataFrame(feats)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"[features] {name}: {len(df)} segs, overlap>0: {(df.overlap > 0).mean():.1%}, "
          f"median SNR {df.snr_db.median():.1f} dB", flush=True)
    return df


# ---------------- per-run tagging ----------------
def _tag_one(args):
    rec, f, lang, forced = args
    ref_t = tokenize(normalize(rec["ref"], lang), lang)
    if not ref_t:
        return None
    hyp_norm = normalize(rec["hyp"] or "", lang)
    hyp_t = tokenize(hyp_norm, lang)
    tk = lambda s: tokenize(s or "", lang)  # noqa: E731
    o = tag_utterance(ref_t, hyp_t, hyp_norm, lang, rec.get("pred_lang"), forced, tk(f["ctx"]), tk(f["ctrl"]),
                      tk(f["own"]))
    o.update(id=rec["id"], n_ref=len(ref_t), n_hyp=len(hyp_t), pred_lang=rec.get("pred_lang") or "")
    return o


def build_run(model, out_file, feats, workers):
    stem = out_file.stem  # ds__cond[__forced]
    parts = stem.split("__")
    ds, cond, forced = parts[0], parts[1], len(parts) > 2
    dst = AN / "utt" / model / f"{stem}.parquet"
    if dst.exists() and dst.stat().st_mtime > out_file.stat().st_mtime:
        return
    fmap = feats.set_index("id")[["ctx", "ctrl", "own"]].to_dict("index")
    recs = list({r["id"]: r for r in map(json.loads, open(out_file)) if r["id"] in fmap}.values())  # dedupe
    lang = "zh" if ds == "alimeeting" else "en"
    with Pool(workers) as pool:
        res = [o for o in pool.map(_tag_one, [(r, fmap[r["id"]], lang, forced) for r in recs], chunksize=256) if o]
    df = pd.DataFrame(res).merge(feats.drop(columns=["ctx", "ctrl", "own"]), on="id", how="left")
    df.insert(0, "model", model)
    df.insert(1, "dataset", ds)
    df.insert(2, "condition", cond)
    df.insert(3, "lang_mode", "forced" if forced else "auto")
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    E = df[["S", "D", "I"]].sum().sum()
    print(f"[tag] {model} {stem}: {len(df)} utts, err={100 * E / df.n_ref.sum():.2f}%  " +
          " ".join(f"{c}={100 * df[c].sum() / df.n_ref.sum():.2f}" for c in CATS), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen3-ASR-0.6B", "Qwen3-ASR-1.7B"])
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    _wada_table()
    runs = sorted(p for m in args.models for p in (ROOT / "data/outputs" / m).glob("*.jsonl")
                  if not p.name.startswith("librispeech"))
    feats = {}
    for p in runs:
        name = "__".join(p.stem.split("__")[:2])
        if name not in feats:
            feats[name] = build_features(name, args.workers)
        build_run(p.parent.name, p, feats[name], args.workers)


if __name__ == "__main__":
    main()
