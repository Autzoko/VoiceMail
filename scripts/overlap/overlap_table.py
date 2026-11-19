"""Overlap study, step 2 (CPU): word-level overlap features + overlap-specific error attribution per utterance.

Inputs
    data/datasets/manifests/<ds>__<cond>.jsonl                 segments (oracle segmentation, same as phase 1)
    data/outputs/analysis/words/<ds>__<near>.jsonl             word times from align_words.py (close-talk)
    data/outputs/analysis/features/<ds>__<cond>.parquet        phase-1 features (rms_db, snr_db, overlap, ...)
    data/outputs/<model>/<ds>__<cond>__forced.jsonl           Qwen3-ASR hypotheses (fixed config: forced language)
Output
    data/outputs/analysis/overlap/<model>/<ds>__<cond>.parquet one row per scored utterance

Word clock. Every reference token of every segment gets [start, end] (aligner words -> normalized scoring tokens
by sequence matching; unmatched tokens interpolated). Far-field segments reuse the close-talk times of the same
segment shifted by (far.start - near.start) (non-zero only for DiPCo, whose array clocks differ).

Overlap features (target = the segment's speaker; interferers = all other speakers' words)
    w_ovl      fraction of target tokens whose word interval intersects an interferer word interval
    ovl_sec    seconds of target speech covered by interferer speech (union of word-interval intersections)
    n_int_spk  number of distinct interferer speakers with words overlapping the target
    int_type   'backchannel' (<=2 interferer words / <=3 zh chars inside the target span) | 'speech'
    position   'start' | 'end' | 'middle' | 'full' (w_ovl>=0.8) | 'mixed' -- where the overlapped target words are
    onset      'target_enters' (interferer already speaking at the target's first word) | 'interrupted'
    sir_db     target-to-dominant-interferer level on THIS microphone. far: median RMS of each speaker's
               non-overlapped segments on the device; near: own level on own headset minus the interferer's
               leakage level on the target's headset (measured at the interferer's non-overlapped segments).

Error attribution (partitions S+D+I of the same jiwer alignment as phase 1; LID/HALL/EMPTY/PSEUDO/BOUND
identical to phase 1). Per error region (consecutive non-equal alignment chunks); `local interferer words` =
interferer words within +-1 s of the region:
    SWAP     target words deleted/substituted AND the hypothesis words of the region are the interferer's words
             (>=50% bigram / unigram match): the model followed the other speaker
    INTRUDE  pure insertion of interferer words, target kept (phase-1 XTALK, now time-local)
    OVL_SUB / OVL_DEL / OVL_INS   other substitutions / deletions / insertions on overlapped target words
             (insertions: inside interferer speech) -- target corrupted or dropped, no interferer words emitted
    SUB / DEL / INS               errors on non-overlapped words (in overlapped utterances: spill-over)
    *_ctrl   the same SWAP/INTRUDE tests against interferer words of a window shifted by +60 s (chance level)

    python scripts/overlap/overlap_table.py [--models Qwen3-ASR-1.7B Qwen3-ASR-0.6B] [--workers 32]
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.error_taxonomy import (FILLERS_EN, FILLERS_ZH, ZH_EQUIV, ZH_NUM, _bigrams, _matches,  # noqa: E402
                                      alignment, tag_utterance)
from src.utils.manifest import SR, load_segment, read_manifest  # noqa: E402
from src.utils.text_norm import normalize, tokenize  # noqa: E402

AN = ROOT / "data" / "outputs" / "analysis"
MAN = ROOT / "data" / "datasets" / "manifests"
PAIRS = {"ami": ("ihm", "sdm"), "alimeeting": ("near", "far"), "notsofar": ("ihm", "sdm"),
         "chime6": ("ihm", "sdm"), "dipco": ("ihm", "sdm")}
CATS = ["LID", "HALL", "EMPTY", "PSEUDO", "BOUND", "SWAP", "INTRUDE", "OVL_SUB", "OVL_DEL", "OVL_INS",
        "SUB", "DEL", "INS"]
LOCAL, CTRL_SHIFT, WIN = 1.0, 60.0, 1.0


# ------------------------------------------------------------------ word clock
@lru_cache(maxsize=500_000)
def _norm_word(w, lang):
    return tuple(tokenize(normalize(w, lang), lang))


def token_times(ref_t, words, lang, s0, e0):
    """Map aligner words onto the normalized scoring tokens. Returns (starts, ends, matched_fraction)."""
    flat = []
    for w, s, e in words:
        toks = _norm_word(w, lang)
        if not toks:
            continue
        d = max(e - s, 0.0) / len(toks)
        flat += [(t, s + k * d, s + (k + 1) * d) for k, t in enumerate(toks)]
    n = len(ref_t)
    S, E = [None] * n, [None] * n
    matched = 0
    if flat:
        sm = SequenceMatcher(None, ref_t, [f[0] for f in flat], autojunk=False)
        for a, b, size in sm.get_matching_blocks():
            for k in range(size):
                S[a + k], E[a + k] = flat[b + k][1], flat[b + k][2]
                matched += 1
    i = 0
    while i < n:  # interpolate unmatched runs between known neighbours
        if S[i] is not None:
            i += 1
            continue
        j = i
        while j < n and S[j] is None:
            j += 1
        left = E[i - 1] if i > 0 else s0
        right = S[j] if j < n else e0
        if right < left:
            right = left
        d = (right - left) / (j - i)
        for k in range(i, j):
            S[k], E[k] = left + (k - i) * d, left + (k - i + 1) * d
        i = j
    return np.array(S, float), np.array(E, float), matched / max(n, 1)


def near_key(r, ds):
    spk = r["speaker"][2:] if ds == "alimeeting" and r["speaker"].startswith("N_") else r["speaker"]
    sess = r["session"].split("_sc_")[0]
    return (sess, spk, round(r["start"], 2), round(r["end"], 2), r["text"])


def build_clock(ds, cond):
    """Per segment of <ds>__<cond>: normalized ref tokens with absolute times on this condition's clock."""
    near_c = PAIRS[ds][0]
    lang = "zh" if ds == "alimeeting" else "en"
    near = read_manifest(MAN / f"{ds}__{near_c}.jsonl")
    words = {}
    for line in open(AN / "words" / f"{ds}__{near_c}.jsonl"):
        o = json.loads(line)
        words[o["id"]] = o["words"]
    near_by_id = {r["id"]: r for r in near}
    near_by_key = {near_key(r, ds): r for r in near}
    rows = near if cond == near_c else read_manifest(MAN / f"{ds}__{cond}.jsonl")
    cache, clock, nomap = {}, {}, 0
    for r in rows:
        if r["id"] in near_by_id:
            nr = near_by_id[r["id"]]
        elif ds == "notsofar" and "_sc_" in r["id"]:
            nr = near_by_id.get(r["id"].split("_sc_")[0])
        else:
            nr = near_by_key.get(near_key(r, ds))
        ref_t = tokenize(normalize(r["text"], lang), lang)
        if not ref_t:
            continue
        if nr is None or nr["id"] not in words:
            nomap += 1
            S, E, q = token_times(ref_t, [], lang, r["start"], r["end"])
            clock[r["id"]] = (ref_t, S, E, 0.0)
            continue
        if nr["id"] not in cache:
            cache[nr["id"]] = token_times(ref_t, words[nr["id"]], lang, nr["start"], nr["end"])
        S, E, q = cache[nr["id"]]
        sh = r["start"] - nr["start"]
        clock[r["id"]] = (ref_t, S + sh, E + sh, q)
    print(f"[clock] {ds}__{cond}: {len(clock)} segs, {nomap} without close-talk words, "
          f"mean token match {np.mean([c[3] for c in clock.values()]):.3f}", flush=True)
    return rows, clock


# ------------------------------------------------------------------ microphone levels -> SIR
def _rms_db(args):
    row = args
    try:
        x = load_segment(row).astype(np.float64)
    except Exception:
        return np.nan
    return 10 * np.log10(np.mean(x ** 2) + 1e-20) if len(x) > SR // 4 else np.nan


def mic_levels(ds, cond, rows, feats, workers):
    """{(session, target_spk, other_spk): level of other on target's mic} and {(session, spk): own level}."""
    f = feats.set_index("id")
    solo = [r for r in rows if r["id"] in f.index and f.at[r["id"], "overlap"] == 0 and r["end"] - r["start"] >= 1.0]
    own = defaultdict(list)
    for r in solo:
        own[(r["session"], r["speaker"])].append(f.at[r["id"], "rms_db"])
    own = {k: float(np.nanmedian(v)) for k, v in own.items()}
    if cond != PAIRS[ds][0]:  # far: everybody is recorded by the same device -> SIR = own(T) - own(O)
        return own, None
    # near: leakage of O into T's headset, measured at O's solo segments (max 15 per pair)
    rng = np.random.default_rng(0)
    mic = {}
    for r in rows:
        mic.setdefault((r["session"], r["speaker"]), (r["audio"], r["channel"]))
    by_spk = defaultdict(list)
    for r in solo:
        by_spk[(r["session"], r["speaker"])].append(r)
    jobs, keys = [], []
    for (sess, t), (audio, ch) in mic.items():
        for (s2, o), segs in by_spk.items():
            if s2 != sess or o == t:
                continue
            pick = [segs[i] for i in rng.choice(len(segs), min(15, len(segs)), replace=False)]
            for p in pick:
                jobs.append({"audio": audio, "channel": ch, "start": p["start"], "end": p["end"]})
                keys.append((sess, t, o))
    with Pool(workers) as pool:
        lv = pool.map(_rms_db, jobs, chunksize=32)
    leak = defaultdict(list)
    for k, v in zip(keys, lv):
        leak[k].append(v)
    leak = {k: float(np.nanmedian(v)) for k, v in leak.items()}
    return own, leak


# ------------------------------------------------------------------ per utterance
G = {}


def _session_index(rows, clock):
    """session -> sorted arrays of all tokens (start, end, token, speaker)."""
    by = defaultdict(list)
    for r in rows:
        c = clock.get(r["id"])
        if c is None:
            continue
        for t, s, e in zip(c[0], c[1], c[2]):
            by[r["session"]].append((s, e, t, r["speaker"]))
    idx = {}
    for sess, toks in by.items():
        toks.sort()
        st = np.array([x[0] for x in toks])
        en = np.array([x[1] for x in toks])
        idx[sess] = (st, en, [x[2] for x in toks], [x[3] for x in toks], float((en - st).max()))
    return idx


def _window(sess, spk, a, b):
    """Interferer tokens (other speakers) intersecting [a, b]: list of (s, e, tok, spk)."""
    st, en, tk, sp, maxd = G["idx"][sess]
    lo = int(np.searchsorted(st, a - maxd, "left"))
    hi = int(np.searchsorted(st, b, "right"))
    return [(st[i], en[i], tk[i], sp[i]) for i in range(lo, hi) if en[i] > a and st[i] < b and sp[i] != spk]


def _union(iv):
    iv = sorted(iv)
    tot, cs, ce = 0.0, None, None
    for a, b in iv:
        if ce is None or a > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = a, b
        else:
            ce = max(ce, b)
    return tot + (ce - cs if ce is not None else 0.0)


def analyze(args):
    rec, r, fe = args
    lang, forced = G["lang"], True
    zh = lang == "zh"
    ref_t, S, E, q = G["clock"][r["id"]]
    hyp_norm = normalize(rec["hyp"] or "", lang)
    hyp_t = tokenize(hyp_norm, lang)
    n = len(ref_t)
    sess, spk = r["session"], r["speaker"]
    a0, b0 = float(S[0]), float(E[-1])
    ints = _window(sess, spk, a0 - WIN, b0 + WIN)
    ctrl = _window(sess, spk, a0 - WIN + CTRL_SHIFT, b0 + WIN + CTRL_SHIFT)
    ctrl = [(s - CTRL_SHIFT, e - CTRL_SHIFT, t, p) for s, e, t, p in ctrl]

    # ---- token-level overlap
    ovl = np.zeros(n, bool)
    cover, int_spk, spk_time = [], set(), defaultdict(float)
    for s, e, t, p in ints:
        hit = (S < e) & (E > s)
        if hit.any():
            ovl |= hit
            int_spk.add(p)
            for i in np.flatnonzero(hit):
                lo, hi = max(S[i], s), min(E[i], e)
                cover.append((lo, hi))
                spk_time[p] += hi - lo
    w_ovl = ovl.mean()
    span_int = [x for x in ints if x[1] > a0 and x[0] < b0]
    fill = FILLERS_ZH if zh else FILLERS_EN
    n_int_tok = sum(1 for x in span_int if x[2] not in fill)
    if not ovl.any():
        position, onset, int_type = "none", "none", "none"
    else:
        k = np.flatnonzero(ovl)
        f0, f1 = k[0] / n, (k[-1] + 1) / n
        position = ("full" if w_ovl >= 0.8 else "start" if f1 <= 0.4 else "end" if f0 >= 0.6 else
                    "middle" if f0 >= 0.15 and f1 <= 0.85 else "mixed")
        onset = "target_enters" if any(s <= S[0] < e for s, e, _, _ in ints) else "interrupted"
        int_type = "backchannel" if n_int_tok <= (3 if zh else 2) else "speech"
    dom = max(spk_time, key=spk_time.get) if spk_time else None
    sir = np.nan
    if dom is not None:
        own, leak = G["own"], G["leak"]
        if leak is None:
            sir = own.get((sess, spk), np.nan) - own.get((sess, dom), np.nan)
        else:
            sir = own.get((sess, spk), np.nan) - leak.get((sess, spk, dom), np.nan)

    out = dict.fromkeys(CATS, 0)
    out.update(id=r["id"], n_ref=n, n_hyp=len(hyp_t), w_ovl=round(float(w_ovl), 3), n_ovl_tok=int(ovl.sum()),
               ovl_sec=round(_union(cover), 3), n_int_spk=len(int_spk), n_int_tok=n_int_tok, int_type=int_type,
               position=position, onset=onset, sir_db=round(float(sir), 2) if sir == sir else np.nan,
               clock_q=round(q, 3), SWAP_ctrl=0, INTRUDE_ctrl=0, err_ovl_tok=0, err_clean_tok=0)
    # ---- phase-1 utterance-level categories first (identical priority)
    own_t = tokenize(fe.get("own") or "", lang) if isinstance(fe.get("own"), str) else []
    base = tag_utterance(ref_t, hyp_t, hyp_norm, lang, rec.get("pred_lang"), forced, [x[2] for x in ints],
                         [x[2] for x in ctrl], own_t)
    out.update(S=base["S"], D=base["D"], I=base["I"])
    Etot = base["S"] + base["D"] + base["I"]
    for c in ("LID", "HALL", "EMPTY"):
        if base[c]:
            out[c] = Etot
            if c == "EMPTY":
                out["err_ovl_tok"] = int(ovl.sum())
                out["err_clean_tok"] = int(n - ovl.sum())
            return out
    if Etot == 0:
        return out

    chunks = alignment(ref_t, hyp_t)
    regions, cur = [], []
    for c in chunks:
        if c[0] == "equal":
            if cur:
                regions.append(cur)
            cur = []
        else:
            cur.append(c)
    if cur:
        regions.append(cur)
    nhyp = len(hyp_t)
    edge_ok, edge_max = n >= (6 if zh else 4), 3 if zh else 2
    own_set, own_bg = set(own_t), _bigrams(own_t)
    strip = lambda ts: re.sub(r"[\s\-']", "", "".join(ts))  # noqa: E731

    def local(pool, w0, w1):
        toks = [t for s, e, t, p in pool if e > w0 - LOCAL and s < w1 + LOCAL]
        return set(toks), _bigrams(toks)

    for reg in regions:
        r0, r1 = reg[0][1], reg[-1][2]
        h0, h1 = reg[0][3], reg[-1][4]
        R, H = ref_t[r0:r1], hyp_t[h0:h1]
        if r1 > r0:
            w0, w1 = S[r0], E[r1 - 1]
            in_ovl = bool(ovl[r0:r1].any())
        else:
            w0 = E[r0 - 1] if r0 > 0 else a0
            w1 = S[r0] if r0 < n else b0
            in_ovl = any(e > w0 and s < w1 for s, e, _, _ in ints) or \
                (r0 > 0 and ovl[r0 - 1]) or (r0 < n and ovl[r0])
        n_err = sum((c[2] - c[1]) if c[0] != "insert" else (c[4] - c[3]) for c in reg)
        num_swap = zh and ((any(c.isdigit() for c in R) and any(c in ZH_NUM for c in H)) or
                           (any(c.isdigit() for c in H) and any(c in ZH_NUM for c in R)))
        if strip(R) == strip(H) or num_swap:
            out["PSEUDO"] += n_err
            continue
        run = [h for h in H if h not in fill]
        iset, ibg = local(ints, w0, w1)
        cset, cbg = local(ctrl, w0, w1)
        m_int, m_ctl = _matches(run, iset, ibg), _matches(run, cset, cbg)
        at_edge = edge_ok and (r0 == 0 or r1 == n) and (r1 - r0) <= edge_max
        if r1 > r0 and run and m_ctl:
            out["SWAP_ctrl"] += n_err
        if r1 > r0 and run and m_int:  # target words replaced by the interferer's words
            nf = (h1 - h0) - len(run)
            out["SWAP"] += n_err - nf
            out["PSEUDO"] += nf
            out["err_ovl_tok"] += int(ovl[r0:r1].sum())
            out["err_clean_tok"] += int((~ovl[r0:r1]).sum())
            continue
        for t, x0, x1, y0, y1 in reg:
            if t == "substitute":
                for i, (rt, ht) in zip(range(x0, x1), zip(ref_t[x0:x1], hyp_t[y0:y1])):
                    if rt in fill or ht in fill or (zh and any(rt in s_ and ht in s_ for s_ in ZH_EQUIV)):
                        out["PSEUDO"] += 1
                        continue
                    out["err_ovl_tok" if ovl[i] else "err_clean_tok"] += 1
                    out["OVL_SUB" if ovl[i] else "BOUND" if at_edge else "SUB"] += 1
            elif t == "delete":
                for i in range(x0, x1):
                    if ref_t[i] in fill:
                        out["PSEUDO"] += 1
                        continue
                    out["err_ovl_tok" if ovl[i] else "err_clean_tok"] += 1
                    out["OVL_DEL" if ovl[i] else "BOUND" if at_edge else "DEL"] += 1
            else:
                ins = [h for h in hyp_t[y0:y1] if h not in fill]
                out["PSEUDO"] += (y1 - y0) - len(ins)
                if not ins:
                    continue
                if _matches(ins, cset, cbg):
                    out["INTRUDE_ctrl"] += len(ins)
                if _matches(ins, iset, ibg):
                    out["INTRUDE"] += len(ins)
                elif (y0 == 0 or y1 == nhyp) and _matches(ins, own_set, own_bg):
                    out["BOUND"] += len(ins)
                elif in_ovl:
                    out["OVL_INS"] += len(ins)
                else:
                    out["INS"] += len(ins)
    assert sum(out[c] for c in CATS) == Etot, (ref_t, hyp_t, out)
    return out


def build(ds, cond, models, workers):
    lang = "zh" if ds == "alimeeting" else "en"
    rows, clock = build_clock(ds, cond)
    feats = pd.read_parquet(AN / "features" / f"{ds}__{cond}.parquet")
    own, leak = mic_levels(ds, cond, rows, feats, workers)
    G.update(lang=lang, clock=clock, idx=_session_index(rows, clock), own=own, leak=leak)
    by_id = {r["id"]: r for r in rows}
    fmap = feats.set_index("id").to_dict("index")
    for model in models:
        hyp_file = ROOT / "data" / "outputs" / model / f"{ds}__{cond}__forced.jsonl"
        if not hyp_file.exists():
            print(f"  missing {hyp_file}", flush=True)
            continue
        recs = list({o["id"]: o for o in map(json.loads, open(hyp_file))}.values())
        jobs = [(o, by_id[o["id"]], fmap.get(o["id"], {})) for o in recs if o["id"] in clock and o["id"] in by_id]
        with Pool(workers) as pool:  # fork: workers inherit G
            res = pool.map(analyze, jobs, chunksize=256)
        df = pd.DataFrame(res)
        keep = ["id", "session", "speaker", "dur", "overlap", "snr_db", "rms_db", "native"]
        df = df.merge(feats[[c for c in keep if c in feats]], on="id", how="left")
        for k, v in (("model", model), ("dataset", ds), ("condition", cond), ("lang_mode", "forced")):
            df.insert(0, k, v)
        dst = AN / "overlap" / model / f"{ds}__{cond}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst, index=False)
        Etot = df[["S", "D", "I"]].sum().sum()
        print(f"[overlap] {model} {ds}__{cond}: {len(df)} utts err={100 * Etot / df.n_ref.sum():.2f} " +
              " ".join(f"{c}={100 * df[c].sum() / df.n_ref.sum():.2f}" for c in CATS), flush=True)


def check_aligner():
    """Aligner vs NOTSOFAR-1 human word timings (close-talk): boundary error of exactly matched words."""
    rows = read_manifest(MAN / "notsofar__ihm.jsonl")
    words = {json.loads(l)["id"]: json.loads(l)["words"] for l in open(AN / "words" / "notsofar__ihm.jsonl")}
    gts = {}
    root = ROOT / "data/datasets/notsofar/benchmark-datasets/eval_set/240825.1_eval_full_with_GT/MTG"
    ds, de = [], []
    for r in rows:
        mtg, idx = r["session"], int(r["id"].split("_")[2])
        if mtg not in gts:
            gts[mtg] = json.load(open(root / mtg / "gt_transcription.json"))
        gt = [(w.lower().strip(".,?!"), s, e) for w, s, e in gts[mtg][idx].get("word_timing", [])]
        al = [(w.lower(), s, e) for w, s, e in words.get(r["id"], [])]
        if not gt or not al:
            continue
        sm = SequenceMatcher(None, [g[0] for g in gt], [a[0] for a in al], autojunk=False)
        for a, b, size in sm.get_matching_blocks():
            for k in range(size):
                ds.append(abs(gt[a + k][1] - al[b + k][1]))
                de.append(abs(gt[a + k][2] - al[b + k][2]))
    ds, de = np.array(ds), np.array(de)
    res = {"n_words": len(ds), "start_med_ms": 1000 * np.median(ds), "start_p90_ms": 1000 * np.percentile(ds, 90),
           "end_med_ms": 1000 * np.median(de), "end_p90_ms": 1000 * np.percentile(de, 90),
           "start_within_100ms": float((ds <= 0.1).mean()), "start_within_200ms": float((ds <= 0.2).mean())}
    out = ROOT / "data/results/overlap"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([res]).round(3).to_csv(out / "aligner_check.csv", index=False)
    print("[aligner vs NOTSOFAR GT]", res, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen3-ASR-1.7B", "Qwen3-ASR-0.6B"])
    ap.add_argument("--datasets", nargs="+", default=list(PAIRS))
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    if "notsofar" in args.datasets:
        check_aligner()
    for ds in args.datasets:
        for cond in PAIRS[ds]:
            build(ds, cond, args.models, args.workers)


if __name__ == "__main__":
    main()
