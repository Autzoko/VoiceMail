"""Overlap study, controlled experiment: re-mix REAL non-overlapped far-field segments into overlaps.

Why: in real meetings overlap amount, loudness, position, interferer type, speaker count are confounded (and with
duration / SNR). Here they are set by design while everything else stays natural:
  * target and interferer are both real conversational speech of the SAME session, recorded by the SAME far-field
    device (same room, reverberation, mic, noise floor) -- no simulated RIRs, no read speech;
  * both are "solo" segments (no other speaker's word within +-1 s), and the interferer is cut at its own word
    boundaries (aligner times, 50 ms margin, 20 ms fades) so no word is chopped;
  * backchannel interferers are placed in a pause of the target (>=150 ms gap), like real backchannels;
  * every target is also decoded alone (`clean`), so effects are paired within the same target segment;
  * adding a far-field segment also adds its noise floor: `noise twins` add a speech-free window of the same
    device with the same gain and placement, separating "more noise" from "another talker".

Design per target (random, balanced across targets):
    clean                     target alone (identical cut)
    6 overlap variants        type: speech (75%) | backchannel (25%)
                              sir_set: natural (gain 1, recorded levels) | -5 | 0 | 5 | 10 | 20 dB
                              ratio_set (speech): 0.25 | 0.5 | 0.75 | 1.0 of the target's speech span
                              position (speech): start (interferer already talking, ends inside) | end (interrupts
                              and keeps talking) | middle (contained) | full (ratio 1.0)
                              n_int: 1 (80%) | 2 (20%, speech only; second talker at the same SIR)
    2 noise twins             of two of the overlap variants

Output: data/datasets/remix/<ds>/<id>.wav, manifests remix__<ds>.jsonl (run_asr.py compatible: whole file),
        data/datasets/remix/<ds>.meta.jsonl (design, interferer words on the mixture clock, gains)

    python scripts/overlap/make_remix.py [--targets 500] [--datasets ami alimeeting notsofar chime6 dipco]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "overlap")]
from overlap_table import PAIRS, build_clock  # noqa: E402
from src.utils.manifest import SR, load_segment, write_manifest  # noqa: E402

OUT = ROOT / "data" / "datasets" / "remix"
MAN = ROOT / "data" / "datasets" / "manifests"
FILL = {"en": {"mm", "mmm", "mhm", "hm", "hmm", "um", "uh", "uhm", "erm", "er", "eh", "ah", "aha", "oh", "huh",
               "uhhuh", "mmhmm", "yeah", "yes", "okay", "ok", "right", "no", "yep", "sure"},
        "zh": set("嗯啊呃额哦噢唉哎诶欸喔唔哈嘿呀哇对好是")}
MARGIN, FADE = 0.05, 0.02


def _power_db(x):
    return 10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-20)


def _fade(x):
    n = min(int(FADE * SR), len(x) // 2)
    if n > 0:
        w = np.linspace(0, 1, n, dtype=np.float32)
        x[:n] *= w
        x[-n:] *= w[::-1]
    return x


class Pool_:
    """Per-session pools of solo targets, solo interferers (speech / backchannel) and speech-free windows."""

    def __init__(self, ds, rows, clock, lang):
        self.lang = lang
        by = defaultdict(list)
        for r in rows:
            if r["id"] in clock:
                by[r["session"]].append(r)
        self.sess = {}
        for sess, segs in by.items():
            toks = sorted((s, e, r["speaker"]) for r in segs for s, e in zip(clock[r["id"]][1], clock[r["id"]][2]))
            st = np.array([t[0] for t in toks])
            en = np.array([t[1] for t in toks])
            spk = np.array([t[2] for t in toks])
            maxd = float((en - st).max()) if len(st) else 0.0

            def others(a, b, who):
                lo = int(np.searchsorted(st, a - maxd))
                hi = int(np.searchsorted(st, b, "right"))
                m = (en[lo:hi] > a) & (st[lo:hi] < b) & (spk[lo:hi] != who)
                return bool(m.any())

            tg, sp, bc = [], [], []
            for r in segs:
                ref_t, S, E, q = clock[r["id"]]
                a, b = float(S[0]), float(E[-1])
                if q < 0.8 or others(a - 1.0, b + 1.0, r["speaker"]):
                    continue
                dur, n = b - a, len(ref_t)
                if 2.0 <= dur <= 12.0 and n >= (8 if lang == "zh" else 5):
                    tg.append(r)
                if dur >= 1.5 and n >= (6 if lang == "zh" else 4):
                    sp.append(r)
                if n <= (3 if lang == "zh" else 2) and 0.2 <= dur <= 1.5 and all(t in FILL[lang] for t in ref_t):
                    bc.append(r)
            # speech-free windows: gaps between all segments (+-0.5 s), >= 1.5 s
            iv = sorted((r["start"] - 0.5, r["end"] + 0.5) for r in segs)
            gaps, ce = [], iv[0][1] if iv else 0
            for a, b in iv[1:]:
                if a - ce >= 1.5:
                    gaps.append((ce, a))
                ce = max(ce, b)
            self.sess[sess] = {"targets": tg, "speech": sp, "bc": bc, "gaps": gaps,
                               "audio": (segs[0]["audio"], segs[0]["channel"])}


def cut_words(clock_row, row, want, rng, mode):
    """Contiguous words of an interferer segment: returns (t0, t1) absolute, word list [(tok, s, e)].
    mode 'any': ~want seconds starting at a random word; 'tail'/'head': last/first words totalling ~want."""
    ref_t, S, E, _ = clock_row
    n = len(ref_t)
    if mode == "head":
        i0 = 0
    elif mode == "tail":
        i0 = n - 1
        while i0 > 0 and E[n - 1] - S[i0 - 1] <= want:
            i0 -= 1
    else:
        cands = [i for i in range(n) if E[n - 1] - S[i] >= want] or [0]
        i0 = int(rng.choice(cands))
    i1 = i0 + 1
    while i1 < n and E[i1 - 1] - S[i0] < want:
        i1 += 1
    t0, t1 = max(S[i0] - MARGIN, row["start"]), min(E[i1 - 1] + MARGIN, row["end"])
    return t0, t1, [(ref_t[k], float(S[k]), float(E[k])) for k in range(i0, i1)]


def read(audio, ch, t0, t1):
    return load_segment({"audio": audio, "channel": ch, "start": t0, "end": t1})


def make(ds, n_targets, seed):
    cond = PAIRS[ds][1]
    lang = "zh" if ds == "alimeeting" else "en"
    rows, clock = build_clock(ds, cond)
    if ds == "notsofar":  # one single-channel device per meeting (the first), all meetings
        first = {}
        for r in rows:
            m = r["session"].split("_sc_")[0]
            first.setdefault(m, r["session"])
        rows = [r for r in rows if first[r["session"].split("_sc_")[0]] == r["session"]]
    pool = Pool_(ds, rows, clock, lang)
    rng = np.random.default_rng(seed)
    cand = [(s, t) for s, p in pool.sess.items() if len(p["speech"]) and len({r["speaker"] for r in p["speech"]}) > 1
            for t in p["targets"]]
    rng.shuffle(cand)
    # balance sessions: round-robin over sessions
    by_s = defaultdict(list)
    for s, t in cand:
        by_s[s].append(t)
    picks, k = [], 0
    while len(picks) < n_targets and any(by_s.values()):
        for s in list(by_s):
            if by_s[s]:
                picks.append((s, by_s[s].pop()))
            if len(picks) >= n_targets:
                break
        k += 1
    odir = OUT / ds
    odir.mkdir(parents=True, exist_ok=True)
    man, meta = [], []
    for ti, (sess, tr) in enumerate(picks):
        P = pool.sess[sess]
        audio, ch = P["audio"]
        ref_t, S, E, _ = clock[tr["id"]]
        seg0, seg1 = tr["start"], tr["end"]
        xt = read(audio, ch, seg0, seg1)
        a0, b0 = float(S[0]) - seg0, float(E[-1]) - seg0  # target speech span on mixture clock
        span = b0 - a0
        pt = _power_db(xt[int(a0 * SR):int(b0 * SR)])
        tgt_words = [(t, float(s) - seg0, float(e) - seg0) for t, s, e in zip(ref_t, S, E)]
        others = [r for r in P["speech"] if r["speaker"] != tr["speaker"]]
        bcs = [r for r in P["bc"] if r["speaker"] != tr["speaker"]]
        base = {"ds": ds, "target_id": tr["id"], "cluster_id": f"{ds}/{sess.split('_sc_')[0]}/{tr['speaker']}",
                "target_words": tgt_words, "span": [a0, b0]}

        def emit(vid, x, info, ints):
            mid = f"{ds}_{ti:04d}_{vid}"
            path = odir / f"{mid}.wav"
            peak = float(np.max(np.abs(x))) if len(x) else 0.0
            if peak > 0.99:
                x = x * (0.99 / peak)
            sf.write(path, x, SR, subtype="PCM_16")
            man.append({"id": mid, "session": mid, "speaker": "T", "audio": str(path), "channel": 0, "start": 0.0,
                        "end": round(len(x) / SR, 3), "text": tr["text"], "lang": lang})
            meta.append({"id": mid, **base, **info, "interferers": ints})

        emit("clean", xt.copy(), {"variant": "clean", "type": "none", "sir_set": "none", "ratio_set": 0.0,
                                  "position_set": "none", "n_int": 0, "noise_twin": False}, [])
        variants = []
        for v in range(6):
            typ = "backchannel" if (bcs and rng.random() < 0.25) else "speech"
            sir_set = rng.choice(["natural", "-5", "0", "5", "10", "20"])
            if typ == "speech":
                ratio = float(rng.choice([0.25, 0.5, 0.75, 1.0]))
                pos = "full" if ratio == 1.0 else str(rng.choice(["start", "middle", "end"]))
                n_int = 2 if (rng.random() < 0.2 and len({r["speaker"] for r in others}) > 1) else 1
            else:
                ratio, pos, n_int = 0.0, "pause", 1
            variants.append((typ, sir_set, ratio, pos, n_int))
        twins = set(rng.choice(6, 2, replace=False).tolist())
        for v, (typ, sir_set, ratio, pos, n_int) in enumerate(variants):
            x = xt.copy()
            ints, placed, gains = [], [], []
            used = set()
            for j in range(n_int):
                if typ == "speech":
                    cands = [r for r in others if r["speaker"] not in used] or others
                    ir = cands[int(rng.integers(len(cands)))]
                    used.add(ir["speaker"])
                    want = ratio * span
                    if pos == "start":
                        t0, t1, w = cut_words(clock[ir["id"]], ir, want + 1.0, rng, "tail")
                        off = (a0 + want) - (t1 - t0)  # interferer ends `want` s into the target speech
                    elif pos == "end":
                        t0, t1, w = cut_words(clock[ir["id"]], ir, want + 1.0, rng, "head")
                        off = b0 - want
                    elif pos == "middle":
                        t0, t1, w = cut_words(clock[ir["id"]], ir, want, rng, "any")
                        off = a0 + (span - (t1 - t0)) / 2
                    else:  # full
                        t0, t1, w = cut_words(clock[ir["id"]], ir, span + 0.5, rng, "any")
                        off = a0 - 0.25
                else:
                    ir = bcs[int(rng.integers(len(bcs)))]
                    t0, t1 = max(ir["start"], clock[ir["id"]][1][0] - MARGIN), min(ir["end"], clock[ir["id"]][2][-1] + MARGIN)
                    w = [(t, float(s), float(e)) for t, s, e in zip(*clock[ir["id"]][:3])]
                    gaps = [(tgt_words[k][2], tgt_words[k + 1][1]) for k in range(len(tgt_words) - 1)
                            if tgt_words[k + 1][1] - tgt_words[k][2] >= 0.15 and 0.2 * span <= tgt_words[k][2] - a0 <= 0.8 * span]
                    g0 = gaps[int(rng.integers(len(gaps)))][0] if gaps else a0 + span * float(rng.uniform(0.3, 0.7))
                    off = g0 - 0.05
                xi = _fade(read(audio, ch, t0, t1).copy())
                pi = _power_db(xi)
                g = 1.0 if sir_set == "natural" else 10 ** ((pt - pi - float(sir_set)) / 20)
                sir_act = pt - (pi + 20 * np.log10(g))
                i0 = int(round(off * SR))
                s_src, s_dst = max(0, -i0), max(0, i0)
                L = min(len(xi) - s_src, len(x) - s_dst)
                if L > 0:
                    x[s_dst:s_dst + L] += g * xi[s_src:s_src + L]
                sh = off - t0
                for tok, s, e in w:
                    if s + sh < len(x) / SR and e + sh > 0:
                        ints.append((tok, round(s + sh, 3), round(e + sh, 3), f"I{j + 1}"))
                placed.append((off, t1 - t0, i0, s_src, s_dst, L))
                gains.append((g, round(float(sir_act), 2)))
            info = {"variant": f"v{v}", "type": typ, "sir_set": sir_set, "ratio_set": ratio, "position_set": pos,
                    "n_int": n_int, "noise_twin": False, "gains": gains,
                    "sir_act": round(float(np.mean([s for _, s in gains])), 2)}
            emit(f"v{v}", x, info, ints)
            if v in twins and P["gaps"]:
                xn = xt.copy()
                for (off, dur, i0, s_src, s_dst, L), (g, _) in zip(placed, gains):
                    gs = [gp for gp in P["gaps"] if gp[1] - gp[0] >= dur] or [max(P["gaps"], key=lambda q: q[1] - q[0])]
                    ga = gs[int(rng.integers(len(gs)))]
                    z = _fade(read(audio, ch, ga[0], ga[0] + min(dur, ga[1] - ga[0])).copy())
                    Lz = min(len(z) - s_src, len(xn) - s_dst)
                    if Lz > 0:
                        xn[s_dst:s_dst + Lz] += g * z[s_src:s_src + Lz]
                emit(f"v{v}n", xn, {**info, "variant": f"v{v}n", "noise_twin": True}, [])
        if ti % 50 == 0:
            print(f"  [{ds}] {ti}/{len(picks)} targets", flush=True)
    write_manifest(man, MAN / f"remix__{ds}.jsonl")
    with open(OUT / f"{ds}.meta.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    hrs = sum(r["end"] for r in man) / 3600
    print(f"[remix] {ds}: {len(picks)} targets (of {len(cand)} candidates), {len(man)} mixtures, {hrs:.2f} h", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(PAIRS))
    ap.add_argument("--targets", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for i, ds in enumerate(args.datasets):
        make(ds, args.targets, args.seed + i)


if __name__ == "__main__":
    main()
