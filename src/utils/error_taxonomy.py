"""Rule-based attribution of every scored ASR error (S/D/I token) to one error type.

The categories partition the errors exactly: for each utterance sum(category counts) == S + D + I of the
same jiwer alignment used for WER/CER, so a corpus WER decomposes into per-category percentage points.

Utterance-level categories (if one fires, ALL errors of the utterance go to it), in priority order:
    LID     wrong language / translated output. English sets: hypothesis contains CJK/kana/hangul, or (auto
            mode) the predicted language is not English. Mandarin set: >50% latin letters or kana/hangul in the
            hypothesis, or (auto mode) predicted language is not Chinese (e.g. Cantonese).
    HALL    hallucination / runaway decoding: an n-gram (n=1..4) repeated >=4 times in a row in the hypothesis
            and >=3 more times than in the reference ("repeat"); or, after the word-level pass, unexplained
            insertions >= max(8, |ref|) for English / max(15, |ref|) for Mandarin ("long").
    EMPTY   empty hypothesis (after normalization) for a non-empty reference.
Token-level categories, per alignment op:
    PSEUDO  scoring artifacts, not recognition errors: an error region whose ref and hyp are equal once spaces,
            hyphens and apostrophes are removed (compound splits, "all right"/"alright"); Arabic vs Chinese
            numerals (zh); ops that involve a filler/backchannel token; zh 他/她/它 and 的/地/得 swaps.
    XTALK   inserted tokens that were said by ANOTHER speaker overlapping in time (+-0.5 s): a run of >=2
            insertions with >=50% of its bigrams in the other speakers' reference text, or a single inserted
            token present in it. The same test against a time-shifted window (no overlap) is reported as
            `xtalk_ctrl` to estimate chance matches.
    BOUND   segment-edge errors: deletions/substitutions in an error region touching the first or last ref token
            (<=2 words / <=3 chars, utterance >=4 words / >=6 chars), and edge insertions matching the same
            speaker's adjacent segments (+-1 s).
    ACOUS_S / ACOUS_D / ACOUS_I   everything else (substitution / deletion / insertion).
"""
import re
from itertools import groupby

import jiwer

CATS = ["LID", "HALL", "EMPTY", "XTALK", "BOUND", "PSEUDO", "ACOUS_S", "ACOUS_D", "ACOUS_I"]

FILLERS_EN = {"mm", "mmm", "mhm", "hm", "hmm", "um", "uh", "uhm", "erm", "er", "eh", "ah", "aha", "oh", "ooh",
              "huh", "uhhuh", "uh-huh", "mm-hmm", "mmhmm", "hmmm", "ha", "haha"}
FILLERS_ZH = set("嗯啊呃额哦噢唉哎诶欸喔唔哈嘿呀哇")
ZH_EQUIV = [set("他她它"), set("的地得")]
CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
KANA_HANGUL = re.compile(r"[぀-ヿ가-힯]")
LATIN = re.compile(r"[a-z]")
ZH_NUM = set("零一二两三四五六七八九十百千万亿")


def max_repeat(tokens, nmax=4):
    """Largest number of consecutive repetitions of any n-gram (n<=nmax)."""
    best = 1 if tokens else 0
    for n in range(1, nmax + 1):
        grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        for off in range(n):  # consecutive repeats are n apart
            seq = grams[off::n]
            for _, g in groupby(seq):
                best = max(best, sum(1 for _ in g))
    return best


def is_lid_error(hyp_norm, hyp_t, pred_lang, lang, forced):
    pl = (pred_lang or "").lower()
    if lang == "zh":
        if KANA_HANGUL.search(hyp_norm):
            return True
        if hyp_t and sum(bool(LATIN.match(c)) for c in hyp_t) > 0.5 * len(hyp_t):
            return True
        return (not forced) and pl != "" and "chinese" not in pl
    if CJK.search(hyp_norm):
        return True
    return (not forced) and pl != "" and "english" not in pl


def _bigrams(t):
    return set(zip(t, t[1:]))


def _matches(run, ctx_tokens, ctx_bigrams):
    if not run or not ctx_tokens:
        return False
    if len(run) == 1:
        return run[0] in ctx_tokens
    bg = list(zip(run, run[1:]))
    return sum(b in ctx_bigrams for b in bg) >= 0.5 * len(bg)


def alignment(ref_t, hyp_t):
    """List of (type, r0, r1, h0, h1) chunks (jiwer >= 3)."""
    if not hyp_t:
        return [("delete", 0, len(ref_t), 0, 0)]
    o = jiwer.process_words(" ".join(ref_t), " ".join(hyp_t))
    return [(c.type, c.ref_start_idx, c.ref_end_idx, c.hyp_start_idx, c.hyp_end_idx) for c in o.alignments[0]]


def tag_utterance(ref_t, hyp_t, hyp_norm, lang, pred_lang, forced, ctx, ctrl, own):
    """ref_t/hyp_t: scoring tokens. ctx/ctrl/own: token lists of overlapping other speakers, the control window,
    and the same speaker's adjacent segments. Returns dict with S, D, I, per-category counts, flags."""
    zh = lang == "zh"
    chunks = alignment(ref_t, hyp_t)
    S = sum(r1 - r0 for t, r0, r1, _, _ in chunks if t == "substitute")
    D = sum(r1 - r0 for t, r0, r1, _, _ in chunks if t == "delete")
    I = sum(h1 - h0 for t, _, _, h0, h1 in chunks if t == "insert")
    out = dict.fromkeys(CATS, 0)
    out.update(S=S, D=D, I=I, xtalk_ctrl=0, hall_kind="")
    E = S + D + I
    if E == 0:
        return out
    if is_lid_error(hyp_norm, hyp_t, pred_lang, lang, forced):
        out["LID"] = E
        return out
    hr, rr = max_repeat(hyp_t), max_repeat(ref_t)
    if hr >= 4 and hr >= rr + 3:
        out["HALL"], out["hall_kind"] = E, "repeat"
        return out
    if not hyp_t:
        out["EMPTY"] = E
        return out

    fill = FILLERS_ZH if zh else FILLERS_EN
    nref, nhyp = len(ref_t), len(hyp_t)
    edge_ok = nref >= (6 if zh else 4)
    edge_max = 3 if zh else 2
    ctx_set, ctx_bg = set(ctx), _bigrams(ctx)
    ctrl_set, ctrl_bg = set(ctrl), _bigrams(ctrl)
    own_set, own_bg = set(own), _bigrams(own)

    # group consecutive non-equal chunks into error regions
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

    for reg in regions:
        r0, r1 = reg[0][1], reg[-1][2]
        h0, h1 = reg[0][3], reg[-1][4]
        R, H = ref_t[r0:r1], hyp_t[h0:h1]
        n_err = sum((c[2] - c[1]) if c[0] != "insert" else (c[4] - c[3]) for c in reg)
        strip = lambda ts: re.sub(r"[\s\-']", "", "".join(ts))  # noqa: E731
        num_swap = zh and ((any(c.isdigit() for c in R) and any(c in ZH_NUM for c in H)) or
                           (any(c.isdigit() for c in H) and any(c in ZH_NUM for c in R)))
        if strip(R) == strip(H) or num_swap:
            out["PSEUDO"] += n_err
            continue
        at_edge = edge_ok and (r0 == 0 or r1 == nref) and (r1 - r0) <= edge_max
        for t, a0, a1, b0, b1 in reg:
            if t == "substitute":
                for r, h in zip(ref_t[a0:a1], hyp_t[b0:b1]):
                    if r in fill or h in fill or (zh and any(r in s and h in s for s in ZH_EQUIV)):
                        out["PSEUDO"] += 1
                    elif at_edge:
                        out["BOUND"] += 1
                    else:
                        out["ACOUS_S"] += 1
            elif t == "delete":
                for r in ref_t[a0:a1]:
                    out["PSEUDO" if r in fill else "BOUND" if at_edge else "ACOUS_D"] += 1
            else:  # insert
                run = [h for h in hyp_t[b0:b1] if h not in fill]
                out["PSEUDO"] += (b1 - b0) - len(run)
                if not run:
                    continue
                out["xtalk_ctrl"] += len(run) if _matches(run, ctrl_set, ctrl_bg) else 0
                if _matches(run, ctx_set, ctx_bg):
                    out["XTALK"] += len(run)
                elif (b0 == 0 or b1 == nhyp) and _matches(run, own_set, own_bg):
                    out["BOUND"] += len(run)
                else:
                    out["ACOUS_I"] += len(run)

    if out["ACOUS_I"] >= max(15 if zh else 8, nref):
        for c in CATS:
            out[c] = 0
        out["HALL"], out["hall_kind"] = E, "long"
    assert sum(out[c] for c in CATS) == E, (ref_t, hyp_t, out)
    return out
