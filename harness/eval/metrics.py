"""Per-session counts. Every reported metric is a ratio of two summed counts (METRICS), so a corpus value is
sum(num) / sum(den) over sessions and the gate can bootstrap sessions without re-scoring anything.

ASR (meeteval, on the normalized units of normalize.words):
    orc   speaker-agnostic error rate (ORC-WER: every reference segment may go to any output stream)
    cp    cpWER (speaker permutation fixed per session)
    tcp   tcpWER, collar 5 s (word timing approximated inside a line, meeteval default)
    dcp   cp - orc: errors caused by speaker attribution alone
    ar sessions are also scored per character (orc_char) because Arabic WER over-penalizes clitic splits
Constraints / anti-gaming:
    del, ins           deletion / insertion share of the orc alignment (empty output must not pay off)
    foreign            share of output units in a script that never belongs to the session language
    silence_words/min  output units more than 0.5 s away from any reference speech (hallucination in silence)
    spk_abs_err        |estimated - true| number of speakers; spk_exact: share of sessions with it right
Real-time UX (same unit rules as Vocali tools/streaming_eval.py):
    first_lag, final_lag, NE (normalized erasure), big_rewrites/min, cost_rtf (decoded audio seconds / audio)
Translation:
    chrf   chrF++ of the session's translation (utterances in order) against the reference translation
"""
import unicodedata

from .normalize import FOREIGN, chars, normalize, script, words

COLLAR = 5.0
SILENCE_GAP = 0.5
BIG = 4

# name: (numerator key, denominator key, scale, lower is better)
METRICS = {
    "orc": ("orc_err", "orc_len", 100, True),
    "cp": ("cp_err", "cp_len", 100, True),
    "tcp": ("tcp_err", "tcp_len", 100, True),
    "dcp": ("dcp_err", "orc_len", 100, True),
    "orc_char": ("orcc_err", "orcc_len", 100, True),
    "del": ("orc_del", "orc_len", 100, True),
    "ins": ("orc_ins", "orc_len", 100, True),
    "foreign": ("foreign_units", "hyp_units", 100, True),
    "silence_words/min": ("silence_units", "minutes", 1, True),
    "spk_abs_err": ("spk_abs_err", "sessions", 1, True),
    "spk_exact": ("spk_exact", "sessions", 100, False),
    "first_lag": ("first_lag_sum", "first_lag_n", 1, True),
    "final_lag": ("final_lag_sum", "final_lag_n", 1, True),
    "NE": ("erased", "final_units", 1, True),
    "big_rewrites/min": ("big", "minutes", 1, True),
    "cost_rtf": ("decode_audio_seconds", "audio_seconds", 1, True),
    "chrf": ("chrf_sum", "chrf_n", 1, False),
}


# ---------------- ASR ----------------
def _seglst(session, rows, lang, text_key):
    from meeteval.io import SegLST

    items = []
    for r in rows:
        units = words(normalize(r[text_key], lang), lang)
        items.append({"session_id": session, "speaker": str(r["speaker"]), "start_time": float(r["start"]),
                      "end_time": float(r["end"]), "words": " ".join(units)})
    if not items:  # meeteval needs the session on both sides
        items.append({"session_id": session, "speaker": "none", "start_time": 0.0, "end_time": 0.0, "words": ""})
    return SegLST(items)


def _char_seglst(seglst):
    from meeteval.io import SegLST

    return SegLST([{**s, "words": " ".join(chars(s["words"]))} for s in seglst])


def _one(result, session):
    er = result[session]
    return er.errors, er.length, er.insertions, er.deletions


def asr_counts(session, ref_segments, lang, hyp_lines):
    from meeteval.wer import cpwer, greedy_orcwer, tcpwer

    # ORC-WER (speaker-agnostic): greedy assignment. Exact ORC is exponential in the number of hypothesis
    # streams (9 diarized streams on a NOTSOFAR session needed ~79e9 GB, job 19520224); greedy ORC is the
    # MeetEval variant for many streams and gives an upper bound that is tight in practice.

    ref = _seglst(session, ref_segments, lang, "text")
    hyp = _seglst(session, hyp_lines, lang, "text")
    o_err, o_len, o_ins, o_del = _one(greedy_orcwer(reference=ref, hypothesis=hyp), session)
    c_err, c_len, _, _ = _one(cpwer(reference=ref, hypothesis=hyp), session)
    t_err, t_len, _, _ = _one(tcpwer(reference=ref, hypothesis=hyp, collar=COLLAR), session)
    out = {"orc_err": o_err, "orc_len": o_len, "orc_ins": o_ins, "orc_del": o_del,
           "cp_err": c_err, "cp_len": c_len, "tcp_err": t_err, "tcp_len": t_len, "dcp_err": c_err - o_err}
    if lang == "ar":
        cc_err, cc_len, _, _ = _one(greedy_orcwer(reference=_char_seglst(ref), hypothesis=_char_seglst(hyp)), session)
        out.update(orcc_err=cc_err, orcc_len=cc_len)
    return out


def _merge(intervals, pad):
    out = []
    for a, b in sorted((a - pad, b + pad) for a, b in intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def output_counts(ref_segments, lang, hyp_lines, duration):
    speech = _merge([(r["start"], r["end"]) for r in ref_segments], SILENCE_GAP)
    foreign = total = silence = 0
    for line in hyp_lines:
        units = words(normalize(line["text"], lang), lang)
        total += len(units)
        foreign += sum(script(u) in FOREIGN[lang] for u in units)
        span = max(line["end"] - line["start"], 1e-6)
        for i in range(len(units)):  # units spread evenly over the line
            t = line["start"] + span * (i + 0.5) / len(units)
            silence += not any(a <= t <= b for a, b in speech)
    true_spk = len({str(r["speaker"]) for r in ref_segments})
    est_spk = len({str(l["speaker"]) for l in hyp_lines if l["speaker"] is not None and l["text"].strip()})
    return {"foreign_units": foreign, "hyp_units": total, "silence_units": silence, "minutes": duration / 60,
            "spk_abs_err": abs(est_spk - true_spk), "spk_exact": int(est_spk == true_spk), "sessions": 1,
            "spk_true": true_spk, "spk_est": est_spk}


# ---------------- real-time UX (unit rules copied from Vocali tools/streaming_eval.py) ----------------
def _ideographic(c):
    o = ord(c)
    return (0x3040 <= o <= 0x30FF or 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF
            or 0xAC00 <= o <= 0xD7AF)


def units(text):
    out, cur, word = [], "", False
    for c in text:
        if _ideographic(c):
            if cur:
                out.append(cur)
            cur, word = c, False
        elif c == " ":
            cur += c
            out.append(cur)
            cur, word = "", False
        elif unicodedata.category(c).startswith("P") and cur and not word:
            cur += c
        else:
            if cur and not word and not cur.endswith(" "):
                out.append(cur)
                cur = ""
            cur += c
            word = True
    if cur:
        out.append(cur)
    return out


def _norm_unit(u):
    return u.strip().lower()


def _lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if _norm_unit(x) != _norm_unit(y):
            break
        n += 1
    return n


def ux_counts(captions, utterances, duration):
    by_utt = {}
    for c in captions:
        by_utt.setdefault(str(c["utterance"]), []).append(c)
    erased = final_units = big = 0
    first, final = [], []
    for u, events in by_utt.items():
        shown = [units(e["text"]) for e in events]
        final_units += len(shown[-1])
        for i in range(1, len(shown)):
            if [_norm_unit(x) for x in shown[i]] == [_norm_unit(x) for x in shown[i - 1]]:
                continue
            gone = len(shown[i - 1]) - _lcp(shown[i - 1], shown[i])
            erased += gone
            big += gone >= BIG
        if u in utterances:
            a, b = utterances[u]
            first.append(events[0]["t"] - a)
            final.append(events[-1]["t"] - b)
    return {"erased": erased, "final_units": final_units, "big": big,
            "first_lag_sum": sum(first), "first_lag_n": len(first),
            "final_lag_sum": sum(final), "final_lag_n": len(final)}


# ---------------- translation ----------------
def chrf_counts(ref_segments, translations):
    refs = [r.get("translation") for r in ref_segments]
    if not any(refs):
        return {"chrf_sum": 0.0, "chrf_n": 0}
    from sacrebleu.metrics import CHRF

    ref = " ".join(t for t in refs if t)
    hyp = " ".join(t["text"] for t in sorted(translations, key=lambda t: t["utterance"]))
    return {"chrf_sum": CHRF(word_order=2).sentence_score(hyp, [ref]).score, "chrf_n": 1}


# ---------------- one session ----------------
def session_counts(session, reference, hypothesis):
    lang, segs = reference["lang"], reference["segments"]
    if hypothesis is None:  # a missing output file scores as total deletion, never as "skipped"
        hypothesis = {"duration": max((s["end"] for s in segs), default=0.0), "lines": [], "captions": [],
                      "utterances": {}, "translations": [], "cost": {}}
    out = {"session": session, "lang": lang, "detected": hypothesis.get("language")}
    out.update(asr_counts(session, segs, lang, hypothesis["lines"]))
    out.update(output_counts(segs, lang, hypothesis["lines"], hypothesis["duration"]))
    out.update(ux_counts(hypothesis["captions"], {str(k): v for k, v in hypothesis["utterances"].items()},
                         hypothesis["duration"]))
    out.update(decode_audio_seconds=float(hypothesis["cost"].get("decode_audio_seconds", 0.0)),
               audio_seconds=float(hypothesis["duration"]))
    out.update(chrf_counts(segs, hypothesis["translations"]))
    return out


def value(rows, name):
    num, den, scale, _ = METRICS[name]
    d = sum(r.get(den, 0) for r in rows)
    return scale * sum(r.get(num, 0) for r in rows) / d if d else None
