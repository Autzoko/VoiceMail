"""Score the twin on code-switched speech whose utterances mix languages (intra-sentential zh/en/ar synthetic set,
ASCEND zh-en, MIXAT ar-en). Test / demo only; the locked evaluator is not used (one language per session).

    python -m harness.cs.score_mix --hyp RUN_DIR --ref MANIFEST --out RUN_DIR/score_mix
Reference text is split into script runs (Han -> zh, Arabic -> ar, Latin / digits -> en); each run goes through the
evaluator's normalizer for its language, giving tagged units (zh characters, en / ar words). The same is done to
the output. Per reference utterance (output lines assigned by time overlap) a Levenshtein alignment attributes
every reference unit to correct / substituted / deleted:
  MER                  all units pooled
  err_<lang>           (sub + del) of reference units of that language / their count
  swap_<lang>          reference units of that language substituted by a unit of another language
                       (translated or transliterated instead of transcribed)
  ins                  inserted output units / reference units
  engine_<utt type>    share of utterances decoded by Audar, per reference utterance type (zh / en / ar / mix-*)
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from harness.eval.normalize import normalize, words
from harness.cs.score_cs import assign

HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
ARA = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
LAT = re.compile(r"[A-Za-z0-9À-ɏ]")


def kind(c):
    return "zh" if HAN.match(c) else "ar" if ARA.match(c) else "en" if LAT.match(c) else None


def units(text):
    """[(unit, lang)] in order."""
    runs, cur, lang = [], "", None
    for c in text:
        k = kind(c)
        if k is None:
            cur += c; continue
        if lang is not None and k != lang:
            runs.append((cur, lang)); cur = ""
        cur += c; lang = k
    if cur and lang:
        runs.append((cur, lang))
    out = []
    for r, lg in runs:
        out += [(u, lg) for u in words(normalize(r, lg), lg)]
    return out


def align(R, H):
    """Levenshtein with backtrace. Returns per reference unit 'ok' / ('sub', hyp_lang) / 'del', and #insertions."""
    n, m = len(R), len(H)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): D[i][0] = i
    for j in range(m + 1): D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j - 1] + (R[i - 1][0] != H[j - 1][0]), D[i - 1][j] + 1, D[i][j - 1] + 1)
    i, j, out, ins = n, m, [None] * n, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (R[i - 1][0] != H[j - 1][0]):
            out[i - 1] = "ok" if R[i - 1][0] == H[j - 1][0] else ("sub", H[j - 1][1]); i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            out[i - 1] = "del"; i -= 1
        else:
            ins += 1; j -= 1
    return out, ins


def utt_type(u, tagged):
    langs = sorted({lg for _, lg in tagged})
    if u.get("lang") in ("zh", "en", "ar") and len(langs) <= 1:
        return u["lang"]
    return "mix-" + "".join(langs) if len(langs) > 1 else (langs[0] if langs else "empty")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    refs = defaultdict(list)
    for line in open(a.ref, encoding="utf-8"):
        r = json.loads(line); refs[r["session"]].append(r)
    tot = Counter(); per = defaultdict(Counter); eng = defaultdict(Counter); rows = []; missing = []
    for session, us in sorted(refs.items()):
        p = Path(a.hyp) / f"{session}.json"
        h = json.loads(p.read_text()) if p.exists() else {"lines": []}
        if not p.exists(): missing.append(session)
        us = sorted(us, key=lambda u: u["start"])
        for u, ls in zip(us, assign(h["lines"], us)):
            ls = sorted(ls, key=lambda x: x["start"])
            hyp = " ".join(x["text"] for x in ls)
            R, H = units(u["text"]), units(hyp)
            res, ins = align(R, H)
            t = utt_type(u, R)
            e = Counter(x.get("engine") for x in ls).most_common(1)
            eng[t][e[0][0] if e else "none"] += 1
            tot["ref"] += len(R); tot["ins"] += ins
            for (tok, lg), o in zip(R, res):
                per[lg]["n"] += 1
                if o != "ok":
                    per[lg]["err"] += 1; tot["err"] += 1
                    if isinstance(o, tuple) and o[1] != lg: per[lg]["swap"] += 1
            tot["err"] += 0
            rows.append(dict(session=session, start=u["start"], end=u["end"], type=t, ref=u["text"], hyp=hyp,
                             ref_units=len(R), err=sum(o != "ok" for o in res), ins=ins,
                             engine=e[0][0] if e else None))
    summ = {"sessions": len(refs), "utterances": len(rows), "missing_hypotheses": missing,
            "MER": round(100 * (tot["err"] + tot["ins"]) / max(1, tot["ref"]), 2),
            "ins": round(100 * tot["ins"] / max(1, tot["ref"]), 2)}
    for lg in ("zh", "en", "ar"):
        if per[lg]["n"]:
            summ[f"err_{lg}"] = round(100 * per[lg]["err"] / per[lg]["n"], 2)
            summ[f"swap_{lg}"] = round(100 * per[lg]["swap"] / per[lg]["n"], 2)
            summ[f"units_{lg}"] = per[lg]["n"]
    by_type = defaultdict(Counter)
    for r in rows:
        by_type[r["type"]]["n"] += 1; by_type[r["type"]]["units"] += r["ref_units"]
        by_type[r["type"]]["e"] += r["err"] + r["ins"]
    summ["by_type"] = {t: {"utterances": c["n"], "MER": round(100 * c["e"] / max(1, c["units"]), 2),
                           "engine": dict(eng[t])} for t, c in sorted(by_type.items())}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "utterances.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
