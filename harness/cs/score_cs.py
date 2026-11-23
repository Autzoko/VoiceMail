"""Score the twin (Vocali auto Qwen3-ASR + Audar routing, Nemotron-3 low) on the mixed zh/en/ar set.

The locked evaluator assumes one language per session; this set changes language every utterance, so it is scored
here, per reference utterance, with the evaluator's own normalizer (imported, not modified). Test / demo only.

    python -m harness.cs.score_cs --hyp data/outputs/harness/runs/cs_mixed \
        --ref data/datasets/manifests/cs_synth__mix.jsonl --out data/outputs/harness/runs/cs_mixed/score_cs
Per reference utterance: output lines are assigned by time overlap (largest overlap, else nearest).
  route_acc   routed language of the assigned output (word-weighted) == reference language
  engine_ok   Audar for Arabic, Qwen for Chinese / English
  err         edit distance on normalize.words units (zh characters, en / ar words) / reference units; ar also CER
  MER         all languages pooled (zh characters + en / ar words)
  merged_utt  share of the system's utterances that span reference utterances of two languages
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from harness.eval.normalize import chars, normalize, words

LANGS = ("zh", "en", "ar")


def edit(a, b):
    """Levenshtein distance with (sub, del, ins) counts."""
    prev = [(j, 0, 0, j) for j in range(len(b) + 1)]  # (cost, sub, del, ins)
    for i in range(1, len(a) + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, len(b) + 1):
            s = prev[j - 1]; s = (s[0] + (a[i - 1] != b[j - 1]), s[1] + (a[i - 1] != b[j - 1]), s[2], s[3])
            d = prev[j]; d = (d[0] + 1, d[1], d[2] + 1, d[3])
            n = cur[j - 1]; n = (n[0] + 1, n[1], n[2], n[3] + 1)
            cur.append(min(s, d, n))
        prev = cur
    return prev[-1]


def assign(lines, segs):
    out = [[] for _ in segs]
    for ln in lines:
        ov = [max(0.0, min(ln["end"], g["end"]) - max(ln["start"], g["start"])) for g in segs]
        if max(ov) > 0:
            k = ov.index(max(ov))
        else:
            mid = (ln["start"] + ln["end"]) / 2
            k = min(range(len(segs)), key=lambda i: abs(mid - (segs[i]["start"] + segs[i]["end"]) / 2))
        out[k].append(ln)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    refs = defaultdict(list)
    for line in open(a.ref, encoding="utf-8"):
        r = json.loads(line); refs[r["session"]].append(r)
    rows, merged, total_utt, missing = [], 0, 0, []
    for session, segs in sorted(refs.items()):
        p = Path(a.hyp) / f"{session}.json"
        if not p.exists():
            missing.append(session); h = {"lines": [], "finals": []}
        else:
            h = json.loads(p.read_text())
        segs = sorted(segs, key=lambda g: g["start"])
        for g, ls in zip(segs, assign(h["lines"], segs)):
            ls = sorted(ls, key=lambda x: x["start"])
            text = " ".join(x["text"] for x in ls)
            routed = Counter()
            for x in ls:
                routed[x.get("language")] += max(1, len(x["text"].split()))
            eng = Counter(x.get("engine") for x in ls)
            lang = g["lang"]
            R, H = words(normalize(g["text"], lang), lang), words(normalize(text, lang), lang)
            c, s, d, i = edit(R, H)
            row = dict(session=session, id=g["id"], lang=lang, ref=g["text"], hyp=text, ref_units=len(R), err=c,
                       sub=s, dele=d, ins=i, routed=routed.most_common(1)[0][0] if routed else None,
                       engine=eng.most_common(1)[0][0] if eng else None, n_lines=len(ls))
            if lang == "ar":
                rc = chars(normalize(g["text"], "ar")); hc = chars(normalize(text, "ar"))
                row.update(ref_chars=len(rc), char_err=edit(rc, hc)[0])
            rows.append(row)
        # system utterances spanning reference utterances of two languages (a language switch inside one utterance)
        for f in h.get("finals", []):
            total_utt += 1
            langs = {g["lang"] for g in segs if min(f["end"], g["end"]) - max(f["start"], g["start"]) > 0.3}
            merged += len(langs) > 1
    summ = {"sessions": len(refs), "missing_hypotheses": missing, "utterances_ref": len(rows),
            "utterances_sys": total_utt, "merged_utt%": round(100 * merged / max(1, total_utt), 2)}
    for lang in LANGS + ("all",):
        rs = [r for r in rows if lang == "all" or r["lang"] == lang]
        if not rs: continue
        want = {"ar": "audar"}
        summ[lang] = {
            "n": len(rs),
            "route_acc": round(100 * sum(r["routed"] == r["lang"] for r in rs) / len(rs), 2),
            "engine_ok": round(100 * sum(r["engine"] == want.get(r["lang"], "qwen") for r in rs) / len(rs), 2),
            "err": round(100 * sum(r["err"] for r in rs) / sum(r["ref_units"] for r in rs), 2),
            "del": round(100 * sum(r["dele"] for r in rs) / sum(r["ref_units"] for r in rs), 2),
            "ins": round(100 * sum(r["ins"] for r in rs) / sum(r["ref_units"] for r in rs), 2),
            "empty%": round(100 * sum(not r["hyp"].strip() for r in rs) / len(rs), 2),
            "routed_as": dict(Counter(r["routed"] for r in rs)),
        }
        if lang == "ar":
            summ[lang]["cer"] = round(100 * sum(r["char_err"] for r in rs) / sum(r["ref_chars"] for r in rs), 2)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "utterances.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
