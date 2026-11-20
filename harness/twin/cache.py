"""Decode cache: what Qwen3-ASR / Audar return for one span of a session's audio (design doc §3.2).

A decode depends only on (engine, audio span, context): the router's Qwen pass always runs the zh/en/ar language
step and Audar is always forced to Arabic, and the `fast` flag changes nothing but speed. So the twin can replay
any threshold setting from cached decodes. Spans are quantized to 0.1 s (start floored, end rounded) so that
nearby settings share entries; the twin then uses the quantized span (an approximation, checked by the macmini
calibration).

File per session: {cache_dir}/{session}.jsonl, one JSON object per decode:
  {"engine": "qwen"|"audar", "q0": int, "q1": int, "ctx": str, "text": str,
   "tokens": [[start, end, text], ...]   seconds from the span start (VoicedClock spreading, as QwenDecoder)
   "language": "zh"|"en"|"ar", "probs": {"zh": p, "en": p, "ar": p} | null}
Lookups that miss are appended to {session}.misses.jsonl for the GPU filler (harness/twin/fill.py).
"""
import hashlib
import json
from pathlib import Path

from .router import Decoded

Q = 1600  # samples per 0.1 s


def quantize(span):
    s0, s1 = span
    q0 = s0 // Q
    q1 = max(q0 + 1, int(round(s1 / Q)))
    return q0, q1


def context_key(context):
    return hashlib.sha1((context or "").encode()).hexdigest()[:10] if context else ""


class DecodeCache:
    def __init__(self, cache_dir, session, context=None):
        self.dir = Path(cache_dir)
        self.session = session
        self.ctx = context_key(context)
        self.entries = {}
        self.misses = set()
        p = self.dir / f"{session}.jsonl"
        if p.exists():
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    e = json.loads(line)
                    self.entries[(e["engine"], e["q0"], e["q1"], e.get("ctx", ""))] = e

    def decode(self, engine, span):
        q0, q1 = quantize(span)
        e = self.entries.get((engine, q0, q1, self.ctx))
        if e is None:
            self.misses.add((engine, q0, q1, self.ctx))
            return None
        tokens = [(float(a), float(b), t.encode("utf-8")) for a, b, t in e["tokens"]]
        return Decoded(e["text"], tokens, e["language"], e.get("probs"))

    def offset(self, span):
        """Session seconds where the decoded (quantized) audio starts."""
        return quantize(span)[0] * Q / 16000

    def write_misses(self):
        if not self.misses:
            return 0
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / f"{self.session}.misses.jsonl", "a", encoding="utf-8") as f:
            for engine, q0, q1, ctx in sorted(self.misses):
                f.write(json.dumps({"engine": engine, "q0": q0, "q1": q1, "ctx": ctx}) + "\n")
        return len(self.misses)
