"""RoutedDecoder (Sources/VocaliEngine/RoutedDecoder.swift): Qwen judges, Audar transcribes.

`decode(info, span)` runs one live pass. `backend.decode(engine, span)` returns a Decoded (text, tokens relative to
the span, language, P(lang) for Qwen). Returns (result, passes, trace): `passes` lists (engine, seconds) actually
decoded, for the cost model.
"""
from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass


@dataclass
class Decoded:
    text: str
    tokens: list          # (start, end, bytes) seconds from the span start
    language: str
    probs: dict | None = None  # Qwen only: {"zh": p, "en": p, "ar": p}


def arabic_share(text, min_letters):
    """Share of Arabic letters; Foundation's CharacterSet.letters is Unicode categories L* and M*."""
    letters = arabic = 0
    for ch in text:
        if unicodedata.category(ch)[0] in "LM":
            letters += 1
            o = ord(ch)
            if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F:
                arabic += 1
    return None if letters < min_letters else arabic / letters


class Router:
    def __init__(self, settings, backend, forced=None):
        self.s = settings
        self.backend = backend
        self.forced = forced
        self.current = None           # dict(id, engine, decided, speaker, evidence, switched)
        self.speaker_arabic = {}
        self.last_arabic = None
        self.last_qwen_probs = {}     # QwenDecoder.lastLanguageProbabilities survives Audar passes

    def prior(self, speaker):
        if speaker is not None and speaker in self.speaker_arabic:
            return self.s["speakerWeight"] if self.speaker_arabic[speaker] else -self.s["speakerWeight"]
        if self.last_arabic is not None:
            return self.s["recentWeight"] if self.last_arabic else -self.s["recentWeight"]
        return 0.0

    def evidence(self, result, speaker):
        prior, prob, script = self.prior(speaker), 0.0, 0.0
        p = self.last_qwen_probs.get("ar")
        if p is not None:
            f = self.s["probabilityFloor"]
            q = min(max(p, f), 1 - f)
            prob = math.log(q / (1 - q))
        if result is not None:
            share = arabic_share(result.text, self.s["scriptMinLetters"])
            if share is not None:
                if share >= self.s["arabicShareHigh"]:
                    script = self.s["scriptWeight"]
                elif share <= self.s["arabicShareLow"]:
                    script = -self.s["scriptWeight"]
        return prior, prob, script

    def _run(self, engine, span):
        r = self.backend.decode(engine, span)
        if engine == "qwen" and r is not None:
            self.last_qwen_probs = dict(r.probs or {})
        if engine == "audar" and r is not None:
            r = Decoded(r.text, r.tokens, "ar", r.probs)
        return r

    def decode(self, utterance, final, speaker, span):
        seconds = (span[1] - span[0]) / 16000
        if self.forced:
            engine = "audar" if self.forced == "ar" else "qwen"
            return self._run(engine, span), [(engine, seconds)], {"engine": engine, "forced": self.forced}
        if self.current is None or self.current["id"] != utterance:
            self.current = {"id": utterance, "engine": "qwen", "decided": False, "speaker": speaker,
                            "evidence": 0.0, "switched": False}
        st = self.current
        if speaker is not None:
            st["speaker"] = speaker
        result = self._run(st["engine"], span)
        passes = [(st["engine"], seconds)]
        parts = None
        if st["engine"] == "qwen":
            parts = self.evidence(result, st["speaker"])
            st["evidence"] = sum(parts)
        switched = False
        if not st["decided"]:
            if st["evidence"] >= self.s["margin"] or (final and st["evidence"] > 0):
                st.update(engine="audar", switched=True, decided=True)
                switched = True
                result = self._run("audar", span)
                passes.append(("audar", seconds))
            elif seconds >= self.s["decideWithin"] or final:
                st["decided"] = True
        trace = {"utterance": utterance, "final": final, "seconds": seconds, "speaker": st["speaker"],
                 "pArabic": self.last_qwen_probs.get("ar") if parts else None,
                 "prior": parts[0] if parts else 0.0, "probability": parts[1] if parts else 0.0,
                 "script": parts[2] if parts else 0.0, "evidence": st["evidence"], "engine": st["engine"],
                 "decided": st["decided"], "switched": switched, "forced": None}
        if final:
            is_ar = st["engine"] == "audar"
            self.last_arabic = is_ar
            if st["speaker"] is not None and abs(st["evidence"]) >= self.s["rememberEvidence"]:
                self.speaker_arabic[st["speaker"]] = is_ar
            self.current = None
        return result, passes, trace
