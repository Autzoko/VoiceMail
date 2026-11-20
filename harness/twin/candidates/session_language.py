"""L1 candidate (a): session-level language posterior (design doc §5.2 a). Twin only until it passes the gate.

A session is in one language that is not known in advance. The baseline router decides every utterance from
scratch (prior = the person's / last utterance's language, +-1.0 / 0.5), so each Arabic utterance shows Qwen text
for its first 1-3 s and pays a second decode when it switches to Audar. Here the router accumulates the evidence
of finished utterances into a session log-odds L (with decay):

  unlocked   (|L| < lockAt)  : baseline routing, with prior = sessionWeight x L (clipped) instead of the
                               per-person / last-utterance prior
  locked ar  (L >= lockAt)   : every pass goes straight to Audar (no Qwen text first, no second decode); every
                               `probeEvery`-th final pass also runs Qwen's language step to keep L honest
  locked other (L <= -lockAt): baseline routing with the strong negative prior (Arabic must be very clear)
  unlock     : when L falls back inside +-unlockAt (contrary probes / contrary strong evidence)

Parameters live in the twin-only section "candidates.session_language" of a candidate file, not in the Swift
HarnessConfig (which rejects unknown keys) until the candidate is promoted and ported.
"""
from ..router import Router

DEFAULTS = {"sessionWeight": 1.0, "priorClip": 3.0, "lockAt": 4.0, "unlockAt": 2.0, "decay": 0.9,
            "evidenceClip": 4.0, "probeEvery": 4}


class SessionLanguageRouter(Router):
    def __init__(self, settings, backend, forced=None, params=None):
        super().__init__(settings, backend, forced)
        self.p = {**DEFAULTS, **(params or {})}
        self.L = 0.0
        self.locked = None          # "ar" | "other" | None
        self.finals = 0
        self.own = {}               # utterance -> its latest Qwen evidence without the prior (prob + script)
        self._utt = None

    def evidence(self, result, speaker):
        parts = super().evidence(result, speaker)
        if self._utt is not None:
            self.own[self._utt] = parts[1] + parts[2]
        return parts

    def prior(self, speaker):
        w, c = self.p["sessionWeight"], self.p["priorClip"]
        return max(-c, min(c, w * self.L))

    def _update(self, evidence):
        c = self.p["evidenceClip"]
        self.L = self.p["decay"] * self.L + max(-c, min(c, evidence))
        if self.locked is None and abs(self.L) >= self.p["lockAt"]:
            self.locked = "ar" if self.L > 0 else "other"
        elif self.locked is not None and abs(self.L) < self.p["unlockAt"]:
            self.locked = None

    def decode(self, utterance, final, speaker, span):
        if self.forced:
            return super().decode(utterance, final, speaker, span)
        seconds = (span[1] - span[0]) / 16000
        if self.locked == "ar":
            self._utt = None
            result = self._run("audar", span)
            passes = [("audar", seconds)]
            trace = {"utterance": utterance, "final": final, "seconds": seconds, "speaker": speaker,
                     "pArabic": None, "prior": self.prior(speaker), "probability": 0.0, "script": 0.0,
                     "evidence": self.L, "engine": "audar", "decided": True, "switched": False, "forced": None,
                     "session": "locked-ar"}
            if final:
                self.finals += 1
                if self.finals % self.p["probeEvery"] == 0:
                    probe = self._run("qwen", span)          # language step only matters; its text is not shown
                    passes.append(("qwen", seconds))
                    prior, prob, script = self.evidence(probe, speaker)
                    self._update(prob + script)               # the session prior itself is not evidence
                    trace["probe"] = prob + script
                self.last_arabic = True
            return result, passes, trace
        self._utt = utterance
        result, passes, trace = super().decode(utterance, final, speaker, span)
        trace["session"] = "locked-other" if self.locked else "open"
        if final:
            self.finals += 1
            own = self.own.pop(utterance, None)   # an utterance moved to Audar keeps its last Qwen evidence
            if own is not None:
                self._update(own)
        return result, passes, trace
