"""Discrete-event replica of Vocali's live pipeline for one session (design doc §3.3).

Clock: session-audio seconds, as if the file were played at speed 1 (the replay harness's clock).
Threads of the app and how they are modelled:
  capture      one step per 512-sample block: Transcriber.accept (VAD state from frontend.py)
  diarizer     Nemotron observations arrive at their publication time (diar.observations) and go through
               TurnSplitter -> Transcriber.speakerActive / diarized / speakerChanged
  ASR queue    one serial worker; a decode takes `cost(engine, seconds)` seconds (cost model, measured on the
               macmini); its result, the pending count and the partial cost become visible when it ends
Ported from Sources/SpeakerInference/Transcriber.swift (external-engine path, prefix commitment off as in the app).
Not modelled yet: SteadyText display steadying (captions are the raw passes), translation, GPU contention with
the translator and the diarizer (folded into the cost model).
"""
from __future__ import annotations

import bisect
import json
from collections import Counter
from dataclasses import dataclass, field

from . import aligner
from .config import ENGINE
from .diar import TurnSplitter, decision_log
from .router import Router

SR = 16000
BLOCK = 512


@dataclass
class CostModel:
    """Seconds one decode takes on the target Mac: base + per_second x audio seconds, per engine."""
    base: dict = field(default_factory=lambda: {"qwen": 0.12, "audar": 0.12})
    per_second: dict = field(default_factory=lambda: {"qwen": 0.035, "audar": 0.035})
    calibrated: bool = False

    def __call__(self, engine, seconds):
        return self.base[engine] + self.per_second[engine] * seconds

    @classmethod
    def load(cls, path):
        d = json.loads(open(path).read())
        return cls(d["base"], d["per_second"], bool(d.get("calibrated", True)))


@dataclass
class Job:
    ticket: int
    utterance: int
    final: bool
    must_run: bool
    span: tuple
    start: float
    segment: int
    submit: float


class Twin:
    def __init__(self, cfg, backend, cost=None, forced_language=None, router=None):
        self.cfg = cfg
        tc = cfg["transcription"]
        self.partial_interval = tc["partialInterval"] if tc["partialInterval"] is not None else ENGINE[
            "decoderPartialInterval"]
        self.first_partial_after = tc["firstPartialAfter"]
        self.maximum = tc["maximumUtterance"]
        self.cost_factor = tc["partialCostFactor"]
        self.backend = backend
        self.cost = cost or CostModel()
        self.router = router or Router(cfg["router"], backend, forced_language)
        # capture-side state
        self.in_speech = False
        self.segment = 0
        self.utterance = 0
        self.utterance_start = 0.0
        self.last_partial = 0.0
        self.aud = [0, 0]           # current utterance audio [start, end) in samples
        self.ticket = 0
        self.submitted = 0
        self.vad_segment = 0
        # worker
        self.queue = []
        self.free = 0.0
        self.completions = []       # times at which a submitted decode finished (or was dropped)
        self.partial_costs = []     # (finish time, seconds)
        self.submit_times = []
        # diarizer -> transcriber (each with its arrival time)
        self.cuts = []              # (arrival, time)
        self.speakers = []          # (arrival, time, speaker)
        self.diarized = []          # (arrival, until)
        # outputs
        self.transcripts = []       # dicts: utterance, start, end, final, language, tokens, t, engine
        self.traces = []
        self.decoded_seconds = 0.0

    # ---------- capture thread ----------
    # completions, partial_costs and submit_times grow in time order (one serial worker), so bisect works
    def pending_at(self, t):
        return self.submitted - bisect.bisect_right(self.completions, t)

    def partial_cost_at(self, t):
        i = bisect.bisect_right(self.partial_costs, (t, float("inf")))
        return self.partial_costs[i - 1][1] if i else 0.0

    def accept(self, end_sample, speech, now):
        if speech and not self.in_speech:
            self.in_speech = True
            self.vad_segment += 1
            self.segment = self.vad_segment
            block_start = end_sample - BLOCK
            pre = min(ENGINE["prerollSamples"], block_start)
            self.start_utterance((block_start - pre) / SR)
            self.aud = [block_start - pre, block_start]
        if not self.in_speech:
            return
        self.aud[1] = end_sample
        ready = [c for c in self.cuts if c[0] <= now]
        self.cuts = [c for c in self.cuts if c[0] > now]
        for _, time in ready:
            self.split(time, now)
        duration = (self.aud[1] - self.aud[0]) / SR
        if not speech:
            self.in_speech = False
            self.finalize(now)
            return
        if duration >= self.maximum:
            self.finalize(now)
            self.start_utterance(now)
            self.aud = [end_sample, end_sample]
            return
        cost = self.partial_cost_at(now)
        if duration >= self.first_partial_after and now - self.last_partial >= max(self.partial_interval,
                                                                                 cost * self.cost_factor):
            self.last_partial = now
            self.decode(False, now)

    def split(self, time, now):
        index = int((time - self.utterance_start) * SR)
        count = self.aud[1] - self.aud[0]
        if index < 8000 or index > count - 4800:
            return
        rest = (self.aud[0] + index, self.aud[1])
        self.aud = [self.aud[0], self.aud[0] + index]
        self.decode(True, now, must_run=True)
        self.start_utterance(time)
        self.aud = list(rest)

    def start_utterance(self, time):
        self.utterance += 1
        self.utterance_start = time
        self.aud = [self.aud[1], self.aud[1]]
        self.last_partial = time

    def finalize(self, now):
        if (self.aud[1] - self.aud[0]) / SR >= ENGINE["minFinalSeconds"]:
            self.decode(True, now)
        self.aud = [self.aud[1], self.aud[1]]

    def decode(self, final, now, must_run=False):
        if not final and self.pending_at(now) > 0:
            return
        self.submitted += 1
        self.ticket += 1
        self.submit_times.append(now)
        self.queue.append(Job(self.ticket, self.utterance, final, must_run, tuple(self.aud), self.utterance_start,
                              self.segment, now))

    # ---------- diarizer thread ----------
    def speaker_at(self, start, as_of):
        i = bisect.bisect_right(self.diarized, (as_of, float("inf")))
        until = max((u for _, u in self.diarized[:i]), default=float("-inf")) if i else float("-inf")
        if until < start + 0.5:
            return None
        for a, time, spk in reversed(self.speakers):
            if a <= as_of and time <= start + 0.5:
                return spk
        return None

    # ---------- ASR queue ----------
    def run_worker(self, until):
        while self.queue:
            job = self.queue[0]
            s = max(self.free, job.submit)
            if s > until:
                return
            self.queue.pop(0)
            tickets_by_s = bisect.bisect_right(self.submit_times, s)
            behind = tickets_by_s - job.ticket
            if behind >= ENGINE["dropBehind"] and not job.must_run:
                self.completions.append(s)
                self.free = s
                continue
            speaker = self.speaker_at(job.start, s)
            result, passes, trace = self.router.decode(job.utterance, job.final, speaker, job.span)
            service = sum(self.cost(e, sec) for e, sec in passes)
            self.decoded_seconds += sum(sec for _, sec in passes)
            f = s + service
            self.free = f
            self.completions.append(f)
            if not job.final:
                self.partial_costs.append((f, service))
            self.traces.append({**trace, "t": f})
            if result is None:
                continue
            off = self.backend.offset(job.span)
            self.transcripts.append({
                "utterance": job.utterance, "segment": job.segment, "start": job.start,
                "end": job.start + (job.span[1] - job.span[0]) / SR, "final": job.final,
                "language": result.language, "engine": trace["engine"], "t": f,
                "tokens": [(off + a, off + b, raw) for a, b, raw in result.tokens]})

    # ---------- session ----------
    def run(self, blocks, observations, duration):
        """blocks: [(end_sample, speech)]; observations: diar.Observation list (arrival-sorted)."""
        splitter = TurnSplitter(self.cfg["activity"])
        obs = sorted(observations, key=lambda o: o.arrival)
        k = 0
        for end_sample, speech in blocks:
            now = end_sample / SR
            while k < len(obs) and obs[k].arrival <= now:
                o = obs[k]
                reports, decided, changes = splitter.accept(o)
                self.speakers += [(o.arrival, t, s) for t, s in reports]
                if decided is not None:
                    self.diarized.append((o.arrival, decided))
                self.cuts += [(o.arrival, t) for t in changes]
                k += 1
            self.run_worker(now)
            self.accept(end_sample, speech, now)
        if self.in_speech:  # the file ends while someone talks: the replay harness's stop finalizes
            self.in_speech = False
            self.finalize(blocks[-1][0] / SR if blocks else 0.0)
        self.run_worker(float("inf"))
        return self.hypothesis(observations, duration)

    def hypothesis(self, observations, duration):
        decisions = decision_log(sorted(observations, key=lambda o: o.start), self.cfg["activity"])
        lines, captions, utterances = [], [], {}
        finals = {}
        for tr in sorted(self.transcripts, key=lambda x: x["t"]):
            text = b"".join(raw for _, _, raw in tr["tokens"]).decode("utf-8", "ignore").strip()
            captions.append({"t": tr["t"], "utterance": tr["utterance"], "text": text, "final": tr["final"]})
            if tr["final"]:
                finals[tr["utterance"]] = tr
                utterances[str(tr["utterance"])] = [tr["start"], tr["end"]]
        lang_words = Counter()
        for u, tr in sorted(finals.items()):
            # The app keeps re-aligning a caption while new frames arrive and it is < 4 s old.
            visible = [d for d in decisions if d[0] <= tr["end"] + 4.0]
            for ln in aligner.lines(tr["start"], tr["end"], tr["tokens"], visible, ENGINE["alignerWindow"]):
                lines.append({**ln, "speaker": None if ln["speaker"] is None else str(ln["speaker"]),
                              "utterance": u, "engine": tr["engine"]})
                lang_words[tr["language"]] += len(ln["text"].split()) or 1
        return {"duration": duration, "language": lang_words.most_common(1)[0][0] if lang_words else None,
                "lines": lines, "captions": captions, "utterances": utterances, "translations": [],
                "cost": {"decode_audio_seconds": self.decoded_seconds, "decode_calls": len(self.traces),
                         "cost_model_calibrated": self.cost.calibrated},
                "routes": self.traces}
