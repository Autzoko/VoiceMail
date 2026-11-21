"""Speaker side of the twin, ported from Vocali:
    ActivityTracker  Sources/SpeakerCore/ActivityTracker.swift (hysteresis, confirm, dominant with hold)
    TurnSplitter     Sources/VocaliEngine/EngineOutput.swift   (speaker reports and cuts for the Transcriber)
    decision log     Sources/VocaliKit/Model/AppModel.swift consumeActivity (per-frame decisions, corrected back
                     by `hold` when someone takes over)
Input: Nemotron-3-Diarization frame probabilities (T x S, 0..1, 10 ms frames) from harness/diar, averaged to
80 ms rows exactly as NemotronStream.publish does.
Arrival times model NemotronStream: frames come in chunks of `chunk` frames, published `lookahead + compute`
seconds after the chunk's last frame ends (preset low: 9 frames = 0.72 s, look-ahead 0.32 s).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STEP = 0.08


@dataclass
class Observation:
    start: float
    step: float
    probs: np.ndarray  # frames x slots, whole percent (uint8), as the helper sends them
    arrival: float


def observations(probs, chunk=9, lookahead=0.32, compute=0.05, step=STEP, frame=STEP):
    """probs: T x S at `frame` seconds (Nemotron-3 high-resolution output is 10 ms; NeMo and FluidAudio alike).
    As NemotronStream.publish: each published chunk (`chunk` rows of `step`) averages groups of step/frame frames
    in Float32 and rounds to whole percent half away from zero (Swift .rounded())."""
    group = int(round(step / frame))
    x = np.asarray(probs, np.float32)
    x = x.reshape(x.shape[-2], x.shape[-1]) if x.ndim == 3 else x
    out = []
    per = chunk * group
    for i in range(0, len(x), per):
        block = x[i:i + per]
        rows = []
        for f in range(0, len(block), group):
            m = block[f:f + group].sum(axis=0, dtype=np.float32) / np.float32(len(block[f:f + group]))
            rows.append(np.clip(np.floor(m * np.float32(100) + np.float32(0.5)), 0, 100))
        rows = np.array(rows, np.uint8)
        start = i * frame
        end = start + len(rows) * step
        out.append(Observation(start, step, rows, end + lookahead + compute))
    return out


@dataclass
class Frame:
    time: float
    dominant: int | None
    active: list


@dataclass
class ActivityTracker:
    on_th: float = 0.5
    off_th: float = 0.35
    confirm: float = 0.6
    hold: float = 0.24
    known: set = field(default_factory=set)
    dominant: int | None = None
    on: list = field(default_factory=list)
    talked: list = field(default_factory=list)
    candidate: tuple | None = None

    def accept(self, obs):
        frames = []
        for index, row in enumerate(obs.probs):
            n = len(row)
            if len(self.on) < n:
                self.on += [False] * (n - len(self.on))
                self.talked += [0.0] * (n - len(self.talked))
            end = obs.start + (index + 1) * obs.step
            p = [int(v) / 100 for v in row]
            for s in range(n):
                self.on[s] = p[s] >= self.on_th or (self.on[s] and p[s] >= self.off_th)
                if self.on[s]:
                    self.talked[s] += obs.step
                if self.on[s] and s not in self.known and self.talked[s] >= self.confirm:
                    self.known.add(s)
            live = [s for s in range(n) if self.on[s] and s in self.known]
            loudest = _argmax(live, p)
            current = None if self.dominant is None else self.dominant - 1
            if current is not None and current in live:
                if loudest is not None and loudest != current:
                    if self.candidate is None or self.candidate[0] != loudest:
                        self.candidate = (loudest, end - obs.step)
                    if end - self.candidate[1] >= self.hold - 1e-9:
                        self.dominant = loudest + 1
                        self.candidate = None
                else:
                    self.candidate = None
            else:
                self.dominant = None if loudest is None else loudest + 1
                self.candidate = None
            frames.append(Frame(end, self.dominant, [s + 1 for s in live]))
        return frames


def _argmax(indices, p):
    """Swift `max(by: <)` keeps the FIRST of equal maxima (it replaces only on strictly greater)."""
    best = None
    for s in indices:
        if best is None or p[best] < p[s]:
            best = s
    return best


class TurnSplitter:
    """Produces, per observation, the Transcriber calls: speakerActive(speaker, since), diarized(until),
    speakerChanged(at)."""

    def __init__(self, settings):
        self.tracker = ActivityTracker(settings["on"], settings["off"], settings["confirm"], settings["hold"])
        self.last = None
        self.run_start = {}
        self.last_top = {}

    def accept(self, obs):
        tr = self.tracker
        changes, reports = [], []
        frames = tr.accept(obs)
        on_pct, off_pct = round(tr.on_th * 100), round(tr.off_th * 100)
        for index, frame in enumerate(frames):
            start = frame.time - obs.step
            row = obs.probs[index]
            for slot, value in enumerate(row):
                v = float(value)
                if v >= on_pct or (slot in self.run_start and v >= off_pct):
                    self.run_start.setdefault(slot, start)
                else:
                    self.run_start.pop(slot, None)
            top = _argmax(range(len(row)), [int(x) for x in row])
            if top is not None and float(row[top]) >= on_pct:
                self.last_top[top] = frame.time
            speaker = frame.dominant
            if speaker is None:
                continue
            if self.last is None:
                reports.append((frame.time - tr.hold, speaker))
            if self.last is not None and self.last != speaker:
                began = self.run_start.get(speaker - 1, frame.time - tr.hold)
                cut = max(began, self.last_top.get(self.last - 1, began))
                changes.append(cut)
                reports.append((cut, speaker))
            self.last = speaker
        decided = frames[-1].time - tr.hold if frames else None
        return reports, decided, changes


def decision_log(obs_list, settings):
    """AppModel's per-frame decisions (time, speaker) after its take-over correction, for TranscriptAligner."""
    tr = ActivityTracker(settings["on"], settings["off"], settings["confirm"], settings["hold"])
    log = []
    for obs in obs_list:
        for frame in tr.accept(obs):
            if frame.dominant is not None and log and log[-1][1] is not None and log[-1][1] != frame.dominant:
                before = log[-1][1]
                frm = frame.time - tr.hold - 0.001
                i = len(log) - 1
                while i >= 0 and log[i][0] > frm:
                    if log[i][1] == before:
                        log[i] = (log[i][0], frame.dominant)
                    i -= 1
            log.append((frame.time, frame.dominant))
    return log
