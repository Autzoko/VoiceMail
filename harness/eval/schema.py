"""Hypothesis file format (one JSON object per session) and the reference loader.

Hypothesis (written by harness/twin and by the Vocali replay adapter; times in seconds of session audio):
{
  "session": "MTG_30860_sc_rockfall_0_mc",
  "duration": 312.4,                       # seconds of audio
  "language": "en",                        # session language the system settled on (null if it never did)
  "lines": [                               # final, speaker-attributed caption lines
    {"speaker": "1", "start": 3.2, "end": 5.9, "text": "...", "utterance": 0, "engine": "qwen"}
  ],
  "captions": [                            # every text the user saw, in emission order (partials and finals)
    {"t": 4.1, "utterance": 0, "text": "...", "final": false}
  ],
  "utterances": {"0": [3.0, 5.9]},         # audio span of each utterance (for first/final lag)
  "translations": [                        # final translation per utterance (optional)
    {"utterance": 0, "source": "en", "target": "ar", "text": "..."}
  ],
  "cost": {"decode_audio_seconds": 410.2, "decode_calls": 512}
}
`t` is on the audio clock: the moment (in session-audio seconds) the text appeared, as if played at speed 1.

Reference: rows of the shared segment manifests (src/utils/manifest.py: id, session, speaker, start, end, text,
lang), optionally with `translation` (reference translation of the segment into `translation_lang`).
"""
import json
from collections import defaultdict
from pathlib import Path

from .normalize import LANGS

REQUIRED = ("session", "duration", "lines", "captions", "utterances")


class SchemaError(ValueError):
    pass


def load_hypothesis(path) -> dict:
    h = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in h]
    if missing:
        raise SchemaError(f"{path}: missing {missing}")
    for line in h["lines"]:
        for k in ("speaker", "start", "end", "text"):
            if k not in line:
                raise SchemaError(f"{path}: line without {k!r}: {line}")
        if line["end"] < line["start"]:
            raise SchemaError(f"{path}: line ends before it starts: {line}")
    last = -1e9
    for c in h["captions"]:
        if c["t"] < last - 1e-6:
            raise SchemaError(f"{path}: captions not in emission order at t={c['t']}")
        last = c["t"]
    h.setdefault("translations", [])
    h.setdefault("cost", {})
    return h


def load_hypotheses(directory) -> dict:
    out = {}
    for p in sorted(Path(directory).glob("*.json")):
        h = load_hypothesis(p)
        if h["session"] in out:
            raise SchemaError(f"session {h['session']} appears twice in {directory}")
        out[h["session"]] = h
    return out


def load_references(manifests) -> dict:
    """session -> {"lang", "segments": [...]} from one or more manifest JSONL files."""
    sessions = defaultdict(list)
    for m in manifests:
        with open(m, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    sessions[r["session"]].append(r)
    out = {}
    for s, rows in sessions.items():
        langs = {r["lang"] for r in rows}
        if len(langs) != 1 or not langs <= set(LANGS):
            raise SchemaError(f"reference session {s}: languages {langs} (one of {LANGS} expected)")
        out[s] = {"lang": langs.pop(), "segments": sorted(rows, key=lambda r: (r["start"], r["end"]))}
    return out
