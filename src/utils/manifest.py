"""Unified segment manifest (one JSON object per line) shared by all datasets.

Fields:
    id       unique segment id
    session  meeting/session id
    speaker  speaker id
    audio    absolute wav path
    channel  channel index to read from `audio` (0-based)
    start    segment start (seconds) in `audio`
    end      segment end (seconds) in `audio`
    text     raw reference transcription (un-normalized)
    lang     "en" | "zh"
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def write_manifest(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_manifest(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_segment(row) -> np.ndarray:
    """Read [start, end) of one channel as float32 mono at 16 kHz."""
    info = sf.info(row["audio"])
    assert info.samplerate == SR, f"{row['audio']} sr={info.samplerate}"
    s, e = int(round(row["start"] * SR)), int(round(row["end"] * SR))
    x, _ = sf.read(row["audio"], start=s, stop=min(e, info.frames), dtype="float32", always_2d=True)
    return np.ascontiguousarray(x[:, row.get("channel", 0)])
