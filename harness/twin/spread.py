"""QwenDecoder.spread + VoicedClock (Sources/VocaliEngine/QwenDecoder.swift): Qwen3-ASR gives no word times, so
the app spreads the text's units over the voiced 20 ms frames of the decoded audio, in proportion to their length.
"""
import unicodedata

import numpy as np

FRAME = 320


def _is_cjk(ch):
    o = ord(ch)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF


def _is_punct(ch):
    # Swift Character.isPunctuation: Unicode general category P*
    return unicodedata.category(ch).startswith("P")


def units(text):
    out, cur = [], ""
    for ch in text:
        if _is_cjk(ch):
            if cur:
                out.append(cur)
                cur = ""
            out.append(ch)
        elif ch == " ":
            cur += ch
            out.append(cur)
            cur = ""
        else:
            if out and not cur and _is_punct(ch) and not out[-1].endswith(" "):
                out[-1] += ch
            else:
                cur += ch
    if cur:
        out.append(cur)
    return out


class VoicedClock:
    def __init__(self, samples):
        x = np.asarray(samples, np.float32)
        self.duration = len(x) / 16000
        count = len(x) // FRAME
        levels = np.sqrt((x[:count * FRAME].reshape(count, FRAME).astype(np.float32) ** 2).mean(axis=1)) if count \
            else np.zeros(0, np.float32)
        srt = np.sort(levels)
        k = int(count * 0.9)
        loud = float(srt[k]) if k < count else 0.0
        self.voiced = np.nonzero(levels >= loud * 0.12)[0].tolist() if loud > 0 else []

    def time(self, fraction):
        fraction = min(1.0, max(0.0, fraction))
        if len(self.voiced) < 5:
            return self.duration * fraction
        position = fraction * len(self.voiced)
        index = min(len(self.voiced) - 1, int(position))
        within = position - index
        return (self.voiced[index] + min(1.0, within)) * FRAME / 16000


def spread(text, samples):
    """[(start, end, unit text)] in seconds from the start of `samples`."""
    us = units(text)
    # Swift counts Characters (grapheme clusters): combining marks (Arabic harakat) do not add to the weight
    weights = [max(1, sum(1 for c in u.strip(" ") if not unicodedata.combining(c))) for u in us]
    total = max(1, sum(weights))
    clock = VoicedClock(samples)
    out, done = [], 0
    for u, w in zip(us, weights):
        start = clock.time(done / total)
        done += w
        out.append((start, max(start, clock.time(done / total)), u))
    return out
