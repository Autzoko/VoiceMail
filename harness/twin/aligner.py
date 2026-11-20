"""TranscriptAligner (Sources/SpeakerCore/Transcript.swift): split an utterance's tokens between speakers using
the per-frame decision log, then smooth short runs. Tokens: list of (start, end, bytes) on the session clock.
"""


def lines(t_start, t_end, tokens, decisions, window=0.16):
    relevant = [d for d in decisions if t_start - 1.1 <= d[0] <= t_end + window + 1.1]

    def vote(a, b):
        counts = {}
        for time, spk in relevant:
            if time >= a and time - window <= b and spk is not None:
                counts[spk] = counts.get(spk, 0) + 1
        if not counts:
            return None
        best = max(counts.values())
        return min(k for k, v in counts.items() if v == best)  # ties: the smaller speaker id

    def nearest(m):
        best, dist = None, None
        for time, spk in relevant:
            if spk is None:
                continue
            d = abs(time - window / 2 - m)
            if d <= 1.0 and (dist is None or d < dist):  # first of equal minima, as Swift min(by:)
                best, dist = spk, d
        return best

    fallback = vote(t_start, t_end)
    reach = 0.12 if window < 0.5 else 0.0
    speakers = []
    for start, end, _ in tokens:
        mid = (start + end) / 2
        previous = speakers[-1] if speakers else None
        s = vote(mid - reach, mid + reach)
        if s is None:
            s = nearest(mid)
        if s is None:
            s = previous
        if s is None:
            s = fallback
        speakers.append(s)
    smooth(speakers, tokens)
    out, buf, cur = [], b"", None
    for (start, end, raw), spk in zip(tokens, speakers):
        continuation = bool(raw) and (raw[0] & 0xC0) == 0x80
        if cur is not None and cur[0] != spk and not continuation:
            _flush(out, cur, buf)
            buf, cur = b"", None
        if cur is None:
            cur = [spk, start, end]
        cur[2] = end
        buf += raw
    if cur is not None:
        _flush(out, cur, buf)
    return out


def _flush(out, cur, buf):
    text = buf.decode("utf-8", "ignore").strip()
    if text:
        out.append({"speaker": cur[0], "start": cur[1], "end": cur[2], "text": text})


def smooth(speakers, tokens):
    if len(speakers) < 2:
        return
    changed, passes = True, 0
    while changed and passes < len(speakers):
        changed = False
        passes += 1
        runs, i = [], 0
        while i < len(speakers):
            j = i
            while j + 1 < len(speakers) and speakers[j + 1] == speakers[i]:
                j += 1
            runs.append((i, j))
            i = j + 1
        if len(runs) <= 1:
            return
        for r, (a, b) in enumerate(runs):
            words = b - a + 1
            seconds = tokens[b][1] - tokens[a][0]
            if not (words < 3 and seconds < 0.6):
                continue
            before = speakers[runs[r - 1][1]] if r > 0 else None
            after = speakers[runs[r + 1][0]] if r + 1 < len(runs) else None
            if r == 0:
                target, ok = after, True
            elif r == len(runs) - 1:
                target, ok = before, True
            elif before == after:
                target, ok = before, True
            else:
                ok = False
            if not ok or target == speakers[a]:
                continue
            for k in range(a, b + 1):
                speakers[k] = target
            changed = True
            break
