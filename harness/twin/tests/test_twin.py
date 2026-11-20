"""Twin unit tests on synthetic inputs (no audio, no models)."""
import json
from pathlib import Path

import numpy as np
import pytest

from harness.twin import aligner, config
from harness.twin.diar import ActivityTracker, Observation, TurnSplitter, decision_log, observations
from harness.twin.router import Decoded, Router
from harness.twin.sim import CostModel, Twin

VOCALI_DEFAULT = Path.home() / "Desktop/ChatSign/VoiceLocater/config/harness/default.json"


def test_config_defaults_and_rejection():
    c = config.load({"router": {"margin": 2.0}})
    assert c["router"]["margin"] == 2.0 and c["activity"]["hold"] == 0.24
    with pytest.raises(config.ConfigError):
        config.load({"router": {"marginn": 1}})
    assert config.load(Path(__file__).parents[1] / "default.json") == config.DEFAULTS


@pytest.mark.skipif(not VOCALI_DEFAULT.exists(), reason="Vocali checkout not on this machine")
def test_default_file_equals_vocali():
    assert json.loads((Path(__file__).parents[1] / "default.json").read_text()) == json.loads(
        VOCALI_DEFAULT.read_text())


class Fake:
    """Qwen says P(ar) = p_ar for every span; Audar returns Arabic text."""

    def __init__(self, p_ar=0.001, text="hello world again"):
        self.p_ar, self.text, self.calls = p_ar, text, []

    def decode(self, engine, span):
        self.calls.append((engine, span))
        sec = (span[1] - span[0]) / 16000
        txt = "مرحبا بكم جميعا" if engine == "audar" else self.text
        words = txt.split()
        toks = [(sec * i / len(words), sec * (i + 1) / len(words), (w + " ").encode()) for i, w in enumerate(words)]
        lang = "ar" if engine == "audar" else ("ar" if self.p_ar > 0.5 else "en")
        return Decoded(txt, toks, lang, None if engine == "audar" else {"ar": self.p_ar, "en": 1 - self.p_ar})

    def offset(self, span):
        return span[0] / 16000


def test_router_switches_on_strong_arabic_evidence():
    b = Fake(p_ar=0.95)
    r = Router(config.DEFAULTS["router"], b)
    res, passes, tr = r.decode(1, False, None, (0, 16000))
    assert tr["switched"] and res.language == "ar" and [e for e, _ in passes] == ["qwen", "audar"]
    res, passes, _ = r.decode(1, True, None, (0, 32000))
    assert [e for e, _ in passes] == ["audar"]       # decided: Qwen is not run again


def test_router_never_concludes_not_arabic_early():
    r = Router(config.DEFAULTS["router"], Fake(p_ar=0.001))
    _, _, tr = r.decode(1, False, None, (0, 16000))
    assert not tr["decided"]
    _, _, tr = r.decode(1, False, None, (0, 16000 * 3))
    assert tr["decided"] and tr["engine"] == "qwen"


def test_tracker_hold_and_confirm():
    probs = np.zeros((40, 2))
    probs[:20, 0] = 0.9
    probs[20:, 1] = 0.9
    probs[20:, 0] = 0.6
    obs = observations(probs)
    tr = ActivityTracker()
    frames = [f for o in obs for f in tr.accept(o)]
    assert frames[10].dominant == 1
    first2 = next(i for i, f in enumerate(frames) if f.dominant == 2)
    # slot 2 must first be confirmed (0.6 s) and then dominate for hold (0.24 s)
    assert 20 + 7 <= first2 <= 20 + 7 + 4


def test_turn_splitter_cut_at_new_voice():
    probs = np.zeros((60, 2))
    probs[:30, 0] = 0.9
    probs[30:, 1] = 0.9
    sp = TurnSplitter(config.DEFAULTS["activity"])
    cuts = [c for o in observations(probs) for c in sp.accept(o)[2]]
    assert len(cuts) == 1 and cuts[0] == pytest.approx(30 * 0.08, abs=0.09)


def test_aligner_splits_and_smooths():
    dec = [(round(0.08 * (i + 1), 2), 1 if i < 25 else 2) for i in range(50)]
    toks = [(0.2 * i, 0.2 * i + 0.18, f"w{i} ".encode()) for i in range(20)]
    out = aligner.lines(0.0, 4.0, toks, dec)
    assert [l["speaker"] for l in out] == [1, 2]
    # a one-word flip in the middle is smoothed away
    dec2 = [(t, 2 if 1.55 < t < 1.75 else 1) for t, _ in dec]
    assert [l["speaker"] for l in aligner.lines(0.0, 4.0, toks, dec2)] == [1]


def _blocks(segments, seconds):
    n = int(seconds * 16000) // 512
    return [((i + 1) * 512, any(a <= (i + 1) * 512 / 16000 <= b for a, b in segments)) for i in range(n)]


def test_sim_schedule_partials_finals_and_cost():
    probs = np.zeros((int(10 / 0.08), 1))
    probs[10:90, 0] = 0.9
    obs = observations(probs)
    b = Fake()
    tw = Twin(config.DEFAULTS, b, CostModel())
    h = tw.run(_blocks([(1.0, 6.0)], 10.0), obs, 10.0)
    finals = [c for c in h["captions"] if c["final"]]
    partials = [c for c in h["captions"] if not c["final"]]
    assert len(finals) == 1 and len(partials) >= 3
    # partials no more often than the 0.9 s interval
    ts = [c["t"] for c in partials]
    assert all(b2 - a2 >= 0.9 - 1e-6 for a2, b2 in zip(ts, ts[1:]))
    u = h["utterances"]["1"]
    assert u[0] == pytest.approx(1.0 - 0.512 - 0.032, abs=0.04) and u[1] == pytest.approx(6.0, abs=0.07)
    assert h["lines"] and h["lines"][0]["speaker"] == "1" and h["language"] == "en"
    assert h["cost"]["decode_audio_seconds"] > 5.0


def test_sim_long_speech_is_cut_at_maximum_utterance():
    tw = Twin(config.DEFAULTS, Fake(), CostModel())
    h = tw.run(_blocks([(0.5, 30.0)], 31.0), [], 31.0)
    spans = sorted(h["utterances"].values())
    assert len(spans) == 3 and all(b - a <= 12.0 + 0.04 for a, b in spans)


def test_sim_slow_decoder_drops_partials_not_finals():
    slow = CostModel({"qwen": 2.0, "audar": 2.0}, {"qwen": 0.5, "audar": 0.5})
    tw = Twin(config.DEFAULTS, Fake(), slow)
    h = tw.run(_blocks([(0.5, 4.0), (5.0, 8.0), (9.0, 12.0)], 14.0), [], 14.0)
    assert len([c for c in h["captions"] if c["final"]]) == 3


def test_spread_units_and_voiced_clock():
    from harness.twin.spread import spread, units

    assert units("你好, world. ok") == ["你", "好,", " ", "world. ", "ok"]
    assert units("hello, world") == ["hello, ", "world"]
    x = np.zeros(16000 * 2, np.float32)
    x[16000:] = 0.5 * np.sin(np.arange(16000) / 5)        # silence, then voice
    toks = spread("one two", x)
    assert toks[0][0] >= 0.99 and toks[-1][1] == pytest.approx(2.0, abs=0.03)


def test_cache_roundtrip_through_twin(tmp_path=None):
    import tempfile

    from harness.twin.cache import DecodeCache, quantize

    d = Path(tempfile.mkdtemp())
    tw = Twin(config.DEFAULTS, DecodeCache(d, "s"), CostModel())
    tw.run(_blocks([(1.0, 3.0)], 4.0), [], 4.0)
    cache = tw.backend
    assert cache.write_misses() > 0
    with open(d / "s.jsonl", "w") as f:
        for line in open(d / "s.misses.jsonl"):
            m = json.loads(line)
            f.write(json.dumps({**m, "text": "hi there", "tokens": [[0.1, 0.4, "hi "], [0.4, 0.9, "there"]],
                                "language": "en", "probs": {"zh": 0.01, "en": 0.98, "ar": 0.01}}) + "\n")
    c2 = DecodeCache(d, "s")
    h = Twin(config.DEFAULTS, c2, CostModel()).run(_blocks([(1.0, 3.0)], 4.0), [], 4.0)
    assert not c2.misses and h["captions"][-1]["text"] == "hi there"
    assert quantize((1601, 3250)) == (1, 2)


def test_session_language_locks_arabic_and_skips_qwen():
    from harness.twin.candidates.session_language import SessionLanguageRouter

    b = Fake(p_ar=0.97)
    r = SessionLanguageRouter(config.DEFAULTS["router"], b, params={"lockAt": 4.0, "probeEvery": 100})
    for u in range(1, 4):
        r.decode(u, False, None, (0, 16000))
        r.decode(u, True, None, (0, 32000))
    assert r.locked == "ar"
    b.calls.clear()
    res, passes, tr = r.decode(9, False, None, (0, 16000))
    assert [e for e, _ in passes] == ["audar"] and res.language == "ar" and tr["session"] == "locked-ar"


def test_session_language_does_not_lock_english_to_arabic():
    from harness.twin.candidates.session_language import SessionLanguageRouter

    r = SessionLanguageRouter(config.DEFAULTS["router"], Fake(p_ar=0.001))
    for u in range(1, 6):
        r.decode(u, False, None, (0, 16000 * 4))
        r.decode(u, True, None, (0, 16000 * 4))
    assert r.locked == "other" and r.L < 0
    _, passes, _ = r.decode(9, True, None, (0, 16000))
    assert [e for e, _ in passes] == ["qwen"]
