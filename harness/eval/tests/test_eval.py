"""Evaluator self-tests on synthetic sessions (no data). Run on a compute node:
    CMD="python -m pytest -q harness/eval/tests" sbatch harness/cpu.sbatch
"""
import json

import pytest

from harness.eval import gate, lock, score
from harness.eval.metrics import session_counts, value
from harness.eval.normalize import normalize_ar


def ref(lang="en", texts=(("A", 0, 2, "hello there friend"), ("B", 3, 5, "good morning to you"))):
    return {"lang": lang, "segments": [{"speaker": s, "start": a, "end": b, "text": t} for s, a, b, t in texts]}


def hyp(lines, captions=None, utterances=None, duration=6.0, cost=1.0):
    return {"duration": duration, "lines": [{"speaker": s, "start": a, "end": b, "text": t} for s, a, b, t in lines],
            "captions": captions or [], "utterances": utterances or {}, "translations": [],
            "cost": {"decode_audio_seconds": cost * duration}}


def test_arabic_n1_matches_benchmark():
    assert normalize_ar("و قال أحمد، مرحباً") == "وقال احمد مرحبا"
    assert normalize_ar("الـــكتاب ٣") == "الكتاب 3"


def test_perfect_and_swapped_speakers():
    good = session_counts("s", ref(), hyp([("1", 0, 2, "hello there friend"), ("2", 3, 5, "good morning to you")]))
    assert good["orc_err"] == 0 and good["cp_err"] == 0 and good["spk_exact"] == 1
    merged = session_counts("s", ref(), hyp([("1", 0, 5, "hello there friend good morning to you")]))
    assert merged["orc_err"] == 0 and merged["cp_err"] > 0 and merged["dcp_err"] == merged["cp_err"]
    assert merged["spk_abs_err"] == 1


def test_missing_output_is_total_deletion():
    r = session_counts("s", ref(), None)
    assert r["orc_del"] == r["orc_len"] == 7 and value([r], "del") == 100


def test_foreign_script_and_silence():
    r = session_counts("s", ref(), hyp([("1", 0, 2, "hello there friend"), ("2", 3, 5, "صباح الخير"),
                                        ("2", 9, 10, "thank you")], duration=12))
    assert r["foreign_units"] == 2 and r["hyp_units"] == 7
    assert r["silence_units"] == 2


def test_chinese_scored_per_character():
    rz = ref("zh", (("A", 0, 2, "你好，世界"),))
    r = session_counts("s", rz, hyp([("1", 0, 2, "你好世")]))
    assert r["orc_len"] == 4 and r["orc_del"] == 1


def test_arabic_also_per_character():
    ra = ref("ar", (("A", 0, 2, "مرحبا بكم"),))
    r = session_counts("s", ra, hyp([("1", 0, 2, "مرحبا بكن")]))
    assert r["orc_err"] == 1 and r["orcc_err"] == 1 and r["orcc_len"] == 8


def test_ux_erasure_and_lags():
    caps = [{"t": 1.0, "utterance": 0, "text": "hello", "final": False},
            {"t": 1.6, "utterance": 0, "text": "hello there", "final": False},
            {"t": 2.4, "utterance": 0, "text": "yellow there friend", "final": True}]
    r = session_counts("s", ref(), hyp([("1", 0, 2, "hello there friend")], caps, {"0": [0.2, 2.0]}))
    assert r["erased"] == 2 and r["final_units"] == 3
    assert r["first_lag_sum"] == pytest.approx(0.8) and r["final_lag_sum"] == pytest.approx(0.4)


def _sessions(n, err, lag):
    rows = {}
    for i in range(n):
        rows[f"s{i}"] = {"session": f"s{i}", "lang": "en" if i % 2 else "ar", "orc_err": err, "orc_len": 100,
                         "cp_err": err + 2, "cp_len": 100, "first_lag_sum": lag, "first_lag_n": 1,
                         "orc_del": 1, "orc_ins": 1}
    return rows


OFF = {"official": True, "evaluator": "x"}


def test_gate_identical_is_not_promoted_and_not_regressed():
    b = _sessions(20, 10, 1.0)
    res = gate.decide(b, b, OFF, OFF)
    assert not res["promote"] and not res["regressions"]


def test_gate_promotes_a_real_improvement():
    b = _sessions(20, 10, 1.0)
    c = {k: {**v, "orc_err": 8 + (i % 3), "cp_err": 10 + (i % 3)} for i, (k, v) in enumerate(b.items())}
    assert gate.decide(b, c, OFF, OFF)["promote"]


def test_gate_rejects_slower_captions_even_when_more_accurate():
    b = _sessions(20, 10, 1.0)
    c = {k: {**v, "orc_err": 8, "cp_err": 10, "first_lag_sum": 1.1} for k, v in b.items()}
    res = gate.decide(b, c, OFF, OFF)
    assert not res["promote"] and {r["metric"] for r in res["regressions"]} == {"first_lag"}


def test_gate_requires_official_scores():
    b = _sessions(20, 10, 1.0)
    c = {k: {**v, "orc_err": 8, "cp_err": 10} for k, v in b.items()}
    assert not gate.decide(b, c, OFF, {"official": False, "evaluator": "x"})["promote"]


def test_score_cli_end_to_end(tmp_path):
    man = tmp_path / "m.jsonl"
    man.write_text("".join(json.dumps({"session": "OV40_x", "speaker": s, "start": a, "end": b, "text": t,
                                       "lang": "en"}) + "\n"
                           for s, a, b, t in (("A", 0, 2, "hello there friend"), ("B", 3, 5, "good morning"))))
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"splits": {"libricss": {"OV40_x": "select"}}}))
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "OV40_x.json").write_text(json.dumps({"session": "OV40_x", **hyp(
        [("1", 0, 2, "hello there friend"), ("2", 3, 5, "good morning")])}))
    score.main(["--hyp", str(tmp_path / "run"), "--ref", str(man), "--split", "select", "--splits", str(splits),
                "--out", str(tmp_path / "out")])
    s = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert s["metrics"]["all"]["orc"] == 0 and s["missing_hypotheses"] == []


def test_lock_is_current():
    assert lock.check(), "evaluator changed without `python -m harness.eval.lock --write`"
