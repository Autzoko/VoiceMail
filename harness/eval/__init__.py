"""Locked evaluator of the Vocali harness (design doc §4; reports/Vocali Harness RSI 设计方案.md).

Proposers (Optuna, coding agents) must never edit this package: `lock.py` records a SHA-256 of every file here
in harness/eval/LOCK, and `score.py` refuses to produce official numbers when the files and LOCK disagree.
Changing the evaluator is a human decision: edit, run the tests, then `python -m harness.eval.lock --write`.

Inputs (one JSON file per session, the same from the Python twin and from a Vocali replay; see schema.py):
    hypothesis  speaker-attributed final lines, the caption event log, translations and decode cost
    reference   the shared segment manifests (src/utils/manifest.py), plus an optional `translation` field
"""
