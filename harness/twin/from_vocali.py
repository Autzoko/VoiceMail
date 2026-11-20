"""Vocali replay NDJSON -> evaluator hypothesis (harness/eval/schema.py), for macmini runs and twin calibration.

    voice-locater stream --json ... --asr auto-qwen-audar --input-file SESSION.wav > SESSION.ndjson   (macmini)
    python -m harness.twin.from_vocali SESSION.ndjson --session SESSION --out RUN_DIR

Times: every line carries "_t" (wall clock); transcript and activity times are on the same clock at speed 1.
Both are made session-relative by subtracting the replay start. Speaker lines are made the way the app makes them
(decision log + TranscriptAligner, the twin's ports), from the activity lines in the file.
Cost: audio seconds over all decode passes, from the route traces (a switch to Audar decodes the audio twice).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from . import aligner, config
from .config import ENGINE
from .diar import Observation, decision_log


def convert(path, session, cfg=None):
    cfg = cfg or config.DEFAULTS
    start, info, transcripts, obs, routes = None, {}, [], [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = item.get("_t")
        if "replay" in item and item["replay"].get("event") == "start":
            start, info = t, item["replay"]
        elif "transcript" in item:
            transcripts.append((t, item["transcript"]))
        elif "activity" in item:
            a = item["activity"]
            obs.append((t, a))
        elif "route" in item:
            routes.append(item["route"])
    if start is None:
        raise SystemExit(f"{path}: no replay start line")
    if float(info.get("speed", 1) or 1) != 1:
        raise SystemExit(f"{path}: replay speed must be 1 for timing metrics")
    observations = [Observation(a["start"] - start, a["step"], np.asarray(a["probs"], np.uint8), t - start)
                    for t, a in obs]
    decisions = decision_log(sorted(observations, key=lambda o: o.start), cfg["activity"])
    captions, finals, utterances = [], {}, {}
    for t, tr in transcripts:
        toks = [(k["start"] - start, k["end"] - start, bytes(k.get("bytes", []))) for k in tr.get("tokens", [])]
        text = b"".join(r for _, _, r in toks).decode("utf-8", "ignore").strip()
        captions.append({"t": t - start, "utterance": tr["utterance"], "text": text, "final": tr.get("final", False)})
        if tr.get("final"):
            finals[tr["utterance"]] = (tr, toks)
            utterances[str(tr["utterance"])] = [tr["start"] - start, tr["end"] - start]
    engines = {}
    for r in routes:
        if r.get("final"):
            engines[r["utterance"]] = r["engine"]
    lines = []
    for u, (tr, toks) in sorted(finals.items()):
        a, b = tr["start"] - start, tr["end"] - start
        visible = [d for d in decisions if d[0] <= b + 4.0]
        for ln in aligner.lines(a, b, toks, visible, ENGINE["alignerWindow"]):
            lines.append({**ln, "speaker": None if ln["speaker"] is None else str(ln["speaker"]), "utterance": u,
                          "engine": engines.get(u)})
    decoded = sum(r["seconds"] * (2 if r.get("switched") else 1) for r in routes)
    langs = [tr["language"] for tr, _ in finals.values()]
    return {"session": session, "duration": float(info.get("audioSeconds", 0) or 0),
            "language": max(set(langs), key=langs.count) if langs else None,
            "lines": lines, "captions": captions, "utterances": utterances, "translations": [],
            "cost": {"decode_audio_seconds": decoded, "decode_calls": len(routes)}, "routes": routes,
            "source": "vocali"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ndjson")
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    h = convert(a.ndjson, a.session)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"{a.session}.json").write_text(json.dumps(h, ensure_ascii=False))
    print(f"{a.session}: {len(h['lines'])} lines, {len(h['captions'])} passes, {len(h['routes'])} route steps")


if __name__ == "__main__":
    main()
