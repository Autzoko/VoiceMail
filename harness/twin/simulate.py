"""Run the twin on the sessions of one split and write hypothesis files for harness/eval.

    python -m harness.twin.simulate --manifest data/datasets/manifests/notsofar__sdm.jsonl --split search \
        --config harness/twin/default.json --out data/outputs/harness/runs/<name>

Per session it needs (all on JUBAIL, under data/outputs/harness/):
  vad/<settings>/<session>.npy                  speech flag per 512-sample block (made here on first use; Silero via sherpa-onnx)
  diar/nemotron3_low/<session>.npy    Nemotron frame probabilities (harness/diar, preset low); absent = no speakers
  cache/<session>.jsonl               decodes (harness/twin/fill.py); spans not cached are listed in
                                      cache/<session>.misses.jsonl and the run is reported incomplete
Exit status 3 when any decode was missing (fill, then run again: the spans only change where decodes change the
routing, so this converges in a few rounds).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from . import config
from .cache import DecodeCache
from .diar import observations
from .frontend import BLOCK, vad_blocks
from .sim import CostModel, Twin

ROOT = Path(__file__).resolve().parents[2]
H = ROOT / "data/outputs/harness"
VAD_MODEL = ROOT / "data/checkpoints/silero_vad/silero_vad.onnx"


def sessions_of(manifests, split, splits_file):
    from harness.eval.score import split_of

    splits = json.loads(Path(splits_file).read_text())["splits"] if split != "all" else None
    out = {}
    for m in manifests:
        for line in open(m, encoding="utf-8"):
            r = json.loads(line)
            s = r["session"]
            if s in out:
                continue
            if splits is None or split_of(s, splits) == split:
                out[s] = {"audio": r["audio"], "channel": r.get("channel", 0)}
    return out


def blocks_for(session, entry, vad):
    key = hashlib.sha1(json.dumps(vad, sort_keys=True).encode()).hexdigest()[:8]  # VAD output depends on its settings
    p = H / "vad" / key / f"{session}.npy"
    if p.exists():
        flags = np.load(p)
    else:
        from .fill import load_audio

        flags = np.array([s for _, s in vad_blocks(load_audio(entry), str(VAD_MODEL), vad)], bool)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, flags)
    return [((i + 1) * BLOCK, bool(f)) for i, f in enumerate(flags)], len(flags) * BLOCK / 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--split", required=True, choices=["search", "select", "test", "all"])
    ap.add_argument("--splits", default=str(ROOT / "harness/splits.json"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--cost", default=None, help="cost model JSON (macmini measurement)")
    ap.add_argument("--diar", default=str(H / "diar/nemotron3_low"))
    ap.add_argument("--cache", default=str(H / "cache"))
    ap.add_argument("--context", default=None)
    ap.add_argument("--forced-language", default=None, choices=["zh", "en", "ar"])
    ap.add_argument("--candidate", default=None, choices=["session_language"],
                    help="twin-only L1 candidate router (harness/twin/candidates)")
    ap.add_argument("--candidate-params", default=None, help="JSON object of the candidate's parameters")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg = config.load(a.config)
    cost = CostModel.load(a.cost) if a.cost else CostModel()
    sess = sessions_of(a.manifest, a.split, a.splits)
    if a.limit:
        sess = dict(sorted(sess.items())[:a.limit])
    index_path = H / "audio_index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index.update(sess)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=0, sort_keys=True))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "_config.json").write_text(config.dump(cfg) + "\n")
    (out / "_candidate.json").write_text(json.dumps({"candidate": a.candidate, "params": a.candidate_params}) + "\n")
    missing = 0
    for session, entry in sorted(sess.items()):
        blocks, duration = blocks_for(session, entry, cfg["vad"])
        dp = Path(a.diar) / f"{session}.npy"
        obs = observations(np.load(dp).astype(np.float32)) if dp.exists() else []
        cache = DecodeCache(a.cache, session, a.context)
        router = None
        if a.candidate == "session_language":
            from .candidates.session_language import SessionLanguageRouter

            router = SessionLanguageRouter(cfg["router"], cache, a.forced_language,
                                           json.loads(a.candidate_params) if a.candidate_params else None)
        hyp = Twin(cfg, cache, cost, a.forced_language, router).run(blocks, obs, duration)
        n = cache.write_misses()
        missing += n
        hyp["session"] = session
        hyp["incomplete"] = n
        (out / f"{session}.json").write_text(json.dumps(hyp, ensure_ascii=False))
        print(f"{session}: {len(hyp['lines'])} lines, {len(hyp['captions'])} passes, {n} decodes missing"
              f"{'' if dp.exists() else ', NO DIARIZATION'}", flush=True)
    print(f"missing decodes: {missing}")
    sys.exit(3 if missing else 0)


if __name__ == "__main__":
    main()
