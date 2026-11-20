"""Nemotron-3-Diarization in PyTorch/NeMo with Vocali's streaming preset, on LibriCSS far-field sessions.

Checks that the diarizer the app runs on Core ML (FluidAudio, preset `low`) can be replicated on JUBAIL for the
offline harness (docs/harness-rsi.md §3.2): loads the official checkpoint, applies the preset, diarizes N sessions
(channel 0 of the array = single far-field mic), saves per-frame speaker probabilities (the input of the app's
ActivityTracker / TurnSplitter) and RTTMs, and scores DER against the LibriCSS reference segments.

    python harness/diar/nemotron_smoke.py --sessions 4 [--preset low]
Outputs: data/outputs/harness/diar/nemotron3_<preset>/<session>.{npy,rttm}, summary.csv
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
# NeMo SortformerModules values; (chunk_len + chunk_right_context) x 80 ms = latency. `low` is what Vocali uses
# (VoiceLocater NemotronStream.config = Nemotron3Config.low; FluidAudio Nemotron3Types.swift).
PRESETS = {
    "offline": dict(chunk_len=340, chunk_right_context=40, fifo_len=40, spkcache_len=264, spkcache_update_period=300),
    "low": dict(chunk_len=9, chunk_right_context=4, fifo_len=264, spkcache_len=264, spkcache_update_period=222),
    "verylow": dict(chunk_len=6, chunk_right_context=2, fifo_len=264, spkcache_len=264, spkcache_update_period=222),
    "ultra": dict(chunk_len=3, chunk_right_context=1, fifo_len=264, spkcache_len=264, spkcache_update_period=222),
}


def sessions(n):
    rows = [json.loads(line) for line in open(ROOT / "data/datasets/manifests/libricss__sdm.jsonl")]
    by = defaultdict(list)
    for r in rows:
        by[r["session"]].append(r)
    # one session per overlap condition first (0S, 0L, OV10 … OV40), then the rest
    names = sorted(by)
    first = {}
    for s in names:
        first.setdefault(s.split("_")[0], s)
    order = list(first.values()) + [s for s in names if s not in first.values()]
    return [(s, by[s]) for s in order[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--preset", default="low", choices=PRESETS)
    args = ap.parse_args()
    from nemo.collections.asr.models import SortformerEncLabelModel
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    out = ROOT / "data/outputs/harness/diar" / f"nemotron3_{args.preset}"
    out.mkdir(parents=True, exist_ok=True)
    model = SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization", map_location="cuda")
    model.eval()
    for k, v in PRESETS[args.preset].items():
        setattr(model.sortformer_modules, k, v)
    model.sortformer_modules._check_streaming_parameters()
    print("preset", args.preset, PRESETS[args.preset], "left_context", getattr(model.sortformer_modules, "chunk_left_context", None))

    lines = []
    for name, rows in sessions(args.sessions):
        audio, sr = sf.read(rows[0]["audio"], dtype="float32", always_2d=True)
        wav = out / f"{name}.wav"           # channel 0, 16 kHz mono (temporary input)
        sf.write(wav, audio[:, rows[0]["channel"]], sr)
        torch.cuda.synchronize(); t0 = time.time()
        segments, probs = model.diarize(audio=[str(wav)], batch_size=1, include_tensor_outputs=True)
        torch.cuda.synchronize(); elapsed = time.time() - t0
        wav.unlink()
        p = probs[0].float().cpu().numpy() if hasattr(probs[0], "cpu") else np.asarray(probs[0])
        np.save(out / f"{name}.npy", p.astype(np.float16))
        hyp, ref = Annotation(), Annotation()
        with open(out / f"{name}.rttm", "w") as f:
            for seg in segments[0]:
                s, e, spk = str(seg).split()[:3]
                hyp[Segment(float(s), float(e))] = spk
                f.write(f"SPEAKER {name} 1 {float(s):.3f} {float(e) - float(s):.3f} <NA> <NA> {spk} <NA> <NA>\n")
        for r in rows:
            ref[Segment(r["start"], r["end"])] = r["speaker"]
        der0 = DiarizationErrorRate(collar=0.0)(ref, hyp)
        der25 = DiarizationErrorRate(collar=0.25)(ref, hyp)
        dur = len(audio) / sr
        line = dict(session=name, preset=args.preset, hours=dur / 3600, rtf=elapsed / dur, frames=p.shape,
                    ref_speakers=len(ref.labels()), hyp_speakers=len(hyp.labels()), der_c0=der0, der_c025=der25)
        print(line, flush=True)
        lines.append(line)
    import pandas as pd
    pd.DataFrame(lines).to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
