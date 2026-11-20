"""Nemotron-3-Diarization (NeMo, Vocali's streaming preset) over every session of the given manifests / splits:
the speaker-probability input of the twin (harness/twin/diar.py).

    python harness/diar/nemotron_run.py --manifest m1.jsonl m2.jsonl --split search select [--shard 0/4]

Per session (channel `channel` of the manifest audio = the single far-field mic):
  data/outputs/harness/diar/nemotron3_<preset>/<session>.npy   frame probabilities (float16, T x speakers)
                                                               <session>.rttm, <session>.json (DER, RTF, counts)
Sessions whose .npy exists are skipped (resumable; parallel shards own disjoint sessions).
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nemotron_smoke import PRESETS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def split_of(session, splits):  # same rule as harness/eval/score.split_of (not imported: NeMo env has no meeteval)
    for units in splits.values():
        if session in units:
            return units[session]
    return splits.get("notsofar", {}).get(session.split("_sc_")[0])


def sessions(manifests, wanted, splits_file):
    splits = json.loads(Path(splits_file).read_text())["splits"]
    by = defaultdict(list)
    for m in manifests:
        for line in open(m, encoding="utf-8"):
            r = json.loads(line)
            if "all" in wanted or split_of(r["session"], splits) in wanted:
                by[r["session"]].append(r)
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--split", nargs="+", default=["search", "select"])
    ap.add_argument("--splits", default=str(ROOT / "harness/splits.json"))
    ap.add_argument("--preset", default="low", choices=PRESETS)
    ap.add_argument("--shard", default=None, help="I/N")
    a = ap.parse_args()
    from nemo.collections.asr.models import SortformerEncLabelModel
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    out = ROOT / "data/outputs/harness/diar" / f"nemotron3_{a.preset}"
    out.mkdir(parents=True, exist_ok=True)
    by = sessions(a.manifest, a.split, a.splits)
    names = sorted(by)
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        names = names[i::n]
    todo = [s for s in names if not (out / f"{s}.npy").exists()]
    print(f"{len(names)} sessions, {len(todo)} to do", flush=True)
    if not todo:
        return
    model = SortformerEncLabelModel.from_pretrained("nvidia/Nemotron-3-Diarization", map_location="cuda")
    model.eval()
    for k, v in PRESETS[a.preset].items():
        setattr(model.sortformer_modules, k, v)
    model.sortformer_modules._check_streaming_parameters()
    tmp = Path(f"/tmp/nemotron_{os.getpid()}")  # channel-0 wav handed to model.diarize
    tmp.mkdir(parents=True, exist_ok=True)
    for name in todo:
        rows = by[name]
        audio, sr = sf.read(rows[0]["audio"], dtype="float32", always_2d=True)
        assert sr == 16000, (name, sr)
        wav = tmp / f"{name}.wav"
        sf.write(wav, audio[:, rows[0].get("channel", 0)], sr)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.inference_mode():
            segments, probs = model.diarize(audio=[str(wav)], batch_size=1, include_tensor_outputs=True)
        torch.cuda.synchronize(); elapsed = time.time() - t0
        wav.unlink()
        p = probs[0].float().cpu().numpy() if hasattr(probs[0], "cpu") else np.asarray(probs[0])
        p = p.reshape(p.shape[-2], p.shape[-1]) if p.ndim == 3 else p
        hyp, ref = Annotation(), Annotation()
        with open(out / f"{name}.rttm", "w") as f:
            for seg in segments[0]:
                s, e, spk = str(seg).split()[:3]
                hyp[Segment(float(s), float(e))] = spk
                f.write(f"SPEAKER {name} 1 {float(s):.3f} {float(e) - float(s):.3f} <NA> <NA> {spk} <NA> <NA>\n")
        for r in rows:
            if r["end"] > r["start"]:
                ref[Segment(r["start"], r["end"])] = r["speaker"]
        dur = len(audio) / sr
        info = dict(session=name, preset=a.preset, seconds=round(dur, 2), rtf=round(elapsed / dur, 4),
                    frames=list(p.shape), frame_s=round(dur / max(p.shape[0], 1), 4),
                    ref_speakers=len(ref.labels()), hyp_speakers=len(hyp.labels()),
                    der_c0=round(DiarizationErrorRate(collar=0.0)(ref, hyp), 4),
                    der_c025=round(DiarizationErrorRate(collar=0.25)(ref, hyp), 4))
        (out / f"{name}.json").write_text(json.dumps(info))
        np.save(out / f"{name}.npy", p.astype(np.float16))  # last: its presence marks the session done
        print(info, flush=True)


if __name__ == "__main__":
    main()
