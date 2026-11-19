"""Overlap study, LibriCSS (real re-recorded overlap at controlled ratios 0S/0L/OV10/OV20/OV30/OV40).

LibriCSS = LibriSpeech test-clean utterances played by loudspeakers in a real meeting room and recorded by a
7-ch array; the overlap ratio of each 10-min mini-session is set by design. Conditions (oracle utterance
segmentation from meeting_info.txt, same as all other sets):
    libricss__sdm : channel 0 of the array recording (far field, real room)
    libricss__ihm : the clean source of the speaker (each_spk.wav, lhotse channel map) -- word clock + reference

    python scripts/overlap/libricss.py prepare            # manifests (lhotse recipe parsing)
    python scripts/overlap/libricss.py analyze --workers 32   # features + attribution, dataset = libricss_<OV>
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "overlap"), str(ROOT / "scripts" / "analysis"),
                str(ROOT / "scripts" / "data")]
DS = ROOT / "data" / "datasets" / "libricss"


def prepare():
    from lhotse.recipes.libricss import prepare_libricss
    from prepare_manifests import check_audio, from_lhotse, keep
    from src.utils.manifest import write_manifest

    corpus = next(p for p in [DS / "for_release", DS] if (p / "OV10").is_dir())
    for typ in ("sdm", "ihm"):
        m = prepare_libricss(corpus, output_dir=None, type=typ)
        rows = [r for r in from_lhotse(m["recordings"], m["supervisions"], "en") if keep(r)]
        check_audio(rows, f"libricss/{typ}")
        write_manifest(rows, ROOT / "data/datasets/manifests" / f"libricss__{typ}.jsonl")
        print(f"libricss {typ}: {len(rows)} segments, {sum(r['end'] - r['start'] for r in rows) / 3600:.2f} h")


def analyze(models, workers):
    import pandas as pd

    import build_utt_table as bt
    import overlap_table as ot

    bt._wada_table()
    for cond in ("ihm", "sdm"):
        bt.build_features(f"libricss__{cond}", workers)
    ot.PAIRS["libricss"] = ("ihm", "sdm")
    for cond in ("ihm", "sdm"):
        ot.build("libricss", cond, models, workers)
        for model in models:
            p = ot.AN / "overlap" / model / f"libricss__{cond}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                df["dataset"] = "libricss_" + df.session.str.split("_").str[0]
                df.to_parquet(p, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "analyze"])
    ap.add_argument("--models", nargs="+", default=["Qwen3-ASR-1.7B", "Qwen3-ASR-0.6B"])
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    prepare() if args.step == "prepare" else analyze(args.models, args.workers)


if __name__ == "__main__":
    main()
