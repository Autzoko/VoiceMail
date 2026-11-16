"""Build unified segment manifests (data/datasets/manifests/<dataset>__<condition>.jsonl) for the
official test/eval splits. See src/utils/manifest.py for the row schema.

Conditions (oracle reference segmentation everywhere):
  ami        ihm  : individual headset of the active speaker              (test, full-corpus-asr)
             sdm  : Array1-01, first mic of array 1
  alimeeting near : headset of the active speaker                         (Test)
             far  : channel 0 of the 8-ch table-top array
  notsofar   ihm  : close-talk mic of the active speaker                   (eval 240825.1 full)
             sdm  : every single-channel (SC) device of the meeting, each scored (NOTSOFAR SC track devices)
  chime6     ihm  : left channel of the speaker's binaural worn mic        (eval S01, S21)
             sdm  : CH1 of the per-utterance reference array ("ref" field; CHiME-6 track-1 reference)
  dipco      ihm  : speaker's close-talk mic                              (eval S01 S03 S06 S07 S08)
             sdm  : U01.CH1 (first channel of far-field array 1), array-specific timestamps

    python scripts/data/prepare_manifests.py [--datasets ami alimeeting notsofar chime6 dipco]
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.manifest import write_manifest  # noqa: E402

DS = ROOT / "data" / "datasets"
OUT = DS / "manifests"


def hms(t):
    """'0:01:02.34' / '00:01:02.34' / float -> seconds."""
    if isinstance(t, (int, float)):
        return float(t)
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def row(id_, session, speaker, audio, channel, start, end, text, lang="en"):
    return {"id": id_, "session": session, "speaker": speaker, "audio": str(audio), "channel": int(channel),
            "start": round(float(start), 3), "end": round(float(end), 3), "text": text.strip(), "lang": lang}


def keep(r):
    return r["end"] - r["start"] > 0.05 and r["text"]


def from_lhotse(recordings, supervisions, lang, id_prefix=""):
    """Convert lhotse manifests to rows; resolve supervision channel -> (file, channel-in-file)."""
    recs = {r.id: r for r in recordings}
    rows = []
    for s in supervisions:
        rec = recs[s.recording_id]
        ch = s.channel[0] if isinstance(s.channel, (list, tuple)) else s.channel
        src = next(x for x in rec.sources if ch in x.channels)
        rows.append(row(id_prefix + s.id, rec.id, s.speaker, src.source, src.channels.index(ch),
                        s.start, s.end, s.text or "", lang))
    return rows


# ------------------------------------------------------------------------------------------------ AMI
def prep_ami():
    import lhotse.recipes.ami as ami_recipe

    # only test audio is downloaded -> train/dev parts are empty and lhotse's fix_manifests asserts on
    # empty sets; skip the fix for empty parts (the test part still goes through the normal QA path).
    _fix = ami_recipe.fix_manifests
    ami_recipe.fix_manifests = lambda r, s: (r, s) if len(r) == 0 else _fix(r, s)
    prepare_ami = ami_recipe.prepare_ami

    root = DS / "ami"
    out = {}
    for mic in ["ihm", "sdm"]:
        m = prepare_ami(root / "wav_db", annotations_dir=root / "ami_public_manual_1.6.2.zip", output_dir=None,
                        mic=mic, partition="full-corpus-asr", normalize_text="none")
        m = m["test"]
        out[mic] = from_lhotse(m["recordings"], m["supervisions"], "en")
    return out


# ------------------------------------------------------------------------------------------ AliMeeting
def prep_alimeeting():
    """Parse M2MeT Test TextGrids directly (lhotse's recipe requires the Train split to be present).
    far : <room>_<meeting>.TextGrid, one tier per speaker  -> <room>_<meeting>_MS*.wav (8 ch), channel 0
    near: <room>_<meeting>_N_SPKxxxx.TextGrid, single tier -> same-stem headset wav
    """
    import textgrid

    root = DS / "alimeeting" / "Test_Ali"
    out = {"near": [], "far": []}
    for cond in ["near", "far"]:
        d = root / f"Test_Ali_{cond}"
        for tgf in sorted((d / "textgrid_dir").glob("*.TextGrid")):
            stem = tgf.stem
            if cond == "far":
                wavs = sorted((d / "audio_dir").glob(f"{stem}_MS*.wav"))
                assert len(wavs) == 1, (tgf, wavs)
                wav, session = wavs[0], stem
            else:
                wav, session = d / "audio_dir" / f"{stem}.wav", stem.split("_N_")[0]
            tg = textgrid.TextGrid.fromFile(str(tgf))
            for tier in tg.tiers:
                spk = tier.name if cond == "far" else stem.split("_N_")[1]
                for k, iv in enumerate(tier.intervals):
                    if iv.mark and iv.mark.strip():
                        out[cond].append(row(f"{session}_{spk}_{k:05d}", session, spk, wav, 0,
                                             iv.minTime, iv.maxTime, iv.mark, "zh"))
    return out


# ------------------------------------------------------------------------------------------- NOTSOFAR
def prep_notsofar():
    mtg_root = DS / "notsofar" / "benchmark-datasets" / "eval_set" / "240825.1_eval_full_with_GT" / "MTG"
    ihm, sdm = [], []
    for mdir in sorted(mtg_root.glob("MTG_*")):
        gt = json.load(open(mdir / "gt_transcription.json"))
        sc_devs = sorted(p.name for p in mdir.glob("sc_*") if (p / "ch0.wav").exists())
        for i, u in enumerate(gt):
            uid = f"{mdir.name}_{i:04d}_{u['speaker_id']}"
            ct = mdir / u["ct_wav_file_name"]
            if ct.exists():
                ihm.append(row(uid, mdir.name, u["speaker_id"], ct, 0, u["start_time"], u["end_time"], u["text"]))
            for d in sc_devs:
                sdm.append(row(f"{uid}_{d}", f"{mdir.name}_{d}", u["speaker_id"], mdir / d / "ch0.wav", 0,
                               u["start_time"], u["end_time"], u["text"]))
    return {"ihm": ihm, "sdm": sdm}


# --------------------------------------------------------------------------------------------- CHiME-6
def prep_chime6():
    root = DS / "chime6"
    # layout after untar: CHiME6_eval/CHiME6/audio/eval/*.wav, transcriptions/transcriptions/eval/S*.json
    tdir = next(p for p in root.glob("**/transcriptions/eval") if any(p.glob("S*.json")))
    adir = next(p for p in root.glob("**/audio/eval") if any(p.glob("*.wav")))
    ihm, sdm = [], []
    for jf in sorted(tdir.glob("S*.json")):
        for i, u in enumerate(json.load(open(jf))):
            sess, spk = u["session_id"], u["speaker"]
            st, en = hms(u["start_time"]), hms(u["end_time"])
            uid = f"{sess}_{spk}_{i:05d}"
            ihm.append(row(uid, sess, spk, adir / f"{sess}_{spk}.wav", 0, st, en, u["words"]))
            sdm.append(row(uid, sess, spk, adir / f"{sess}_{u['ref']}.CH1.wav", 0, st, en, u["words"]))
    return {"ihm": ihm, "sdm": sdm}


# ----------------------------------------------------------------------------------------------- DiPCo
def prep_dipco():
    root = DS / "dipco"
    base = next(p for p in root.glob("**/transcriptions") if (p / "eval").is_dir()).parent
    ihm, sdm = [], []
    for jf in sorted((base / "transcriptions" / "eval").glob("S*.json")):
        for i, u in enumerate(json.load(open(jf))):
            sess, spk = u["session_id"], u["speaker_id"]
            uid = f"{sess}_{spk}_{i:05d}"
            st, en = u["start_time"], u["end_time"]
            ihm.append(row(uid, sess, spk, base / "audio" / "eval" / f"{sess}_{spk}.wav", 0,
                           hms(st["close-talk"]), hms(en["close-talk"]), u["words"]))
            if "U01" in st:
                sdm.append(row(uid, sess, spk, base / "audio" / "eval" / f"{sess}_U01.CH1.wav", 0,
                               hms(st["U01"]), hms(en["U01"]), u["words"]))
    return {"ihm": ihm, "sdm": sdm}


def check_audio(rows, tag):
    """Every referenced wav must exist, be 16 kHz, have the channel, and contain the segment."""
    import soundfile as sf

    info = {}
    for a in sorted({r["audio"] for r in rows}):
        assert os.path.exists(a), f"{tag}: missing {a}"
        i = sf.info(a)
        assert i.samplerate == 16000, f"{tag}: {a} sr={i.samplerate}"
        info[a] = (i.channels, i.frames / i.samplerate)
    over = 0
    for r in rows:
        ch, dur = info[r["audio"]]
        assert r["channel"] < ch, f"{tag}: {r['id']} channel {r['channel']} >= {ch}"
        over += r["end"] > dur + 0.01
    assert over <= 0.001 * len(rows), f"{tag}: {over} segments end past the audio"
    print(f"  {tag}: {len(info)} audio files OK ({over} segments slightly past end)", flush=True)


def prep_librispeech():
    """Sanity-check set (not a meeting corpus): LibriSpeech test-clean, whole utterances."""
    import soundfile as sf

    rows = []
    for tf in sorted((DS / "librispeech").glob("LibriSpeech/test-clean/*/*/*.trans.txt")):
        for line in open(tf):
            uid, text = line.strip().split(" ", 1)
            wav = tf.parent / f"{uid}.flac"
            rows.append(row(uid, uid.rsplit("-", 1)[0], uid.split("-")[0], wav, 0, 0.0,
                            sf.info(str(wav)).duration, text))
    return {"clean": rows}


PREP = {"librispeech": prep_librispeech, "ami": prep_ami, "alimeeting": prep_alimeeting, "notsofar": prep_notsofar,
        "chime6": prep_chime6, "dipco": prep_dipco}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(PREP))
    args = ap.parse_args()
    for ds in args.datasets:
        for cond, rows in PREP[ds]().items():
            n0 = len(rows)
            rows = [r for r in rows if keep(r)]
            check_audio(rows, f"{ds}/{cond}")
            path = OUT / f"{ds}__{cond}.jsonl"
            write_manifest(rows, path)
            hrs = sum(r["end"] - r["start"] for r in rows) / 3600
            print(f"{ds:10s} {cond:5s} segs={len(rows):6d} (dropped {n0 - len(rows)}) hours={hrs:6.2f} "
                  f"sessions={len(Counter(r['session'] for r in rows))} -> {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
