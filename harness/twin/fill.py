"""Fill the decode cache: decode every span the twin asked for and did not find (GPU, JUBAIL).

    python -m harness.twin.fill --engine qwen  --checkpoint data/checkpoints/Qwen3-ASR-1.7B \
        --cache data/outputs/harness/cache --audio-index data/outputs/harness/audio_index.json
    python -m harness.twin.fill --engine audar --checkpoint data/checkpoints/Audar-ASR-V1-Turbo ...

Reproduces Vocali's QwenDecoder / Qwen3ASRKit (PyTorch bf16 instead of MLX 8-bit; the macmini calibration
measures the difference):
  - the span is peak-normalized to 0.6 (Transcriber.normalised)
  - prompt <|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>...<|audio_end|><|im_end|>\n
    <|im_start|>assistant\n
  - qwen (router mode, candidates zh/en/ar): the prompt stops after "language"; P(candidate) = softmax over the
    logits of the first token of " Chinese" / " English" / " Arabic"; the most likely is appended with <asr_text>
    and decoding continues greedily
  - audar: "language Arabic<asr_text>", greedy
  - at most 448 new tokens; text trimmed; token times spread over voiced frames (spread.py)
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .cache import Q
from .spread import spread

NAMES = {"zh": "Chinese", "en": "English", "ar": "Arabic"}
ASR_TEXT = "<asr_text>"
MAX_NEW = 448


def load_audio(entry):
    import soundfile as sf

    x, sr = sf.read(entry["audio"], dtype="float32", always_2d=True)
    assert sr == 16000, (entry["audio"], sr)
    return np.ascontiguousarray(x[:, entry.get("channel", 0)])


def normalised(x):
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return x * (0.6 / peak) if peak > 1e-6 else x


class Engine:
    def __init__(self, checkpoint, engine):
        import torch
        from qwen_asr import Qwen3ASRModel

        self.torch = torch
        self.engine = engine
        self.asr = Qwen3ASRModel.from_pretrained(str(checkpoint), dtype=torch.bfloat16, device_map="cuda:0",
                                                 max_new_tokens=MAX_NEW)
        self.model, self.proc = self.asr.model, self.asr.processor
        self.tok = self.proc.tokenizer
        word = self.tok.encode("language", add_special_tokens=False)
        self.word = word
        self.name_ids = {}
        for code, name in NAMES.items():
            full = self.tok.encode(f"language {name}", add_special_tokens=False)
            assert full[:len(word)] == word, (name, full, word)
            self.name_ids[code] = full[len(word):]
        self.asr_text = self.tok.convert_tokens_to_ids(ASR_TEXT)

    def _inputs(self, wav, context):
        msgs = [{"role": "system", "content": context or ""},
                {"role": "user", "content": [{"type": "audio", "audio": ""}]}]
        base = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = self.proc(text=[base], audio=[wav], return_tensors="pt", padding=True)
        return inputs.to(self.model.device).to(self.model.dtype)

    def _generate(self, inputs, ids):
        torch = self.torch
        att = torch.ones_like(ids)
        out = self.model.generate(input_ids=ids, attention_mask=att, input_features=inputs["input_features"],
                                  feature_attention_mask=inputs["feature_attention_mask"],
                                  max_new_tokens=MAX_NEW, do_sample=False)
        seq = out.sequences[0, ids.shape[1]:]
        text = self.tok.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if ASR_TEXT in text:
            text = text.split(ASR_TEXT, 1)[1]
        return text.strip()

    def decode(self, wav, context=""):
        torch = self.torch
        inputs = self._inputs(wav, context)
        ids = inputs["input_ids"]
        dev = ids.device
        with torch.inference_mode():
            if self.engine == "audar":
                tail = self.tok.encode(f"language {NAMES['ar']}", add_special_tokens=False) + [self.asr_text]
                ids2 = torch.cat([ids, torch.tensor([tail], device=dev)], dim=1)
                return self._generate(inputs, ids2), "ar", None
            probe = torch.cat([ids, torch.tensor([self.word], device=dev)], dim=1)
            out = self.model.thinker(input_ids=probe, attention_mask=torch.ones_like(probe),
                                     input_features=inputs["input_features"],
                                     feature_attention_mask=inputs["feature_attention_mask"])
            row = out.logits[0, -1].float()
            codes = list(NAMES)
            picked = row[[self.name_ids[c][0] for c in codes]]
            p = torch.softmax(picked, dim=-1).tolist()
            probs = dict(zip(codes, p))
            chosen = max(codes, key=lambda c: probs[c])
            tail = self.name_ids[chosen] + [self.asr_text]
            ids2 = torch.cat([probe, torch.tensor([tail], device=dev)], dim=1)
            return self._generate(inputs, ids2), chosen, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["qwen", "audar"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--audio-index", required=True, help="JSON {session: {audio, channel}}")
    ap.add_argument("--contexts", help="JSON {ctx key: context text}; default only the empty context")
    a = ap.parse_args()
    cache = Path(a.cache)
    index = json.loads(Path(a.audio_index).read_text())
    contexts = {"": ""}
    if a.contexts:
        contexts.update(json.loads(Path(a.contexts).read_text()))
    todo = defaultdict(set)
    for mf in sorted(cache.glob("*.misses.jsonl")):
        session = mf.name[:-len(".misses.jsonl")]
        have = set()
        cf = cache / f"{session}.jsonl"
        if cf.exists():
            for line in open(cf, encoding="utf-8"):
                e = json.loads(line)
                have.add((e["engine"], e["q0"], e["q1"], e.get("ctx", "")))
        for line in open(mf):
            m = json.loads(line)
            key = (m["engine"], m["q0"], m["q1"], m.get("ctx", ""))
            if m["engine"] == a.engine and key not in have:
                todo[session].add(key)
    total = sum(len(v) for v in todo.values())
    print(f"{a.engine}: {total} spans in {len(todo)} sessions", flush=True)
    if not total:
        return
    eng = Engine(a.checkpoint, a.engine)
    t0, done, audio_s = time.time(), 0, 0.0
    for session, keys in sorted(todo.items()):
        x = load_audio(index[session])
        with open(cache / f"{session}.jsonl", "a", encoding="utf-8") as f:
            for engine, q0, q1, ctx in sorted(keys):
                wav = normalised(x[q0 * Q:q1 * Q])
                text, lang, probs = eng.decode(wav, contexts[ctx])
                toks = [[round(s, 4), round(e, 4), u] for s, e, u in spread(text, wav)] if text else []
                f.write(json.dumps({"engine": engine, "q0": q0, "q1": q1, "ctx": ctx, "text": text, "tokens": toks,
                                    "language": lang, "probs": probs}, ensure_ascii=False) + "\n")
                done += 1
                audio_s += (q1 - q0) * Q / 16000
            f.flush()
        print(f"  {session}: {len(keys)} spans; total {done}/{total}, RTF {(time.time() - t0) / max(audio_s, 1e-9):.3f}",
              flush=True)
    for mf in cache.glob("*.misses.jsonl"):  # every miss of this engine is filled now
        rest = [l for l in open(mf) if json.loads(l)["engine"] != a.engine]
        mf.write_text("".join(rest)) if rest else mf.unlink()


if __name__ == "__main__":
    main()
