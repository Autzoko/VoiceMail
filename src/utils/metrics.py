"""Corpus-level error rates (WER/CER) with substitution/deletion/insertion breakdown.

Error rate = (S + D + I) / N_ref, accumulated over all utterances (NOT the mean of per-utterance rates).
"""
from dataclasses import asdict, dataclass

import jiwer


@dataclass
class ErrorCounts:
    sub: int = 0
    dele: int = 0
    ins: int = 0
    hits: int = 0
    n_ref: int = 0
    n_utt: int = 0

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def rate(self) -> float:
        return 100.0 * self.errors / max(self.n_ref, 1)

    def __iadd__(self, o: "ErrorCounts"):
        for k in asdict(self):
            setattr(self, k, getattr(self, k) + getattr(o, k))
        return self

    def as_dict(self, prefix=""):
        n = max(self.n_ref, 1)
        return {
            f"{prefix}error_rate": round(self.rate, 2),
            f"{prefix}sub_rate": round(100.0 * self.sub / n, 2),
            f"{prefix}del_rate": round(100.0 * self.dele / n, 2),
            f"{prefix}ins_rate": round(100.0 * self.ins / n, 2),
            f"{prefix}n_ref_tokens": self.n_ref,
            f"{prefix}n_utts": self.n_utt,
        }


def count_errors(ref_tokens: list, hyp_tokens: list) -> ErrorCounts:
    if not ref_tokens:
        return ErrorCounts(ins=len(hyp_tokens), n_utt=1)
    if not hyp_tokens:
        return ErrorCounts(dele=len(ref_tokens), n_ref=len(ref_tokens), n_utt=1)
    o = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens))
    return ErrorCounts(o.substitutions, o.deletions, o.insertions, o.hits, len(ref_tokens), 1)
