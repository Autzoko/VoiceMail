# Components of an Arabic/English/Chinese meeting pipeline (Qwen3-ASR + Audar): Audar models, LID, diarization, generalist+specialist combination

Research date: 2026-11-20. Model cards fetched live from the Hugging Face API/raw README and GitHub. Everything Audar says about itself is a **vendor claim** unless marked otherwise. Note that the Audar numbers changed between documents. The GitHub README and older card text give Turbo 24.78 avg WER, #1 of 36. The current HF card gives 23.17, #1 of 37, citing an "Aug 2026 maintainer reproduction". Cite with the date.

## Q1. Audar models (audarai org): cards, architecture, data, claims, license, diarization

### Takeaway
All five audarai repos are 2026 vendor releases with no peer-reviewed paper. The ASR models' parameter counts match Qwen3-ASR exactly (Turbo ≈ Qwen3-ASR-1.7B, Flash ≈ Qwen3-ASR-0.6B), and both are served through `qwen-asr`/vLLM's Qwen3-ASR code path. The diarizer is a NeMo **Streaming Sortformer v2.1** fine-tune extended from 4 to 8 speakers. Its card reports **no Arabic diarization numbers**: its claims rest on English and Mandarin meeting corpora.

### Cited Findings
**Org inventory (HF API, 2026-11-20)**
- Five public models (downloads/likes, created date):
  - Audar-ASR-V1-Turbo: ASR, 2026-04-23, 2,596 dl / 21 likes
  - Audar-ASR-V1-Flash: ASR, 2026-07-02, 1,946 dl
  - Audar-Diarization-V1: pipeline tag "audio-classification", 2026-06-30, 0 dl / 7 likes
  - Audar-TTS-V1-Turbo: 2026-01-27
  - Audar-TTS-V1-Flash: 2026-07-02

  All are `license:other`. — [HF API audarai](https://huggingface.co/api/models?author=audarai&full=true)

**Audar-ASR-V1-Turbo** — [model card](https://huggingface.co/audarai/Audar-ASR-V1-Turbo)
- Architecture: decoder 2,031,739,904 params (2.03B), audio encoder 317,477,504 (0.32B), total 2,349,217,408 (2.35B, bf16). Input is 16 kHz mono with a "30 s context; longer audio is chunked/streamed". The model is "built on a permissively-licensed open-weight audio-LLM foundation". The card does not name the base, but ships "self-contained Qwen3-ASR modeling code" and recommends `qwen-asr` for "benchmark parity".
- The GitHub README describes the architecture as a "Whisper-style 128-mel audio encoder → Qwen3 decoder, 30 s context". — [GitHub AudarAI/Audar-ASR-V1](https://github.com/AudarAI/Audar-ASR-V1)
- Training (vendor claim): "300,000+ hours of labeled audio, primarily Arabic and English" (MSA, Gulf, Egyptian, Levantine, Maghrebi, code-switching). There is a 4-stage curriculum: bilingual pretraining → multi-task fine-tuning → dialect PEFT → **KTO preference alignment**. KTO uses native annotators scoring 5 axes: verbatim accuracy, diacritics, code-switch handling, named entities and formatting. — [Turbo card](https://huggingface.co/audarai/Audar-ASR-V1-Turbo)
- Languages claimed: "Arabic + English + 28 more (30 total)". Metadata lists only `ar`, `en`. — [Turbo card](https://huggingface.co/audarai/Audar-ASR-V1-Turbo)
- Benchmarks (vendor-reported; the card says the leaderboard maintainers independently reproduced them in Aug 2026):
  - Open Universal Arabic ASR Leaderboard, #1 of 37: avg WER 23.17 / CER 9.20
  - Per set: SADA 28.92, CV-18 8.09, MASC-clean 16.73, MASC-noisy 27.19, MGB-2 11.08, Casablanca 47.02
  - Same table for Qwen3-ASR-1.7B: avg 33.36 (SADA 45.53, CV-18 16.90, MASC-clean 24.37, MASC-noisy 34.29, MGB-2 16.57, Casablanca 64.47)
  - Same table for Qwen3-ASR-0.6B: avg 42.19
  - Closest competitors: Cohere transcribe-arabic-07-2026 at 25.87 and omniASR_LLM_7B at 28.32

  — [Turbo card](https://huggingface.co/audarai/Audar-ASR-V1-Turbo)
- Emirati (Mixat full 1,585-clip test): 19.4 WER / 7.3 CER. Audar says the gap is mostly orthography, including Latin-vs-Arabic rendering of English loanwords. — [Turbo card](https://huggingface.co/audarai/Audar-ASR-V1-Turbo)
- An earlier version of the GitHub README reports Turbo at 24.78 avg WER (#1 of 36) and Flash at 33.31 (#11). A search snippet of an older card showed "24.7% … CV-18 3.55". The numbers drifted across revisions. — [GitHub README](https://github.com/AudarAI/Audar-ASR-V1)
- **LID / prompting protocol (important for us):**
  - The card says `language="Arabic"` "makes the package prefill `language Arabic` into the prompt, so the model never free-runs language identification". Protocol handling is "**mandatory**".
  - `language None` (no-speech) must be mapped to an empty transcript, or music/silence "can produce repetition loops".
  - Decode settings behind the published numbers: max_new_tokens=256, bf16.
  - Alternative path: a system prompt in Arabic, "فرّغ الكلام العربي التالي." The model is described as "prompt-steerable… steer other languages with the equivalent instruction".

  — [Turbo card](https://huggingface.co/audarai/Audar-ASR-V1-Turbo)
- Neither card mentions a context/hotword feature. The cards publish **no English or Chinese ASR numbers** apart from a FLEURS AR CER of 2.46% (BF16) and a Flash "FLEURS AR/EN CER ≈ 3.3%". — [Turbo card](https://huggingface.co/audarai/Audar-ASR-V1-Turbo); [GitHub](https://github.com/AudarAI/Audar-ASR-V1)
- Serving:
  - vLLM ≥0.24 natively runs it as `Qwen3ASRForConditionalGeneration` and `Qwen3ASRRealtimeGeneration` (`/v1/realtime` WebSocket).
  - Builds: W4A16 at ~2.6 GB (FLEURS-AR CER 2.46 → 3.73) and FP8 (lossless).
  - GGUF via llama.cpp mtmd. The mmproj must stay BF16.
  - Streaming uses LocalAgreement-2. The vendor claims ≥64 concurrent streams on one A100-80GB.

  — [Turbo card](https://huggingface.co/audarai/Audar-ASR-V1-Turbo)
- Limitations the card admits: Casablanca/Darija "~63% WER" (that figure contradicts the 47.02 in its own table), heavily code-switched telephony and low-SNR audio, and drift on long-form audio. — [Turbo card](https://huggingface.co/audarai/Audar-ASR-V1-Turbo)
- License: **AudarAI Community License v1.0**. It allows research and limited commercial use for qualifying "Community Entities". Enterprise, large-scale or MaaS use requires an Enterprise License. — [license link](https://www.audarai.com/license/audarai-community-license-v1.0/)
- Technical report: a PDF exists in the repo (`report/Audar-ASR-V1-Technical-Report.pdf`). I found no arXiv version. — [GitHub](https://github.com/AudarAI/Audar-ASR-V1)

**Audar-ASR-V1-Flash** — [model card](https://huggingface.co/audarai/Audar-ASR-V1-Flash)
- Decoder 596,049,920 params (0.60B), encoder 186,376,192 (0.19B), total 0.78B. Same 300k+ h data claim, SFT and KTO.
- Leaderboard avg WER 32.04 / CER 13.43, ranked #8. Per set: SADA 44.36, CV-18 15.38, MASC-clean 22.56, MASC-noisy 34.19, MGB-2 17.05, Casablanca 58.71.
- License: **AudarAI Open License v1.0** (different from Turbo). Raw output carries a `language <Lang><asr_text>` prefix that must be stripped. — [GitHub](https://github.com/AudarAI/Audar-ASR-V1)

**Audar-Diarization-V1** — [model card](https://huggingface.co/audarai/Audar-Diarization-V1)
- Base: NVIDIA **Streaming Sortformer v2.1** (NeMo, `SortformerEncLabelModel`). It is not MSDD.
- Changes:
  - The speaker head was expanded from 4 to 8 slots by modifying two Linear layers (+2,312 params).
  - The FastConformer encoder (109.55M) is frozen and only 8.15M params are trained. The card says this beats full fine-tuning by 4.2 DER.
  - Total params: 117.7M.
- Data: 200 h of synthetic 5–8-speaker conversations plus 486 h of real far-field meetings.
- Uses the Arrival-Order Speaker Cache. Identity is claimed to hold for up to 74 min.
- Streaming: 80 ms frames, 1.04 s algorithmic latency, RTF 0.003. Requires NeMo ≥2.6. fp32 safetensors.
- Metadata languages: ar, en, multilingual. Training/eval datasets: AMI, AliMeeting, DiPCo, ICSI, VoxConverse, CHiME-6, MSDWild. **None of them is Arabic.**
- DER claims (vendor; dscore, 0.25 s collar, **overlap ignored**):

  | Corpus | Audar | pyannote 3.1 (Audar's run) | Sortformer v2.1 (Audar's run) |
  |---|--:|--:|--:|
  | AMI | 15.24 | 28.60 | 24.84 |
  | AliMeeting | 18.70 | 27.38 | 25.94 |
  | DiPCo | 23.77 | | |
  | ICSI | 14.46 | | |
  | MSDWild-few | 21.09 | | |
  | MSDWild-many | 29.41 | | |
  | VoxConverse | 8.55 | | |
  | CHiME-6 | 45.00 | | |
  | **Macro** | **22.03** | 29.66 | 34.19 |
  | CALLHOME (out-of-domain) | 10.29 | 18.51 | 12.22 |

  Confusion share: 4.80 for Audar vs ~17.2–17.6 for the baselines. — [Diarization card](https://huggingface.co/audarai/Audar-Diarization-V1)
- Audar's pyannote 3.1 numbers are much worse than pyannote's own (AMI SDM 22.7, AliMeeting 24.5). Audar's numbers are stricter in some respects even though they are computed with a collar and overlap ignored. The baselines may have been run differently, so treat these comparisons with caution. — [pyannote README](https://github.com/pyannote/pyannote-audio)
- The serving gateway assigns each ASR word to the speaker who is dominant during that word's time span, so latency is max(ASR, diar). — [Diarization card](https://huggingface.co/audarai/Audar-Diarization-V1)

**Audar-TTS**
- Flash: 553M params, Open License. Turbo: 1.64B, Community License.
- Both are Arabic (MSA + Gulf/Emirati) + English. LM backbone with a single-codebook 50 Hz "audar-codec", zero-shot cloning from 5–15 s, 8 inline style tags, no G2P.

— [TTS-Flash](https://huggingface.co/audarai/Audar-TTS-V1-Flash); [TTS-Turbo](https://huggingface.co/audarai/Audar-TTS-V1-Turbo)

**Community evaluations**
- A search found only vendor material and Product Hunt launch pages. I found no independent third-party evaluation apart from the leaderboard reproduction that Audar itself reports. — [Product Hunt](https://www.producthunt.com/products/audar-asr-v1)
- The Open Universal Arabic ASR Leaderboard paper is Wang, Alhmoud & Alqurishi (arXiv 2412.13788). — [arXiv 2412.13788](https://arxiv.org/html/2412.13788v1)

### Inferences
- Parameter counts match Qwen3-ASR almost exactly:
  - Turbo's 0.32B encoder + 2.03B decoder ≈ Qwen3-ASR-1.7B (AuT 300M encoder + Qwen3-1.7B, whose full weight count including embeddings is ≈2.03B).
  - Flash's 0.19B + 0.60B ≈ Qwen3-ASR-0.6B (AuT 180M + Qwen3-0.6B).

  Combined with the shipped Qwen3-ASR code and native vLLM support, it is near-certain that Turbo and Flash are full or PEFT fine-tunes of Qwen3-ASR-1.7B and -0.6B. The tokenizer and the `language X<asr_text>` output protocol are the same.
- Because the training mix is "primarily Arabic and English", with KTO rewarding Arabic orthography, the LID head (the "language X" token) was probably skewed toward Arabic. This is consistent with our observation that 63–82% of non-Arabic utterances came out in Arabic script under auto-LID. The vendor's own instruction is to always force `language Arabic`. So **Audar should only be run with a forced language, and only on segments already routed as Arabic.** It should never decide LID.
- The "30 languages" claim seems inherited from the Qwen3-ASR base (Qwen3-ASR also supports 30 languages). The vendor gives no evidence that Audar kept Chinese or English quality. Our ~2× worse zh/en result suggests catastrophic forgetting.
- Audar's "30 s context" statement conflicts with `qwen-asr`, which chunks at ≤1200 s (see Q2). If we run Audar through `qwen-asr`, long-audio chunking is handled by the package, but we should check Audar's behaviour on segments longer than 30 s.
- Audar-Diarization-V1 has no Arabic evidence. Sortformer is largely acoustic, so it should transfer across languages, but we should validate it on our own Arabic meetings.

### Gaps
- I did not open the Audar technical-report PDF (not on arXiv). Base-model naming, exact data composition, and English/Chinese results may be in it.
- The cards do not document context/hotword support. Audar has the Qwen3-ASR prompt slot, so system-prompt context likely works mechanically, but no effect is documented.
- I found no independent DER for Audar-Diarization-V1 and no Arabic DER anywhere for it.
- The license text was not fetched in full. The thresholds for a "Community Entity" are unknown.

## Q2. Qwen3-ASR built-in LID, context prompting, timestamps, vLLM and streaming

### Takeaway
Qwen3-ASR does LID as a generated prefix (`language X<asr_text>`). It covers 30 languages + 22 Chinese dialects and reports 97.9% average LID accuracy (1.7B). That is strong on full utterances, but there is no published short-segment or code-switch LID evaluation. Context biasing exists through the system prompt (`context=`), but **the paper reports no number for its effect**. **Qwen3-ForcedAligner does not support Arabic** (11 languages only).

### Cited Findings
- Output protocol: `language English<asr_text>…`. For non-speech the output is `language None<asr_text>`. — [Qwen3-ASR paper §2.2, arXiv 2601.21337](https://arxiv.org/html/2601.21337)
- Language set: zh, en, yue, **ar**, de, fr, es, pt, id, it, ko, ru, th, vi, ja, tr, hi, ms, nl, sv, da, fi, pl, cs, fil, fa, el, hu, mk, ro, plus 22 Chinese dialects ("52 languages and dialects"). — [GitHub QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR); [paper](https://arxiv.org/html/2601.21337)
- LID accuracy (%) for Whisper-large-v3 / Qwen3-ASR-0.6B / 1.7B:

  | Set | Whisper-large-v3 | 0.6B | 1.7B |
  |---|--:|--:|--:|
  | MLS | 99.9 | 99.3 | 99.9 |
  | CommonVoice | 92.7 | 98.2 | 98.7 |
  | MLC-SLM (conversational) | 89.2 | 92.7 | 94.1 |
  | FLEURS (30 languages) | 94.6 | 97.1 | 98.7 |
  | Avg | 94.1 | 96.8 | 97.9 |

  Most remaining FLEURS errors are ms↔id confusion. — [paper Table 6](https://arxiv.org/html/2601.21337); [README](https://github.com/QwenLM/Qwen3-ASR)
- The paper has no zh-en code-switching ASR or LID benchmark. "Cross-lingual/code-switched" is claimed only for the ForcedAligner. — [paper](https://arxiv.org/html/2601.21337)
- `qwen-asr` code:
  - `transcribe(audio, context="", language=None, return_time_stamps=…)`.
  - Forcing a language appends `language {X}<asr_text>` to the assistant prompt, so the output is text-only.
  - Long audio is split at low-energy points into chunks of ≤1200 s for ASR, or ≤180 s when timestamps are requested. The minimum input is 0.5 s; shorter input is zero-padded.
  - Per-chunk languages are merged into a string like "Chinese,English".

  — [qwen_asr/inference/qwen3_asr.py](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py); [utils.py](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/utils.py)
- Context biasing: SFT includes "context biasing data". The model "learns to utilize the context tokens inside the system prompt as background knowledge". **No WER or biasing benchmark is reported.** — [paper §2.2](https://arxiv.org/html/2601.21337)
- Architecture:
  - AuT encoder with 8× downsampling of 128-dim fbank → 12.5 Hz tokens, using a dynamic flash-attention window of 1–8 s.
  - 1.7B model: AuT 300M (hidden size 1024) + Qwen3-1.7B. 0.6B model: AuT 180M (hidden size 896) + Qwen3-0.6B.
  - AuT was pretrained on ~40M h of pseudo-labelled audio, mostly zh/en.
  - RL stage (GSPO) used ~50k utterances.

  — [paper §2](https://arxiv.org/html/2601.21337)
- Arabic WER, Qwen3-ASR's own numbers:

  | Set | 0.6B | 1.7B | Flash-1208 API |
  |---|--:|--:|--:|
  | FLEURS-ar | 25.51 | 16.98 | 14.78 |
  | CommonVoice-ar | 45.99 | 37.97 | 33.86 |

  — [paper appendix](https://arxiv.org/html/2601.21337)
- Streaming (vLLM backend only; no batching, no timestamps):
  - Setup: 2 s chunks, 5-token rollback, first 4 chunks unfixed.
  - Average WER for the 1.7B model goes from 2.69 offline to 3.33 streaming. For the 0.6B model it goes from 3.48 to 4.40.
  - No streaming latency number is reported. The 92 ms TTFT figure is from offline/online serving.

  — [paper §4.5 Table 8](https://arxiv.org/html/2601.21337); [README](https://github.com/QwenLM/Qwen3-ASR)
- vLLM has day-0 support. `qwen-asr[vllm]`, `qwen-asr-serve`. The 0.6B model reaches "2000× throughput at concurrency 128". Native Transformers `-hf` checkpoints arrived 2026-06-26. — [README](https://github.com/QwenLM/Qwen3-ASR)
- Qwen3-ForcedAligner-0.6B:
  - NAR, ≤300 s, 80 ms frames.
  - **Languages: zh, en, yue, fr, de, it, ja, ko, pt, ru, es. No Arabic.**
  - AAS on human-labelled data is 32.4 ms, vs NFA 101.2 and Monotonic-Aligner 141.3. Cross-lingual concatenation: 42.5 ms.

  — [paper §3, Table 9](https://arxiv.org/html/2601.21337); [README](https://github.com/QwenLM/Qwen3-ASR)

### Inferences
- For Arabic segments, word timestamps need another aligner, for example NeMo NFA or a CTC aligner (MMS/wav2vec2-Arabic). The alternative is to rely on segment-level timestamps from diarization or VAD. Arabic word timing from Qwen3-ForcedAligner is unsupported, and any results for it in our study need caveats.
- Qwen3-ASR's LID is a reasonable first-pass router for ar/en/zh on segments of a few seconds or more (MLC-SLM conversational 94.1%). The decision comes out "for free" as the first generated token, and its token log-prob can serve as a confidence score.

### Gaps
- No published Qwen3-ASR LID accuracy for segments shorter than 2 s, for Arabic dialects, or for intra-utterance code-switching.
- No quantitative effect of context or hotword prompts has been published.

## Q3. Spoken LID options for ar/en/zh, Arabic dialects, short segments, code-switching

### Takeaway
For a 3-way ar/en/zh decision, Qwen3-ASR's own LID or a VoxLingua107 ECAPA classifier are the practical options. Both are strong on segments of 3 s or more and degrade sharply below about 1–2 s. Arabic dialect ID is far harder: the best NADI 2025 system reached ~80% accuracy, and that comes from a fine-tuned Whisper-large-v3 encoder. I found no standard Arabic-English code-switch LID benchmark.

### Cited Findings
- SpeechBrain `lang-id-voxlingua107-ecapa`: 107 languages, 6.7% error rate on the VoxLingua107 dev set. — [HF card](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)
- A study on VoxLingua107 sub-3-second subsets (mean 2.54 s) reports a 6.54% ECAPA-TDNN error rate. Another paper reports that Whisper LID "rapidly deteriorates with shorter utterances" in 1/2/3 s tests. (These come via search snippets from the Anuvani-LID literature context.) — [Anuvani-LID arXiv 2610.20263](https://arxiv.org/html/2610.20263v1)
- Anuvani-LID (preprint, Indian languages, 1 s input) gives top-1 accuracy of AmberNet 0.771, VoxLingua107 ECAPA 0.745, Voxlect Indic-Whisper 0.726, Whisper-large-v3 0.684. It reports that "at sub one second, most speech models fail". — [arXiv 2610.20263](https://arxiv.org/html/2610.20263v1)
- Whisper detects language once from the first 30 s window. It is not designed for streaming or very short clips. — [Picovoice guide (vendor)](https://picovoice.ai/blog/spoken-language-identification/)
- MMS-LID: 126 to 4,017-language heads on a ~1B wav2vec2 base. The Voxlect benchmark finds MMS-LID-256 drops more than Whisper-Large under heavy noise (5 dB SNR). — [Voxlect arXiv 2508.01691](https://arxiv.org/pdf/2508.01691)
- Qwen3-ASR's own LID is 94.1% on conversational MLC-SLM (1.7B), vs 89.2% for Whisper-large-v3. — [Qwen3-ASR paper](https://arxiv.org/html/2601.21337)
- Arabic dialect ID:
  - NADI 2025 Subtask 1 best is **79.83%** accuracy (ELYADATA-LIA: fine-tuned Whisper-large-v3 encoder + augmentation). Runner-up BYZÖ is 76.4%; voice conversion gave +7.69 pp.
  - Training on ADI-17 alone gives 31.84% zero-shot on NADI validation, vs 78.33% with ADI-20-53h.
  - NADI 2026 moves to out-of-domain ADI because systems overfit to training domains.

  — [NADI 2025 overview arXiv 2509.02038](https://arxiv.org/pdf/2509.02038); [ELYADATA-LIA arXiv 2511.10090](https://arxiv.org/pdf/2511.10090); [Saarland-Groningen](https://aclanthology.org/2025.arabicnlp-sharedtasks.102.pdf); [NADI 2026 arXiv 2609.27086](https://arxiv.org/pdf/2609.27086)
- Arabic-English code-switch resources:
  - Mixat: ~15 h Emirati-English podcasts, roughly a third of utterances code-mixed.
  - ArzEn: ~12 h Egyptian-English interviews.
  - PolyWER: a multi-reference metric for code-mixed ASR that allows transliteration and translation (EMNLP Findings 2024).
  - 2026 commercial-ASR code-switch benchmark covering Arabic.

  — [Mixat](https://aclanthology.org/2024.sigul-1.26.pdf); [PolyWER](https://aclanthology.org/2024.findings-emnlp.356.pdf); [Code-switch Arabic NLP survey arXiv 2501.13419](https://arxiv.org/pdf/2501.13419); [Commercial CS benchmark arXiv 2605.19069](https://arxiv.org/pdf/2605.19069)

### Inferences
- Suggested routing strategy:
  1. Diarize, or VAD-segment, into turns.
  2. Get the language from Qwen3-ASR auto-LID, optionally cross-checked with ECAPA-VoxLingua107. The cost is small, about 20M params.
  3. For segments shorter than ~1.5–2 s, do not trust per-segment LID. Inherit the speaker's or neighbourhood's language prior instead. Speaker-level language priors from diarization are a cheap, robust smoothing method.
  4. Route Arabic segments to Audar with a forced `language Arabic`, and route everything else to Qwen3-ASR.
- Intra-utterance Arabic-English switching: Audar was trained on code-switched Ar-En, so Arabic-dominant code-switched turns should go to Audar. English-dominant turns should go to Qwen3-ASR. Evaluate with a script-agnostic metric such as PolyWER or transliteration-normalised WER, because Mixat errors are largely Latin-vs-Arabic rendering.
- Dialect ID at ~80% is not reliable enough to gate models. It is useful only as soft metadata, for example to choose prompts.

### Gaps
- I found no head-to-head ar/en/zh 3-way LID numbers on segments shorter than 2 s. I could not retrieve AmberNet (NeMo TitaNet-LID) card numbers; the HF repo returned an empty README.
- I found no established Arabic-English code-switch LID/segmentation benchmark with numbers.

## Q4. Arabic and multilingual speaker diarization

### Takeaway
I found no Arabic-specific diarization benchmark in the academic literature. The only Arabic DER I found is from a small vendor test of 12 files. There, pyannote community-1 (15.57) beat Sortformer offline (17.92) and streaming (22.32), and Arabic was the hardest language for Sortformer. Among open models on standard sets, DiariZen and NVIDIA's new Nemotron-3-Diarization (8 speakers) are the strongest.

### Cited Findings
- pyannote numbers (DER %, pyannote's own benchmark, updated 2025-09):

  | Set | 3.1 | community-1 | precision-2 (commercial) |
  |---|--:|--:|--:|
  | AISHELL-4 | 12.2 | 11.7 | 11.4 |
  | AliMeeting ch1 | 24.5 | 20.3 | 15.2 |
  | AMI IHM | 18.8 | 17.0 | 12.9 |
  | AMI SDM | 22.7 | 19.9 | 15.6 |
  | DIHARD3 | 21.4 | 20.2 | 14.7 |
  | MSDWild | 25.4 | 22.8 | 17.3 |
  | VoxConverse | 11.2 | 11.2 | 8.5 |

  Community-1 processes 31 s per hour of AMI audio. — [pyannote-audio README](https://github.com/pyannote/pyannote-audio)
- DiariZen (BUT-FIT, wavlm-large-s80-md), DER vs pyannote 3.1:

  | Set | DiariZen | pyannote 3.1 |
  |---|--:|--:|
  | AMI | 14.0 | 22.4 |
  | AISHELL-4 | 9.8 | |
  | AliMeeting | 12.5 | 24.4 |
  | NOTSOFAR-1 | 17.9 | |
  | MSDWild | 15.6 | |
  | DIHARD3 | 14.5 | |
  | RAMC | 11.0 | |
  | VoxConverse | 9.2 | |

  — [HF card](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md)
- NVIDIA Nemotron-3-Diarization (successor to Streaming Sortformer 4spk v2.1):
  - Up to 8 speakers, streaming latency down to 0.32 s, license `openmdw-1.1`.
  - DIHARD III eval DER (full): 12.73 offline / 13.18 at 1.04 s latency, vs Sortformer v2.1 at 19.09 / 19.60. For 5–9 speakers: 27.58 vs 40.21.
  - CALLHOME part 2 (full): 9.10 vs 10.32.
  - Training: ~10k h of real conversations plus ~28k h of single-speaker audio for simulation, multilingual sources (VoxConverse, DIHARD, CALLHOME, YODAS-v2, "21 languages" of licensed transcripts).

  — [HF card nvidia/nemotron-3-diarization](https://huggingface.co/nvidia/nemotron-3-diarization)
- Streaming Sortformer v2.1 architecture: 17-layer NEST FastConformer + 18-layer Transformer (hidden size 192) with 4 sigmoid outputs. It uses the AOSC speaker cache. — [HF card](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1); [Streaming Sortformer arXiv 2507.18446](https://arxiv.org/abs/2507.18446)
- **Arabic DER**, Soniqo benchmark (vendor). The real-conversation set is 96 recordings, 12.63 h, 0.25 s collar; Arabic is 12 files, source unnamed:

  | System | Arabic | English | Mandarin |
  |---|--:|--:|--:|
  | Sortformer streaming | 22.32 | 13.81 | 9.95 |
  | Sortformer offline | 17.92 | 14.34 | 10.52 |
  | pyannote community-1 | **15.57** | 15.22 | 16.19 |
  | Nemotron-3 (MLX) | 16.54 | 10.31 | 10.00 |

  Sortformer is capped at 4 speakers, and 22 of the 96 files exceed that. — [Soniqo diarization benchmark](https://soniqo.audio/benchmarks/diarization)
- A 2025 multi-model benchmark ranks Sortformer v2-streaming, DiariZen and pyannoteAI as the best overall. The excerpt has no Arabic breakdown. — [Benchmarking Diarization Models (ResearchGate)](https://www.researchgate.net/publication/396048562_Benchmarking_Diarization_Models)

### Inferences
- Cross-lingual robustness: in the Soniqo test, end-to-end Sortformer-family models degrade more on Arabic than pyannote's clustering pipeline does. Arabic DER is roughly double the Mandarin DER, though the sample is tiny and confounded with recording source. Audar-Diarization-V1 inherits the Sortformer base and was fine-tuned without Arabic data. In our pipeline, A/B it against pyannote community-1 and DiariZen on Arabic meetings before choosing.
- The Audar diar card's comparison numbers (pyannote 3.1 at AMI 28.6) are inconsistent with pyannote's self-report (18.8–22.7). Prefer our own evaluation.

### Gaps
- No public Arabic meeting diarization corpus or benchmark results. MGB-2 has speaker labels, but I found no recent MGB-2 diarization results.
- The source of the Soniqo Arabic test set is unnamed.

## Q5. Combining a generalist and an Arabic-specialist ASR of the same architecture

### Takeaway
There is no published recipe for routing between Qwen3-ASR and a Qwen3-ASR-derived specialist. The practical design follows from the shared protocol: let the generalist (Qwen3-ASR) decide the language, and invoke the specialist (Audar) only with a forced `language Arabic` prefix.

### Cited Findings
- The `qwen-asr` package has two relevant controls. A forced language means a prefilled `language X<asr_text>`. The `context` string goes into the system prompt. Both models accept these the same way, because Audar ships Qwen3-ASR code and recommends `qwen-asr`. — [qwen3_asr.py](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py); [Audar Turbo card](https://huggingface.co/audarai/Audar-ASR-V1-Turbo)
- The Audar vendor makes forced language mandatory and maps `language None` to empty output to avoid repetition loops on non-speech. — [Audar Turbo card](https://huggingface.co/audarai/Audar-ASR-V1-Turbo)
- Both models serve natively in vLLM (`Qwen3ASRForConditionalGeneration`). They can run as two served models on one GPU; Audar W4A16 is ~2.6 GB. — [Audar Turbo card](https://huggingface.co/audarai/Audar-ASR-V1-Turbo); [Qwen3-ASR README](https://github.com/QwenLM/Qwen3-ASR)
- Literature on in-model routing (MoE projectors, language-classifier-weighted encoders) exists. Examples are SMEAR-MoE (arXiv 2601.19451), MOSA (arXiv 2508.18998), HDMoLE (arXiv 2409.19878) and Ideal-LLM (dual encoders weighted by a language classifier). I found no work on confidence-based selection between two separate same-architecture ASR models. — [SMEAR-MoE](https://arxiv.org/html/2601.19451v1); [MOSA](https://arxiv.org/html/2508.18998v3); [HDMoLE](https://arxiv.org/pdf/2409.19878)

### Inferences
- **Shared-encoder trick:** if Audar's AuT encoder weights are identical or close to Qwen3-ASR's, one encoder pass could feed both decoders. Check this by diffing `audio_tower.*` tensors. The Audar card says the 4-stage curriculum includes pretraining and multi-task FT, so the encoder was probably updated, and the two models may need separate passes. Treat this as unverified.
- **LID from Qwen3-ASR, confidence for verification:**
  - Use Qwen3-ASR's first-token `language` probability, P(Arabic) vs P(English) vs P(Chinese), as the router score.
  - Audar's LID is biased toward Arabic, so do not compare the two models' language posteriors directly.
  - Decoder log-probs are not calibrated across differently fine-tuned models: KTO and RL change confidence. If a run-both-and-choose fallback is needed, compare script and language consistency, and cross-score each hypothesis with the other model by teacher-forcing likelihood. Do not compare raw average log-probs.
- **Run both on ambiguous segments only:** for segments where Qwen3-ASR's P(ar) is in a middle band (e.g. 0.3–0.7) or that are shorter than 2 s, decode with both. Pick Audar's output if it is mostly Arabic script and Qwen3-ASR's language is ar or uncertain. Otherwise pick Qwen3-ASR.
- **Forced-language output check:** if Audar is forced to Arabic on a segment that is actually English or Chinese, it will likely translate or transliterate. This is the observed failure mode. A cheap guard is to compare script ratio and length against the Qwen3-ASR hypothesis.
- **Timestamps:** Qwen3-ForcedAligner has no Arabic, so plan a separate Arabic aligner. Or use word→speaker assignment based on diarization segment overlap, which is what Audar's own gateway does.

### Gaps
- Whether Audar's encoder differs from Qwen3-ASR's is not documented. It can be checked locally by tensor diff.
- No public study measures Audar with context prompts, or Audar's behaviour on English/Chinese with forced language. Our own benchmark appears to be the only evidence.
