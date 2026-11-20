# Harnesses, agentic controllers and self-improving loops around diarization + LID + ASR pipelines (2023 – Nov 2026)

Scope note: this file avoids repeating the sibling notes in `research_notes/Harness RSI 语音识别准确率提升/`. Those notes already cover GER/over-correction, contextual biasing, RECOVER (arXiv 2603.16411), agentic interactive ASR (arXiv 2605.29430), Voice Memory (arXiv 2607.26410), ROVER/Sommelier, DiCoW/SE-DiCoW, the NOTSOFAR-1/CHiME-8/MLC-SLM winners, Sortformer, and Arabic dialect-ID routing. Here those items are referenced only by ID. Search budget was limited (about 22 tool calls), and many items are verified only at abstract level; each item says how deeply it was checked.

## Q0. Headline: are there papers that wrap a full diarization + LID + ASR pipeline in a harness, agent or self-improving loop?

### Takeaway
I found **no published paper that puts an agentic or self-improving controller around a complete VAD/diarization → LID → multi-ASR routing pipeline**, with or without speaker-attributed multilingual output. The literature has three separate pieces:
- LLM post-processors that fix speaker labels: DiarizationLM, SEAL, LSEC, a training-free identity-aware refiner, and an ACL-SRW 2026 conservative merger.
- Human- or LLM-in-the-loop correction of speaker attribution: interactive in-meeting correction (Google, 2025).
- Single-ASR agentic or self-editing loops with no diarization and no LID: Voice Memory, RECOVER, Speech-Hands, agentic ASR 2605.29430.

So a combined pipeline-level harness would be novel.

### Cited Findings
- The closest modular "LLM-in-the-loop over SD + ASR" system is a training-free pipeline. It reconciles separate diarization and ASR outputs, then prompts an LLM to fix low-confidence and split speaker labels and to assign role identities. It reports a 29.7% relative error reduction on a real patient–clinician dataset, with no extra training. — [arXiv 2509.15082](https://arxiv.org/abs/2509.15082)
- Interactive in-meeting speaker correction with human feedback (Google, Sep 2025). Streaming ASR + diarization; an LLM writes in-meeting summaries that surface likely speaker errors; brief user feedback updates the transcript and adds online speaker enrollments. The authors use an LLM-simulated user to evaluate the loop at scale.
  - Later version, AMI headset test vs a Google-ASR + ECAPA baseline: DER −31.99% relative, speaker substitution error −52.68%.
  - The v1 snippet reported DER −9.92% and speaker confusion −44.23%. The numbers changed between versions. — [arXiv 2509.18377](https://arxiv.org/abs/2509.18377); [v3 HTML](https://arxiv.org/html/2509.18377v3)
- Speech-Hands (ACL 2026) is an omni-model with a learnable "self-reflection" primitive that decides when to trust its own perception and when to trust external ASR or audio tools.
  - It reports "12.1% WER" better than baselines across 7 OpenASR benchmarks; whether this is absolute or relative is unclear.
  - It reports 77.37% audio-QA accuracy.
  - The abstract mentions no multi-speaker or multilingual evaluation. — [arXiv 2601.09413](https://arxiv.org/abs/2601.09413)
- AudioToolAgent: a central LLM agent calls audio-LLMs and ASR as tools, issues follow-up queries and arbitrates conflicting tool outputs without hearing the audio itself.
  - MMAU up to 77.50%, MMAR 77.00%, MMAU-Pro 61.90%.
  - It uses Shapley analysis to find good agent–tool combinations.
  - The abstract mentions no diarization or multilingual handling. — [arXiv 2510.02995](https://arxiv.org/abs/2510.02995)
- Other audio-agent or agentic lineage surfaced but not opened:
  - AURA, a voice-driven tool-use agent — [arXiv 2408.05517](https://arxiv.org/abs/2408.05517)
  - Audio-mind, an "auditable agentic framework for audio understanding" — [arXiv 2605.28480](https://arxiv.org/abs/2605.28480)
  - Agentic multi-speaker dialogue TTS with critique-and-revise; it lists WavJourney, PodAgent and AuDirector as LLM-planned audio *generation* agents — [arXiv 2609.33362](https://arxiv.org/pdf/2609.33362)
  - None of these targets speaker-attributed multilingual transcription.
- Already covered in sibling notes (single-stream ASR only, no diarization or LID): Voice Memory (validation-gated self-edited correction memory), RECOVER (multi-hypothesis entity correction agent) and agentic interactive ASR (2605.29430). — [arXiv 2607.26410](https://arxiv.org/html/2607.26410v1); [arXiv 2603.16411](https://arxiv.org/abs/2603.16411); [arXiv 2605.29430](https://arxiv.org/abs/2605.29430)
- The end-to-end alternative to a harness is a single speech-LLM that emits speaker-attributed text directly:
  - Granite-speech extended to SAA with "[Speaker 1 cluster 42]" tags; reported to beat the SD→ASR cascade (numbers not in the abstract) — [arXiv 2604.11269](https://arxiv.org/abs/2604.11269)
  - DM-ASR (diarization-aware multi-speaker LLM-ASR) — [arXiv 2604.22467](https://arxiv.org/pdf/2604.22467)
  - Balancing ASR and diarization in end-to-end LLMs — [arXiv 2606.13095](https://arxiv.org/pdf/2606.13095)
  - Persistent speaker attribution *across meetings* (an evaluation paper) — [arXiv 2609.39344](https://arxiv.org/pdf/2609.39344)

### Inferences
- The design space for a pipeline-level harness is open. Its natural parts already exist as separate papers:
  - SEAL/DiarizationLM-style speaker correction
  - Speech-Hands/RECOVER-style "trust or re-query" decisions
  - Voice Memory-style validation-gated rule memory
  - Interactive-correction-style LLM user simulation for evaluation
- The useful fact from 2509.18377 is that an **LLM-simulated feedback user** was enough to evaluate a correction loop at scale. That is a cheap proxy for human review in our self-improvement loop.

### Gaps
- I found no paper that reports a self-improving loop (prompt, rule or config evolution) whose objective is cpWER/tcpWER, as opposed to WER.
- I found no multilingual (zh/en/ar) evaluation of any LLM speaker-correction method.

## Q1. LLM post-processing of speaker-attributed transcripts (DiarizationLM, SEAL, LSEC, others): reported gains

### Takeaway
Text-only LLM speaker correction gives large WDER gains on 2-speaker telephone data:
- DiarizationLM: −55.5% relative WDER on Fisher, −44.9% on Callhome (fine-tuned PaLM 2-S).
- SEAL (acoustic-conditioned plus constrained decoding): 24–43% relative Δcp reduction.

The gains depend on the ASR and the domain. Without acoustic confidence or constrained decoding, an un-conditioned LLM can make things *worse*: in SEAL's ablation, CHAE dev cpWER went from 15.52 to 17.5. Almost all evidence is English 2-speaker telephony, not far-field multi-party meetings.

### Cited Findings
- **DiarizationLM (Google; Interspeech 2024; v13 Oct 2026).**
  - Method: ASR and diarization output is serialized as compact text (`<spk:1> good morning <spk:2> how are you`), an optionally fine-tuned LLM rewrites it, and speaker labels are transferred back to the *original* ASR words with TPST (transcript-preserving speaker transfer).
  - Fine-tuned PaLM 2-S: WDER −55.5% relative on Fisher and −44.9% on Callhome English. — [arXiv 2401.03506](https://arxiv.org/abs/2401.03506)
  - Fine-tuned Llama 3 8B: Fisher cpWER 21.19 → 18.37 (search snippet of the PDF). — [arXiv PDF](https://arxiv.org/pdf/2401.03506)
  - The authors note the evaluation is mainly 2-speaker telephone speech, which is a limitation. — [arXiv PDF](https://arxiv.org/pdf/2401.03506)
- **DiarizationLM open-source repo** (google/speaker-id).
  - Released models: Gemma 4 E4B fine-tuned multi-domain (Fisher, Callhome, ICSI, AMI), Llama 3 8B (Fisher) and Llama 2 13B (Fisher). The repo includes unsloth fine-tuning and inference scripts.
  - `metrics.py` computes micro WER, WDER, cpWER and SpkCntMAE. The repo notes that its normalization differs from Google's internal tool.
  - Prompt suffix is `" --> "` and completion suffix is `" [eod]"`. — [GitHub](https://github.com/google/speaker-id/tree/master/DiarizationLM)
- **SEAL (AWS; ICASSP 2025).**
  - LLM: Mistral-7B-Instruct-v0.2 with QLoRA r=64.
  - First pass: an EEND diarizer. Frame posteriors are median-filtered and mean-pooled to word level, then bucketed into "low/med/high" confidence tags (thresholds 0.5/0.8, tuned on Fisher dev). Tags go after each turn (Spkturn) or each word (Spkword).
  - Constrained decoding (CD) masks every token except the next input word or a speaker label, so the output is word-for-word identical to the input and cannot hallucinate.
  - cpWER/Δcp results:

    | Split | Acoustic SD | LSEC | AG-LSEC | LLM, no conditioning | Best SEAL |
    |---|---|---|---|---|---|
    | Fisher test | 12.53/3.72 | 11.62/2.81 | 10.95/2.14 | 11.2/2.71 | 10.92/2.12 (Spkword-CD) |
    | CHAE dev | 15.52/5.44 | — | 14.15/4.07 | **17.5/7.37 (worse than first pass)** | 13.6/3.53 |
    | CHAE test | 14.33/5.41 | — | — | — | 13.1/4.18 |
    | RT03 test | 10.03/3.87 | — | 8.36/2.2 | — | 8.5/2.34 (Spkword) |

  - Caveats: on CHAE test the reduction is only about 13–23%, below the abstract's 24–43%, and Table I and Table II numbers are inconsistent. The non-LLM AG-LSEC baseline sometimes matches or beats SEAL (RT03). — [arXiv 2501.08421 (HTML)](https://arxiv.org/html/2501.08421v1)
- **LSEC (AWS, 2023):** a second-pass lexical LM speaker-error corrector trained only on Fisher. It gives 15–30% relative WDER reduction on telephony. — [arXiv 2306.09313](https://arxiv.org/pdf/2306.09313)
- **Generalizability (2024, v3 Mar 2025):** fine-tuned LLM diarization correctors work best on transcripts from the *same ASR* they were trained on. Merging the weights of three models, each fine-tuned on a different ASR's output, gives an ASR-agnostic ensemble that beats each ASR-specific model. Weights are on Hugging Face. — [arXiv 2406.04927](https://arxiv.org/abs/2406.04927)
- **Speaker tagging correction with non-autoregressive LMs** is a lightweight alternative to generative LLMs (not opened). — [arXiv 2409.00151](https://arxiv.org/pdf/2409.00151)
- **GenSEC Task 2 (SLT 2024)** was a post-ASR speaker-tagging correction task scored with cpWER. — [arXiv 2409.09785](https://arxiv.org/pdf/2409.09785)
- **ACL 2026 SRW follow-up** targets boundary placement and false-split merging. Its conservative validation pipeline filters 60–70% of proposed merges and gives zero cpWER degradation. It frames DiarizationLM as a text-to-text rewriter that can alter words. — [ACL Anthology 2026.acl-srw.58](https://aclanthology.org/2026.acl-srw.58.pdf)

### Inferences
- For our pipeline the important design choices are (a) **transcript-preserving** label transfer (TPST) or **constrained decoding**, so the speaker corrector can never change ASR words, and (b) **acoustic confidence tags** from the diarizer. SEAL shows an un-conditioned LLM can raise cpWER (CHAE dev 15.52 → 17.5).
- Speaker correctors are ASR-specific (2406.04927). We route between Qwen3-ASR and Audar, whose output styles differ (Arabic script, dialect orthography). So we would need per-ASR or merged correctors, or a prompt-only corrector validated per language.
- Our dominant overlap failure is SWAP (Qwen3-ASR transcribes the interferer). Label correctors relabel *words*; they cannot fix words that belong to the other speaker but were emitted on this speaker's segment. In cpWER terms SWAP appears as cross-speaker substitutions or insertions. That error class needs re-decoding (target-speaker ASR, separation), not relabeling. This is an inference; no paper measured LLM correction on SWAP-type errors.

### Gaps
- I found no LLM speaker-correction results on Chinese or Arabic, on far-field multi-party (beyond AMI in the interactive paper) or on code-switched transcripts.
- I did not extract DiarizationLM's absolute WDER values or its multi-domain Gemma results (the repo has no tables).

## Q2. Pipeline toolkits, tunable knobs, and auto-tuning against downstream WER

### Takeaway
Standard practice tunes diarization hyperparameters for **DER**, not WER. pyannote uses TPE via pyannote.pipeline; Optuna is common. One paper (TS-SEP) explicitly found that **DER-optimal and WER-optimal segmentation settings move in opposite directions**: some over-estimation of speech activity hurts DER but helps WER. I found no paper that jointly auto-tunes a diarization + LID + ASR pipeline for cpWER/tcpWER.

### Cited Findings
- **pyannote community-1 (pyannote.audio 4.x, CC-BY-4.0).**
  - DER with no collar, overlap included, 3.1 → community-1: AMI-IHM 18.8 → 17.0; DIHARD3 21.4 → 20.2; AliMeeting ch1 24.5 → 20.3; CALLHOME 28.5 → 26.7; VoxConverse 11.2 → 11.2.
  - The hosted Precision-2 model scores lower still (AMI-IHM 12.9, DIHARD3 14.7).
  - New output `exclusive_speaker_diarization`: one speaker at a time, intended to make alignment with ASR timestamps easier. — [HF model card](https://huggingface.co/pyannote/speaker-diarization-community-1); [PyPI](https://pypi.org/project/pyannote.audio)
- **pyannote knobs.**
  - Multi-label segmentation has a detection threshold θ and a clustering threshold δ. Powerset segmentation replaces θ with argmax, so only δ needs tuning. Tuning minimizes DER on a dev set with pyannote.pipeline (TPE). — [Plaquet & Bredin, arXiv 2310.13025](https://arxiv.org/pdf/2310.13025)
  - Community-1 tunables reported by a 2026 paper: `min_duration_off`, `clustering_threshold`, `fa` and `fb` (false-alarm and miss weights). Grid search over 10 training files (about 10 h) moved DER only 0.199 → 0.194. — [ShobdoSetu, arXiv 2603.19256](https://arxiv.org/pdf/2603.19256)
  - Community discussion of 3.1 hyperparameters. — [pyannote discussion #1582](https://www.github.com/pyannote/pyannote-audio/discussions/1582)
- **DER vs WER objectives disagree.** When the authors tuned segmentation hyperparameters, "DER and WER showed opposite behavior": over-estimation hurts DER, while for WER some over-estimation is helpful. — [TS-SEP, arXiv 2303.03849](https://arxiv.org/pdf/2303.03849)
- **Transcription-supported diarization** with continuous speech separation, scored with both cpWER and ORC-WER, shows that the two metrics together describe ASR quality vs whole-pipeline quality. — [arXiv 2309.16482](https://arxiv.org/pdf/2309.16482)
- **CHiME-8 DASR overview:** the gap between the baselines is explained by DER speaker confusion and speaker-counting errors (#SPK-MS missed speakers, #SPK-FA false-alarm speakers). The NeMo baseline is markedly worse on NOTSOFAR-1 because of its pipeline design. — [arXiv 2407.16447](https://arxiv.org/pdf/2407.16447)
- **NTT CHiME-8:** EEND-VC + TS-VAD + multi-channel speaker counting gives a 63% relative macro tcpWER improvement over the DASR baseline, so speaker counting and diarization are the main levers. — [arXiv 2502.09859](https://www.arxiv.org/pdf/2502.09859)
- **USTC-NERCSLIP** reports per-stage FA/Miss/SpkErr/DER next to tcpWER, which allows stage-wise attribution. Example single-channel row: Stage 1 DER 23.43, tcpWER in the high teens. — [arXiv 2409.02041](https://arxiv.org/html/2409.02041v2). Winner numbers are already in the sibling note `meeting_overlap_farfield.md`.
- **Qwen3-ASR as a pipeline component.**
  - LID accuracy (Whisper-large-v3 / Qwen3-0.6B / Qwen3-1.7B): MLS 99.9/99.3/99.9; CommonVoice 92.7/98.2/98.7; **MLC-SLM (conversational) 89.2/92.7/94.1**; FLEURS 94.6/97.1/98.7.
  - `language=None` turns on auto-LID; `language="English"` forces the language; a per-item list works for batches.
  - Qwen3-ForcedAligner-0.6B: 11 languages, up to 5 min. Accumulated alignment shift (AAS) is 42.9 ms vs WhisperX 133.2 ms and NFA 129.8 ms.
  - vLLM streaming supports neither batching nor timestamps. — [Qwen3-ASR README](https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/HEAD/README.md); [arXiv 2601.21337](https://arxiv.org/abs/2601.21337)

### Inferences
- A harness should tune diarization and VAD knobs (pyannote `clustering_threshold`, `min_duration_off`, fa/fb; segment padding; minimum segment length before LID) against **tcpWER/cpWER plus LID-routing accuracy** on a held-out set, not against DER. TS-SEP shows the two optima differ.
- `exclusive_speaker_diarization` maps naturally onto per-segment routing to a single-speaker ASR. Overlapped regions then become explicit "contested" spans that a harness can flag for SWAP checks.
- Conversational LID is the weak point even for Qwen3-ASR-1.7B: 94.1% on MLC-SLM against 98–99% on read speech. At meeting scale that is roughly 1 misrouted segment in 17 before any far-field or overlap degradation (inference from the reported figure).

### Gaps
- I found no published Optuna/BO study whose objective is cpWER/tcpWER for a WhisperX/pyannote/NeMo pipeline.
- I did not retrieve WhisperX's or NeMo MSDD/Sortformer's knob lists with numbers (WhisperX: VAD onset/offset, chunk size; NeMo: MSDD scales, sigmoid threshold). These come from prior knowledge, not verified this session.
- I did not get DiariZen's current numbers in this session (the sibling note covers DiariZen in DiCoW pipelines).

## Q3. Multi-ASR routing: LID-gated routing, code-switching, specialist vs generalist, and evidence that routing errors propagate

### Takeaway
Almost all research "routing" is **inside one model** (LR-MoE, MoLE, MoLGE, LoRA language experts). All of these note that performance collapses when the LID routing signal is wrong or missing. I found **no paper that measures how LID misrouting propagates through a cascade of separate ASR systems**, nor one on confidence-based selection between a generalist and a language-specialist model. Practitioner evidence (Whisper forums) documents exactly our Audar failure mode: a wrong or auto-detected language makes the model *translate* into that language.

### Cited Findings
- **Audar-ASR-V1-Turbo model card.**
  - 2.35B parameters (2.03B decoder, 0.32B encoder) in Qwen3-ASR architecture format. Trained with a four-stage curriculum: bilingual pretraining, multi-task fine-tuning, dialect PEFT, then KTO alignment.
  - Covers Arabic (MSA and dialects) plus English, code-switched Arabic–English and 28 other languages.
  - `language=["Arabic"]` prefills `language Arabic<asr_text>` and skips LID.
  - `language None<asr_text>` (no-speech) must be mapped to empty output, or music and silence trigger repetition loops.
  - Average WER on the Open Universal Arabic ASR leaderboard: 23.17%, ranked #1 of 37. Casablanca (Darija) 47.02% in the table vs "about 63%" in the Limitations section, an internal inconsistency.
  - **The card says nothing about non-Arabic accuracy or translation behaviour.** The "multi-task fine-tuning" stage is a plausible source of the translate-into-Arabic behaviour we observe; this is an inference. — [HF card](https://huggingface.co/audarai/Audar-ASR-V1-Turbo)
  - The card's 24.7% average WER claim conflicts with a third-party 23.17%/27.89% comparison. — [MarkTechPost](https://www.marktechpost.com/2026/10/11/tii-falcon-asr-arabic-speech-recognition/)
- **Whisper (practitioner evidence, not peer-reviewed):** with auto-detected language, Whisper sometimes picks the wrong language from the first chunk (about 30 s) and then *translates* the whole output into it, e.g. clear English output as Welsh. Accents and poor audio make it worse. Forcing the language usually fixes it. — [OpenAI forum: Welsh](https://community.openai.com/t/solved-whisper-translates-into-welsh/120780); [OpenAI forum: translates instead of transcribes](https://community.openai.com/t/whisper-3-detects-incorrect-language-translates-instead-of-transcribes/828870)
- **Short segments break LID.**
  - Whisper and MMS LID accuracy drops most on 5 s clips; going from Whisper-S to Whisper-L gains about 20% balanced accuracy at 5 s vs about 5% at 30 s. — [JASA-EL 2025 / PMC12434620](https://pmc.ncbi.nlm.nih.gov/articles/PMC12434620/)
  - LID is accurate on clips over 3 s but "far from satisfactory" at 1 s or less; BERT-LID improves short-clip accuracy by about 19.9%. — [arXiv 2203.00328](https://arxiv.org/pdf/2203.00328)
  - Whisper's best FLEURS LID is 80.3% over 82 languages (original paper). — [Whisper paper](https://cdn.openai.com/papers/whisper.pdf)
- **In-model routing depends on LID.**
  - LR-MoE (2023): frame-level LID routing to language experts, 99.4% LID accuracy; erroneous routing causes bad gradient back-propagation. — [arXiv 2307.05956](https://arxiv.org/pdf/2307.05956)
  - LoRA language experts are near-optimal when LID is provided but cannot perform well without it. — [Interspeech 2025](https://www.isca-archive.org/interspeech_2025/li25p_interspeech.pdf)
  - MoLGE (COLM 2026): learned language-group routing. — [arXiv 2607.24030](https://arxiv.org/pdf/2607.24030)
  - MoE for multilingual LLM-ASR — [arXiv 2606.10439](https://arxiv.org/html/2606.10439)
- **Code-switching.**
  - Cascaded LID → per-language ASR works for inter-sentential switches aligned with utterance boundaries but fails for intra-sentential switches. On ASCEND, overall MER is 20.4% but the error rate at switch points (PIER) is 34.27% (ICASSP 2025 result, quoted by a vendor guide). — [Deepgram guide](https://deepgram.com/learn/what-is-code-switching-asr-guide)
  - Adding code-switching capability to multilingual ASR — [arXiv 2606.21990](https://arxiv.org/pdf/2606.21990)
  - Switch-aware evaluation beyond WER (English–Yoruba) — [arXiv 2609.11786](https://arxiv.org/pdf/2609.11786)
  - Commercial code-switched ASR benchmark, where Azure-style segment-level LID is a unified, non-routed design — [arXiv 2605.19069](https://arxiv.org/pdf/2605.19069)
- **Multi-system fusion** across same-language systems (ROVER, LLM-Select) is in the sibling notes. No source fused hypotheses from ASR models that **output different languages or scripts**.
- **Arabic dialect-ID routing** (sibling `arabic_asr.md`): out-of-domain dialect ID is about 56% at NADI 2026, so soft conditioning plus an ensemble beats hard routing.

### Inferences (design guidance for our router)
- **Never call Audar with `language=None` in production.** Always prefill `language Arabic<asr_text>`, and send only segments with high Arabic-LID confidence. Treat Audar output in Arabic script on a segment whose acoustic LID says zh/en as a **translation signature**. That check is cheap: script detection on output vs. LID label.
- **Use cross-model agreement as a routing verifier.** Run Qwen3-ASR auto-LID (98.7–99.9% on clean read speech, 94.1% conversational) and an independent LID (e.g. an ECAPA or MMS LID) on each diarized segment. If they disagree, or the segment is under about 3 s, (a) merge with neighbouring same-speaker segments to get longer LID context, or (b) decode with both Qwen3-ASR (forced zh/en) and Audar (forced ar), then pick the output using script and language consistency plus LM or acoustic confidence.
- **Use speaker-level LID priors.** Diarization gives speaker clusters, and a speaker's dominant language is a strong prior. A per-speaker language posterior aggregated over their segments should beat per-segment LID on short turns. No paper verifies this in a routing context; it is a harness-design inference.
- Because Audar translates 63–82% of non-Arabic utterances, the cost of misrouting is asymmetric: a misrouted zh/en segment loses nearly 100% of its content. Set the routing threshold to favour Qwen3-ASR when uncertain, since Qwen3-ASR also handles Arabic, though less well.

### Gaps
- I found no paper measuring LID-error propagation in a cascade of separate ASR models, and none on confidence-based generalist vs specialist selection. This is an open research gap worth claiming.
- I found no peer-reviewed study of "speech-LLM ASR translates instead of transcribes under wrong or no language tag" (language confusion in speech LLMs); only forum evidence for Whisper.
- Audar's real accuracy on English and Chinese is not published.

## Q4. Pipeline-level metrics, evaluation toolkits, and error attribution to modules

### Takeaway
Use **MeetEval**. It provides cpWER, ORC-WER, MIMO-WER, tcpWER, tcORC-WER and DI-cpWER, plus alignment visualization. Score each system with a speaker-attributed metric (cpWER/tcpWER) and a speaker-agnostic one (ORC-WER/tcORC-WER). The difference between the two isolates diarization and attribution error from recognition error. Add DER components (FA/Miss/Confusion, speaker-count error) and LID accuracy per segment to complete the module-level attribution.

### Cited Findings
- **MeetEval (CHiME 2023, von Neumann et al.):**
  - Metrics: cpWER, ORC-WER, MIMO-WER and time-constrained variants (tcpWER, tcORC-WER). The time constraint counts a word as correct only when its timing is plausible.
  - Segment-level timestamps approximate word timing well.
  - Built because NIST SCTK is memory-heavy and format-limited.
  - CLI: `meeteval-wer cpwer|tcpwer|orcwer|tcorcwer|mimower`; also DI-cpWER. Includes an alignment visualizer. — [arXiv 2307.11394](https://arxiv.org/pdf/2307.11394); [GitHub fgnt/meeteval](https://www.github.com/fgnt/meeteval); [PyPI](https://pypi.org/project/meeteval/)
- **ORC-WER ignores speaker-label mistakes, while cpWER penalizes them.** "The combination of both metrics describes the ASR performance and the overall capability of the pipeline." — [arXiv 2309.16482](https://arxiv.org/pdf/2309.16482)
- **NOTSOFAR-1:** speaker-attributed tcpWER is "highly sensitive to speaker diarization errors" because correctly recognized but misattributed words are penalized. The challenge therefore also reports a speaker-agnostic metric. — [arXiv 2401.08887](https://arxiv.org/pdf/2401.08887); [summary, arXiv 2501.17304](https://arxiv.org/html/2501.17304v2)
- **CHiME-8 DASR** attributes baseline differences to DER speaker confusion plus missed and false-alarm speaker counts. — [arXiv 2407.16447](https://arxiv.org/pdf/2407.16447)
- **DiarizationLM `metrics.py`:** WDER (word-level diarization error), cpWER and SpkCntMAE (speaker-count mean absolute error), as micro metrics. — [GitHub](https://github.com/google/speaker-id/tree/master/DiarizationLM)
- **SEAL's Δcp = cpWER − WER**, the diarization-induced share of cpWER, is a single-number attribution of speaker error. — [arXiv 2501.08421](https://arxiv.org/html/2501.08421v1)
- **Code-switching metrics:** PIER, the error rate at switch points (34.27% vs 20.4% overall MER on ASCEND). — [Deepgram guide quoting ICASSP 2025](https://deepgram.com/learn/what-is-code-switching-asr-guide); switch-aware evaluation — [arXiv 2609.11786](https://arxiv.org/pdf/2609.11786)
- **The ACL-SRW 2026 paper argues DER misses boundary-placement and false-split usability**, and uses cpWER-no-degradation as its safety gate. — [ACL Anthology](https://aclanthology.org/2026.acl-srw.58.pdf)

### Inferences (proposed module-attribution ladder for our pipeline)
Each step swaps one module for its oracle on a labeled dev set and reports tcpWER/tcORC-WER (CER for zh, normalized WER for ar):
1. Full system.
2. Oracle LID (forced language per segment). The difference is the LID/routing cost, which includes Audar's translation losses.
3. Oracle diarization (reference segments and speakers). The difference is the diarization cost.
4. Close-talk or oracle-separated audio. The difference is the overlap/far-field cost, including SWAP.

Report Δcp (cpWER − ORC-WER), SpkCntMAE, LID accuracy per segment-duration bucket, and an **output-language mismatch rate** (output script ≠ reference language) as an Audar-translation-specific KPI. A self-improving harness should use these signals as its fitness and failure-taxonomy signals. The ladder is my synthesis; individual oracle-swap ablations of this kind appear in the CHiME/NOTSOFAR system papers (e.g. oracle vs DiariZen diarization in DiCoW, in the sibling note).

### Gaps
- I found no standard metric for "wrong-language output" or translation-instead-of-transcription; we would have to define one.
- MeetEval's handling of mixed-script (zh/ar/en) normalization and CER/WER mixing within one meeting was not checked. Its 0.2.0 release notes say it does not support alternate transcripts.
