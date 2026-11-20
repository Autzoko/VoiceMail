# Model-level self-improvement for ASR / speech LLMs (2023 – Nov 2026), with a view to Qwen3-ASR (0.6B/1.7B), meetings/lectures, EN + AR

Notes are in English; the final report is in Chinese. Items marked **[bg]** are well-known background papers cited from prior knowledge with their arXiv links. They were not re-fetched in this session, so check the exact numbers before quoting them.

## Q1. Iterative pseudo-labeling / self-training: filtering, consensus, rounds, saturation and collapse

### Takeaway
Iterative pseudo-labeling (IPL) still works, but in 2025–26 most of the gain comes from **what verifies the pseudo-labels**: multi-model CER agreement, NER criteria, or an audio-LLM corrector. The number of rounds matters much less. Gains typically **saturate after about 2 rounds**. Plain IPL on hard real-world domains (earnings calls, AMI meetings) can perform *worse* than the supervised baseline because of confirmation bias.

### Cited Findings
- **ReHear (arXiv 2602.18721, 2026).** Whisper-large-v3 is the base model. An audio-aware LLM (Voxtral-Mini-3B-2507) sees the audio plus the ASR hypothesis and refines the pseudo-labels, with an optional DeBERTa-v3 filter. Data: Earnings-21/22, SPGISpeech, AMI-IHM. Test WER on Earnings-21 by iteration: iter0 8.98 → iter1 6.30 → iter2 6.43 → iter3 6.42, so gains "saturate around iteration two". Final comparison on Earnings-21 test: ISL 6.76, plain IPL **8.02**, ReHear **6.34**. The paper states that traditional IPL "often struggles to improve upon the supervised baseline" and that on Earnings and AMI "confirmation bias leads to performance degradation." No LLM is needed at inference. — [ReHear](https://arxiv.org/html/2602.18721)
- **Incremental retraining + data filtering for multi-domain ASR (Interspeech 2025, arXiv 2506.04981).** Filters pseudo-labels with multi-model consensus (CER between ASR systems, ROVER-like) and with an NER-based criterion. CER consensus worked best and beat baselines by up to **23.78% (Wow)** and **14.01% (Fisher)**. — [arXiv 2506.04981](https://arxiv.org/html/2506.04981); [ISCA](https://www.isca-archive.org/interspeech_2025/carofilis25_interspeech.pdf)
- **Efficient data selection for ASR domain adaptation (arXiv 2506.03681).** Uses Whisper and Zipformer pseudo-labels with three heuristics: low predicted WER, high inter-ASR agreement (low CER), and the presence of named entities. Filtering kept **100 h (1.4%)** of the data and matched fine-tuning on **7,500 h** of pseudo-labeled call-center audio. — [arXiv 2506.03681](https://www.arxiv.org/pdf/2506.03681) (via search-result summary)
- **Estonian TV subtitles (arXiv 2501.05234).** Whisper-large-v3 was fine-tuned on subtitles, then went through 2 rounds of pseudo-labeling on unlabeled audio. Iterative PL improved every metric. **LLM post-editing of pseudo-labels during training did not help.** (Contrast this with ReHear, where the corrector is audio-aware.) — [arXiv 2501.05234](https://arxiv.org/html/2501.05234v1)
- **ILT (Interspeech 2025 MLC-SLM workshop, arXiv 2507.08477).** Combines iterative LoRA training ("Focus–Feedback–Fix") with iterative pseudo-labeling on Whisper-large-v3 and Qwen2-Audio for multilingual conversational ASR. — [arXiv 2507.08477](https://arxiv.org/html/2507.08477v1)
- **Elderly speech (arXiv 2606.16546).** Adds confidence-score-guided incremental pseudo-labeling and speaker-adaptive pseudo-labeling to the iterative loop. — [arXiv 2606.16546](https://arxiv.org/pdf/2606.16546)
- **Pseudo2Real (arXiv 2510.08047).** Corrects *systematic* pseudo-label errors in parameter space using task arithmetic. — [arXiv 2510.08047](https://www.arxiv.org/pdf/2510.08047)
- **Qwen3-ASR's own AuT encoder** was pretrained on about **40M hours of pseudo-labeled ASR data**, mostly Chinese and English. Industrial pseudo-labeling at scale is already built into the base model. — [Qwen3-ASR report](https://arxiv.org/html/2601.21337)
- **[bg] Classic recipes.** Noisy Student Training for ASR (Park et al. 2020) used LibriSpeech 100h plus LibriLight with iterative rounds, data augmentation, and LM fusion/filtering — [arXiv 2005.09629](https://arxiv.org/abs/2005.09629). slimIPL (Likhomanenko et al.) uses a dynamic cache of pseudo-labels for continuous IPL, which avoids an explicit round structure — [arXiv 2010.11524](https://arxiv.org/abs/2010.11524). Distil-Whisper filters pseudo-labels by a WER threshold between the Whisper pseudo-label and weak transcripts — [arXiv 2311.00430](https://arxiv.org/abs/2311.00430).

### Inferences
- For Qwen3-ASR on meetings and lectures: use **ensemble agreement** (e.g., Qwen3-ASR-1.7B vs Whisper-large-v3 vs another strong ASR, keeping segments with low pairwise CER), optionally add an **audio-aware LLM corrector**, and run **1–2 rounds**. Monitor a held-out *human-labeled* dev set every round as the collapse detector.
- A text-only LLM "judge" or corrector is risky: the Estonian result was negative and ReHear's positive result relies on audio conditioning. A judge that cannot hear the audio tends to "fix" hard words toward plausible but wrong text.

### Gaps
- No 2025–26 paper found that combines ROVER fusion with an LLM verifier as a pseudo-label gate for a speech-LLM ASR.
- No IPL paper found that specifically targets Qwen3-ASR or far-field meeting audio in Arabic.

## Q2. Reinforcement learning for ASR / speech LLMs (GRPO/GSPO/PPO/DPO, label-free rewards)

### Takeaway
GRPO with a simple **1 − WER (negated WER) reward** is now an established last stage after SFT. Relative WER gains are typically 5–18% in-domain. Gains are much larger on out-of-domain meeting audio, because RL **removes insertion and hallucination errors**. Qwen3-ASR itself already uses GSPO RL on about 50k utterances. **Label-free** variants (attention-saliency reward, semantic CLAP-style reward) exist and give 10–20% relative gains. Self-confidence rewards have been shown to reinforce confident errors.

### Cited Findings
- **GRPO for ASR (Amazon, arXiv 2509.01939, Sept 2025).**
  - Setup: Llama3-style 2B and 8B LLMs with a frozen 2B Conformer encoder, trained on about 88k h of open data (Fleurs, MLS, VoxPopuli, People's Speech, CommonVoice, CoVoST2). Rewards: negated WER (best), exact match, negated edit count. LR 1e-6, batch 64, ≤5k steps, β=0.04, G=6 or 10 (no significant difference).
  - In-domain: People's Speech 2B 23.5 → 21.53, 8B 25.48 → 21.42. VoxPopuli 2B 10.12 → 9.47, 8B 9.66 → 8.98. MLS roughly flat (5.48 → 5.49). The paper claims up to **18.4% relative**.
  - Out-of-domain hallucination (insertions): on AMI-SDM, 2B **87.88 → 37.59** (insertions 56.7 → 7.4) and 8B **227.98 → 39.69**. On AMI-IHM, 2B 56.0 → 20.0. TEDLIUM 3.9 → 3.7.
  - Domain adaptation on AMI: SFT gives IHM 19.17 / SDM 44.34, while GRPO gives **15.55 / 31.98**, about 19–28% relative better than SFT. GRPO also degrades MLS and VoxPopuli less than SFT does.
  - Evaluates no reward without ground truth. Learned reward models are flagged as prone to reward hacking.
  - [arXiv 2509.01939](https://arxiv.org/html/2509.01939v1)
- **Alibaba Tongyi "Explore RL for LLM-based ASR and TTS" (arXiv 2509.18569).** Tests rule-based GRPO rewards for ASR (base reward 1 − WER, plus a hallucination-targeted reward) and studies how the RL data is constructed. Reports significant gains "even with limited training data and few optimization steps". Numbers were not retrievable from the abstract page. — [arXiv 2509.18569](https://arxiv.org/abs/2509.18569v1)
- **"When Synthetic Speech Is All You Have: Better Call GRPO" (arXiv 2607.08409, IEEE SLT 2026, Idiap and others).** Banking domain, LLM-based ASR trained on TTS audio only. SFT 36.71% → GRPO **22.09%**, and SFT then GRPO about **20.2%** (about 45% relative). A plain WER reward is sufficient. The gain comes from **fewer insertions (better stopping calibration)** and better audio anchoring of attention, a change in "behavior, not representation." — [arXiv 2607.08409](https://arxiv.org/abs/2607.08409)
- **Qwen3-ASR's RL stage** uses **GSPO** (Group Sequence Policy Optimization) on about **50k utterances**: 35% Chinese/English, 35% multilingual, 30% "functional" (e.g., non-speech, context, instruction-robustness). — [Qwen3-ASR report](https://arxiv.org/html/2601.21337)
- **Label-free RL: SI-SDA (Interspeech 2025, arXiv 2507.20169, UIUC).**
  - Base model Qwen2-Audio. The reward is an **attention-saliency score** measuring how much each beam hypothesis relies on prompt tokens instead of audio (lower reliance is better), so no ground truth is needed.
  - Optimization is policy-gradient over beam-10 hypotheses with a mean-reward baseline. LoRA r=16, 2 epochs. Uses only **2–5 h of unlabeled target audio**, beyond which validation plateaus. Single round.
  - Zero-shot → SI-SDA WER: CHiME-4 6.6 → 5.7; LS+Freesound babble 25.6 → 20.7, airport 11.9 → 10.2; CV-Indian 6.1 → 5.5, CV-Singapore 5.8 → 5.0; Switchboard 11.1 → 9.8; **TED-LIUM3 7.2 → 5.7 (−20.8% relative)**. DPO was only an ablation baseline.
  - [arXiv 2507.20169](https://arxiv.org/html/2507.20169v1); [ISCA](https://www.isca-archive.org/interspeech_2025/wang25i_interspeech.html)
- **Test-time RL with a semantic reward: ASR-TRA (arXiv 2603.05231).** See Q3. It explicitly argues that confidence or entropy rewards reinforce high-confidence errors. — [arXiv 2603.05231](https://arxiv.org/html/2603.05231)
- **General label-free RL (text LLMs).** **Cross-model consensus** rewards beat **self-consistency** (majority-vote, TTRL-style) rewards: +4.5 vs +1.5 on MATH-500 for Qwen2.5-1.5B. Resampling the policy itself reinforces its own systematic errors, while a decorrelated panel of models does not share them. — [LLMs as a Jury, arXiv 2607.10139](https://arxiv.org/pdf/2607.10139); see also [Cross-model entropy, arXiv 2605.29009](https://arxiv.org/html/2605.29009v1)
- **[bg] MWER / sequence-discriminative training** (Prabhavalkar et al. 2018) is the pre-LLM ancestor of WER-reward RL: it minimizes expected WER over the N-best list. — [arXiv 1712.01818](https://arxiv.org/abs/1712.01818)

### Inferences
- For Qwen3-ASR, the most promising "self-improvement RL" is GRPO or GSPO with a **pseudo-reference WER reward**. The pseudo-reference would be a ROVER/consensus transcript from *other* ASR systems on unlabeled meeting and lecture audio, kept only where the systems agree. This mirrors the text-LLM finding that cross-model consensus beats self-consistency. It is an inference: no ASR paper tests it directly.
- RL's largest documented win is on **insertions and hallucinations in out-of-domain far-field meetings** (AMI-SDM). This matches the long-form, noisy, silence-heavy meeting and forum audio in the target use case.
- Qwen3-ASR has already gone through GSPO, so a further in-house RL stage will likely yield gains closer to the in-domain 5–10% than to the AMI 50%+. That large drop came from an SFT model that hallucinated badly.

### Gaps
- No public paper found reporting RL fine-tuning *of Qwen3-ASR itself* or of Qwen2.5-Omni for ASR, with WER numbers.
- No ASR paper found that uses a TTS round-trip or back-translation reward, or an LLM-judge reward, with reported WER.
- The Alibaba 2509.18569 numbers were not retrieved.

## Q3. Test-time adaptation (TTA) / test-time scaling for ASR

### Takeaway
Entropy-minimization TTA (SUTA family) gives small or inconsistent gains on modern models. In one 2026 study it even **hurt** on accented speech and **exploded on confidently-wrong noisy utterances**. Methods that add an *external* signal do better: LM rescoring (SUTA-LM), a semantic reward (ASR-TRA), or acoustic-likelihood reranking (LCAR). Test-time scaling via N-best reranking mainly helps **hallucination mitigation** in speech LLMs, including Qwen3-ASR-1.7B.

### Cited Findings
- **ASR-TRA (arXiv 2603.05231, 2026).** Whisper-Tiny base. Adds a learnable decoder prompt, samples candidates with temperature, and updates the model with RL using an audio-text semantic reward.
  - Noisy LibriSpeech test-other (MS-SNSD, 10 dB, 8 noise types): baseline 32.71, SUTA 32.27, SGEM 30.22, **ASR-TRA 28.64**. Latency: 0.72 s for ASR-TRA vs 1.69 s for SUTA.
  - L2-Arctic accented English: baseline 32.11, **SUTA 32.59 (worse)**, SGEM 31.40, **ASR-TRA 28.21**.
  - On the 100 highest-confidence noisy utterances: baseline 83.61, **SUTA 122.37**, SGEM 63.00, ASR-TRA 45.17. This illustrates "blind confidence."
  - [arXiv 2603.05231](https://arxiv.org/html/2603.05231)
- **SUTA-LM (arXiv 2506.11121).** Extends SUTA entropy-minimization TTA with LM rescoring and automatic per-utterance step selection. Evaluated on 18 ASR datasets, code public. — [arXiv 2506.11121](https://arxiv.org/html/2506.11121v1)
- **SLM-TTA (arXiv 2512.24739).** The first TTA framework for generative spoken LMs with interleaved audio-text prompts. Uses entropy minimization or confidence-filtered pseudo-labeling, episodic per batch. Evaluated on ASR, speech translation, and 19 AIR-Bench tasks. — [arXiv 2512.24739](https://arxiv.org/html/2512.24739)
- **DM-SUTA (EMNLP 2025).** Uses a dynamic model bank to avoid collapse from error accumulation in *continual* single-utterance TTA. Applies to encoder-decoder ASR. — [ACL Anthology](https://aclanthology.org/anthology-files/pdf/emnlp/2025.emnlp-main.1107.pdf)
- **[bg] SUTA original (2022).** Source-free single-utterance TTA by entropy minimization plus minimum-class-confusion. — [arXiv 2203.14222](https://arxiv.org/pdf/2203.14222)
- **LCAR (arXiv 2608.30776, 2026).** Training-free, likelihood-constrained acoustic reranking of n=4 candidates. Evaluated on **Qwen3-ASR-1.7B**, Qwen2-Audio-7B, Kimi-Audio-7B, and GLM-ASR-Nano. At δ=0.60 it removes **38.8–57.1% of detected hallucination failures** while largely keeping WER/CER on standard sets. — [arXiv 2608.30776](https://arxiv.org/abs/2608.30776)
- **Text-LLM TTS caution.** As the candidate set grows, imperfect proxy scores get over-optimized. — [arXiv 2608.04001](https://arxiv.org/html/2608.04001v1)

### Inferences
- In production meeting ASR, per-utterance gradient TTA is expensive at 0.6B–1.7B scale (latency) and fragile. Small-N sampling plus acoustic-likelihood or consensus reranking (LCAR-style) is the cheaper, safer test-time lever. The transcripts it chooses can also feed the offline self-training loop in Q1.
- SUTA-style entropy minimization should not be used on its own for far-field or accented audio. The evidence above shows it degrading.

### Gaps
- No TTA results found on far-field meeting corpora (AMI-SDM, NOTSOFAR, LibriCSS) for speech LLMs.
- No best-of-N *scaling curve* (WER vs N) found for Whisper or Qwen-family ASR.

## Q4. Synthetic-data loops (TTS domain data, LLM text, simulated far-field/overlap)

### Takeaway
LLM-generated in-domain text → TTS → fine-tune is a validated adaptation route when real audio is scarce or private. RL (GRPO) on synthetic audio closes much more of the gap than SFT does. No paper found shows a *compounding* multi-round TTS loop. The evidence is single-shot adaptation, and synthetic-only SFT tends to overfit TTS artifacts.

### Cited Findings
- **Synthetic-only plus GRPO (SLT 2026).** SFT on TTS-only data gives 36.71% WER; GRPO gives 22.09%; SFT+GRPO gives about 20.2%. — [arXiv 2607.08409](https://arxiv.org/abs/2607.08409)
- **Meta DAS (arXiv 2501.12501).** Adapts Whisper to a target domain using only LLM-generated domain text plus TTS, with **no real audio or text**. WER numbers not retrieved. — [arXiv 2501.12501](https://arxiv.org/html/2501.12501v1)
- **Low-resource synthetic text and TTS (arXiv 2509.15373).** Compares gloss-replacement, random-replacement, and LLM text generation feeding TTS into wav2vec2. Mixing original and synthetic audio gave a **14.3% absolute WER reduction** for Nashta. — [arXiv 2509.15373](https://arxiv.org/pdf/2509.15373v2) (via search summary)
- **Optimized synthetic data generation, Québec French (arXiv 2508.21631).** Voice-cloning TTS with LLM-generated text and out-of-domain voices. — [arXiv 2508.21631](https://arxiv.org/html/2508.21631v1)
- **Hitachi (arXiv 2603.16920).** LLM-based text filtering plus phonetic-respelling augmentation for TTS-based ASR adaptation. Reports gains across several domain datasets. — [arXiv 2603.16920](https://arxiv.org/html/2603.16920)
- **Sudanese Arabic (arXiv 2601.06802).** Also tests TTS augmentation; see Q5. — [arXiv 2601.06802](https://arxiv.org/html/2601.06802v1)
- **[bg] FoundationTTS (2023).** English adaptation test set: RNN-T 15.05% → 9.01% WER with TTS adaptation data. — [arXiv 2303.02939](https://arxiv.org/pdf/2303.02939)
- **Qwen3-ASR SFT** already includes streaming-enhancement, non-speech, and context-biasing data, and its reported robustness is strong: ExtremeNoise 16.17 (1.7B) vs 63.17 for Whisper-large-v3. — [Qwen3-ASR report](https://arxiv.org/html/2601.21337)

### Inferences
- For vocabulary and terminology gaps (course names, forum jargon, Arabic named entities), LLM-text → TTS → light SFT or GRPO is a cheap targeted fix. Simulating far-field conditions (RIR convolution, overlap mixing at controlled overlap ratios) on top of TTS or real near-field solo speech is the natural way to cover meeting acoustics. That simulation step is not tested in the papers above.
- Any loop needs real-audio validation each round. The synthetic → real gap is the main risk, and the SLT-2026 paper suggests RL is more robust to that gap than SFT.

### Gaps
- No 2025–26 paper found that shows a multi-iteration synthetic-data loop compounding WER gains across rounds.
- No speech-LLM paper retrieved with WER numbers for RIR plus overlap simulation specifically.

## Q5. Low-resource languages / Arabic dialects

### Takeaway
Arabic dialect ASR gains in 2025–26 come mostly from **large-scale weak supervision** (about 15k h of weakly labeled MSA plus dialect audio) followed by small high-quality SFT. **Confidence-filtered teacher-student self-training** on dialectal Whisper is reported for Sudanese. Qwen3-ASR's Arabic baseline is weak (CommonVoice-ar about 38–46, Fleurs-ar about 17–26), so Arabic has the most headroom.

### Cited Findings
- **Salhab et al. (arXiv 2504.12254, 2025).** A Conformer trained from scratch on weakly annotated MSA and dialectal Arabic, with no manual transcription. — [arXiv 2504.12254](https://arxiv.org/html/2504.12254v2)
- **Munsit at NADI 2025 Shared Task 2 (arXiv 2508.08912).** Pretrained on **15,000 h of weakly labeled MSA + DA**, then continual SFT on filtered weak labels plus a small high-quality annotated set, for multidialectal Arabic ASR. — [arXiv 2508.08912](https://arxiv.org/pdf/2508.08912)
- **Sudanese Arabic self-training (arXiv 2601.06802, Jan 2026).** A teacher fine-tuned on cleaned Sudanese data pseudo-labels about **17,973 unlabeled recordings**. Confidence filtering at **0.7 and 0.9** thresholds, then student training on the combined data. Also tests TTS augmentation. The best model is Whisper-Medium. — [arXiv 2601.06802](https://arxiv.org/html/2601.06802v1)
- **Qwen3-ASR Arabic numbers.** CommonVoice-ar **37.97 (1.7B) / 45.99 (0.6B)**; Fleurs-ar **16.98 / 25.51**. The metric is not stated in the summary (likely WER). Arabic is among the 30 supported languages. — [Qwen3-ASR report](https://arxiv.org/html/2601.21337)
- **Low-resource synthetic data.** 14.3% absolute WER reduction from mixing synthetic and real audio (Nashta). — [arXiv 2509.15373](https://arxiv.org/pdf/2509.15373v2)

### Inferences
- For Arabic meetings and lectures, the high-yield self-improvement path is: (i) pseudo-label in-domain Arabic audio with an *ensemble* of a strong Arabic ASR and Qwen3-ASR-1.7B, (ii) keep agreement-filtered segments, (iii) SFT, then optionally GRPO against the consensus references. Arabic-specific normalization is critical for both filtering and reward (diacritics, alef/hamza/taa-marbuta variants, Arabic–English code-switching). Otherwise agreement filtering and WER rewards are noisy.
- The 0.6B model is about 8 to 9 points worse than 1.7B on Arabic. That points to a 1.7B teacher → 0.6B student distillation if 0.6B is required for deployment.

### Gaps
- Exact WER numbers for the Sudanese and Munsit systems were not retrieved.
- No RL, TTA, or self-training paper found specifically on Gulf, Levantine, or Egyptian **meeting or lecture** speech with speech LLMs.

## Q6. Qwen3-ASR: technical report, context biasing, fine-tuning tooling

### Takeaway
Qwen3-ASR (arXiv 2601.21337, released 29 Jan 2026, Apache-2.0) is a Qwen3-Omni-derived speech LLM in 1.7B and 0.6B sizes. Its training runs AuT encoder pretraining (40M h pseudo-labeled) → Omni pretraining (about 3T tokens) → ASR SFT (including context biasing) → GSPO RL (about 50k utterances). An official SFT script exists. The fetched pages did not document ms-swift or LoRA support for it.

### Cited Findings
- **Architecture.**
  - 1.7B = Qwen3-1.7B LLM + projector + **300M AuT encoder** (hidden size 1024).
  - 0.6B = Qwen3-0.6B LLM + **180M AuT encoder** (hidden size 896).
  - AuT downsamples Fbank features 8× to **12.5 Hz** and uses dynamic flash-attention windows of 1–8 s.
  - [Qwen3-ASR report](https://arxiv.org/html/2601.21337)
- **Training stages.**
  1. AuT pretraining on about 40M h of pseudo-labeled data, mostly Chinese and English.
  2. Omni pretraining on about 3T tokens.
  3. ASR SFT on a smaller multilingual set, disjoint from pretraining, plus non-speech, streaming-enhancement, and **context-biasing** data. The model outputs a language tag plus transcript, is trained to *ignore instructions* in the prompt, and treats **system-prompt context as background knowledge** for biasing.
  4. RL with **GSPO** on about 50k utterances. SFT and RL hour counts are not given.
  - [Qwen3-ASR report](https://arxiv.org/html/2601.21337)
- **Coverage.** 52 languages and dialects: 30 languages including Arabic, plus 22 Chinese dialects. v1 and v2 of the report differ in how they describe coverage. — [arXiv abs](https://arxiv.org/abs/2601.21337)
- **Benchmarks, 1.7B / 0.6B.**
  - LibriSpeech clean/other: 1.63/3.38 and 2.11/4.55.
  - GigaSpeech 8.45 / 8.88; VoxPopuli 9.15 / 9.96.
  - Internal ExtremeNoise 16.17 / 17.88, vs Whisper-large-v3 at 63.17.
  - Accented English dialog (16 accents) 16.07 / 16.62.
  - **No far-field benchmark is reported.**
  - [Qwen3-ASR report](https://arxiv.org/html/2601.21337)
- **Streaming.** 2 s chunks with a 5-token fallback, and the last 4 chunks left unfixed. Average error 2.69 offline vs 3.33 streaming (1.7B); 3.48 vs 4.40 (0.6B). Under vLLM, the 0.6B model reaches **92 ms TTFT** and about 2000 s of audio per second at concurrency 128. — [Qwen3-ASR report](https://arxiv.org/html/2601.21337)
- **Toolkit.** The `qwen-asr` package has transformers and vLLM backends, plus `qwen-asr-serve` (wraps `vllm serve`) and a Gradio demo. Streaming is vLLM-only, with no batch inference or timestamps in streaming mode. **Qwen3-ForcedAligner-0.6B** is a non-autoregressive aligner that timestamps arbitrary units in ≤5 min of speech in 11 languages. Native Transformers support (including torch.compile) was added on 26 Jun 2026. — [GitHub QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- **Official fine-tuning (`finetuning/qwen3_asr_sft.py`).**
  - Data: JSONL with `audio` (WAV path) and `text`, written as `"language English<asr_text>..."`. Use `language None<asr_text>` when the language is unknown.
  - Example hyperparameters: batch 32, grad_acc 4, **lr 2e-5, 1 epoch**, save every 200 steps. Supports resume and torchrun multi-GPU; the example uses 2 GPUs for 1.7B. FlashAttention-2 is recommended with bf16/fp16.
  - **The README does not mention LoRA or ms-swift**, and it shows no `context` field in the SFT data format.
  - [GitHub finetuning](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning)
- **Related.** A later "Qwen-Audio-3.0-ASR Technical Report" (arXiv 2609.07549, Sept 2026) describes an MoE LLM-based ASR trained on "tens of millions of hours" (30 languages, 16 Chinese dialects, streaming variant). Only the abstract was retrieved, and Qwen authorship/provenance was not independently confirmed. — [arXiv 2609.07549](https://arxiv.org/abs/2609.07549)
- **LCAR reranking** has been tested on Qwen3-ASR-1.7B for hallucination mitigation; see Q3. — [arXiv 2608.30776](https://arxiv.org/abs/2608.30776)

### Inferences
- Self-training data can go straight into the official SFT script's JSONL format. Pseudo-labels should keep the `language X<asr_text>` prefix so language ID does not degrade.
- To preserve context biasing during self-training, the training samples must carry system-prompt context. The public README does not show a context field, so this probably needs script modification. That is an inference.
- No official RL (GSPO) recipe is public. An RL stage would have to be built separately, for example with TRL/verl-style GRPO and a WER reward. Whether ms-swift supports Qwen3-ASR was not verified.
- The 12.5 Hz encoder and ≤5-min aligner fit long-form meeting segmentation. The forced aligner can also supply word timestamps for filtering pseudo-labels (e.g., rejecting segments with misaligned or inserted words).

### Gaps
- The SFT and RL data hours, the context-biasing data construction, and the GSPO reward definition are not given in the retrieved report text.
- ms-swift and LoRA support for Qwen3-ASR could not be confirmed from the official README.
- No published third-party self-training or RL results on Qwen3-ASR were found.
