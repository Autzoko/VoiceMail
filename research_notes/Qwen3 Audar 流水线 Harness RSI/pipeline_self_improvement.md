# Pipeline-level (multi-module) self-improvement loops for diarization → LID → routing → {Qwen3-ASR, Audar-ASR} → post-processing (2023 – Nov 2026)

Scope note: this file complements `Harness RSI 语音识别准确率提升/asr_self_improvement.md`. That file covers single-model ASR pseudo-labeling, GRPO/GSPO for ASR, TTA, synthetic data, Arabic ASR and Qwen3-ASR internals, and none of that is repeated here. This file covers **cross-module / cross-model** supervision, **diarization and LID self-training**, **confidence**, **active learning**, **continual adaptation**, **pipeline-level RL**, and **failure evidence**. Research budget was about 15 tool calls. Many 2025–2026 items were checked only at abstract level, and this is flagged where it applies.

## Q1. Cross-model / cross-module supervision (ASR↔ASR agreement, ASR→LID, ASR↔diarization, ensemble distillation)

### Takeaway
The most proven cross-model mechanism at industrial scale is **two-ASR agreement filtering**. AssemblyAI kept pseudo-labels only where two ASR systems agreed within 20% WER and got 1.62M hours. Newer work replaces voting/ROVER with an LLM or SpeechLLM arbiter over several ASR hypotheses. ASR→diarization supervision is mature as **post-hoc correction** (DiarizationLM, lexical speaker error correction, 15–55% relative WDER reduction). It is *not* yet mature as a training signal that feeds back into the diarizer. Diarization→ASR conditioning (DiCoW / SE-DiCoW) is the natural reverse edge. ASR→LID pseudo-labeling exists in a few papers (attention-derived frame language labels; CTC-based dialect-ID labels from a pretrained ASR) but has little published evidence.

### Cited Findings
- **AssemblyAI Universal-1: two-model agreement filter.** Two ASR models transcribe the unlabeled audio. Samples where the two transcripts differ by more than 20% WER are discarded. The stated reason is to stop the student from copying the teachers' error patterns. The result is about 1.62M hours of pseudo-labeled fine-tuning data, alongside 188k h supervised data and 12.5M h unlabeled data used only for BEST-RQ SSL pretraining. — [arXiv 2404.07341](https://arxiv.org/pdf/2404.07341v2); [AssemblyAI blog](https://www.assemblyai.com/blog/universal-1/)
- **AssemblyAI's predecessor Conformer-1** used a *single* teacher and dropped segments with word-level mean confidence < 0.8. Moving from that to a two-model agreement filter is an industrial data point that cross-model agreement was preferred over single-model confidence. — [AssemblyAI research page](https://www.assemblyai.com/research/universal-1)
- **Multi-ASR fusion with an LLM/SpeechLLM arbiter (Prakash et al., Interspeech 2025; authors include A. Stolcke).** A prompt-driven framework feeds several ASR outputs to a textual LLM or a SpeechLLM, "replacing voting or other arbitration logic" (ROVER). It reports better transcription accuracy than traditional fusion, and semi-supervised ASR trained on these pseudo-labels beats the baselines. Exact numbers were not retrieved (the PDF could not be parsed). — [ISCA abstract](https://www.isca-archive.org/interspeech_2025/prakash25_interspeech.html)
- **DiarizationLM (Google, 2024).** ASR words plus diarization speaker labels are serialized as compact text. A fine-tuned LLM then reassigns speakers, with both ASR and diarizer frozen. Reported WDER reductions differ across arXiv versions: v1 gives 25.9% relative on Fisher and 31% on Callhome-En; a later version gives 55.5% and 44.9%. The paper notes that word timings come from the decoder's probability lattice and are a source of speaker mismatches. Code is at github.com/google/speaker-id. — [arXiv 2401.03506](https://arxiv.org/html/2401.03506v5)
- **Lexical Speaker Error Correction (AWS, 2023).** A second-pass LM uses lexical context to fix speaker errors and reports 15–30% relative WDER reduction on three telephony datasets. — [arXiv 2306.09313](https://arxiv.org/pdf/2306.09313)
- **Diarization → ASR conditioning: DiCoW (BUT, CSL 2025).** Whisper is conditioned on frame-level Silence/Target/Non-target/Overlap (STNO) masks taken from diarization output, via FDDT and query-key biasing. It is evaluated on AMI, NOTSOFAR-1, Libri2Mix and LibriCSS. **SE-DiCoW (2026)** selects self-enrollment segments from the diarization output and reports a **52.4% relative macro tcpWER reduction** vs DiCoW on the EMMA MT-ASR benchmark. No pseudo-label or self-training use of DiCoW was found. — [arXiv 2501.00114](https://arxiv.org/html/2501.00114); [arXiv 2601.19194](https://arxiv.org/html/2601.19194v1); [arXiv 2510.03723](https://arxiv.org/pdf/2510.03723)
- **ASR ↔ LID mutual optimization ("Language Alignment Loss", 2024).** Each frame gets a pseudo language label from the BPE token with the highest attention weight. These labels are iteratively refined during training, and the paper calls this "a mutual optimization process between ASR and LID tasks" for code-switching. — [arXiv 2403.05887](https://arxiv.org/pdf/2403.05887)
- **ASR → dialect-ID pseudo-labels: CTC-DID (Jan 2026, Arabic).** Dialect ID is framed as limited-vocabulary CTC "ASR". Dialect-tag repetition targets come from a Language-Agnostic Heuristic or **a pretrained ASR model**. It reportedly beats fine-tuned Whisper and ECAPA-TDNN on low-resource ADI, generalizes zero-shot to Casablanca, is robust on short utterances and can stream. Numbers were not retrieved (abstract only). — [arXiv 2601.12199](https://arxiv.org/html/2601.12199v1)
- **Committee pseudo-labels for diarization.** Takashima et al. fuse clustering-based and EEND pseudo-labels with DOVER-Lap and fine-tune EEND on unlabeled target audio. They report **37.4% relative DER reduction** on CALLHOME vs the seed model (Interspeech 2021, background). The Hitachi-JHU DIHARD-III system likewise re-adapted EEND-EDA on eval-set pseudo-labels from a DOVER-Lap fused system. — [arXiv 2106.04764](https://arxiv.org/pdf/2106.04764); [arXiv 2102.01363](https://arxiv.org/pdf/2102.01363)
- **Cross-model consensus beats self-consistency as a label-free signal (text LLMs).** This is covered in the sibling note (+4.5 vs +1.5 on MATH-500). It is relevant here because Qwen3-ASR vs Audar-ASR is a natural cross-model pair. — see `asr_self_improvement.md` Q2

### Inferences
- **Generalist-vs-specialist agreement is the cheapest filter for Arabic.** Run Qwen3-ASR (which supports Arabic, with CommonVoice-ar WER of about 38 for 1.7B per the sibling note) and Audar-ASR on the same Arabic segments. Keep segments with low inter-system WER/CER, AssemblyAI-style. Because the two models are not equally strong on Arabic, agreement mostly *certifies easy segments*. Disagreement segments are where the specialist's value lies, so route them to active learning (Q4) or an LLM/SpeechLLM arbiter (Prakash 2025), not to the trash.
- **Shared architecture weakens the independence assumption.** Both models are 2.35B with the same architecture, and Audar may be initialized from a Qwen-family checkpoint. If so, their errors may be correlated: agreement is a weaker guarantee than for heterogeneous pairs such as Whisper vs Zipformer, as used in the sibling note's 2506.03681. Consider adding a third, architecturally different weak verifier, such as a CTC model or a forced aligner score.
- **Practical ASR→LID/routing loop.** Script and lexicon statistics of ASR output (Arabic script vs Latin; English tokens inside Arabic hypotheses) give word-level language pseudo-labels and code-switch boundaries. With word timestamps from a forced aligner, these become frame labels that can train the LID/router. This follows the same logic as LAL (2403.05887) and CTC-DID (2601.12199). This specific loop is my extrapolation; no paper reports it end-to-end for Arabic–English.
- **Practical ASR→diarization loop.** Run DiarizationLM-style correction to get corrected word-level speaker labels. Convert them, via word timings, into frame-level RTTM pseudo-labels, and fine-tune the segmentation model (pyannote/DiariZen) on them. The correction stage is published; the feedback stage is not (see Gaps).

### Gaps
- No published paper was found that runs a closed loop in which ASR-corrected speaker labels retrain the diarizer and the improved diarizer then improves ASR/cpWER over several rounds.
- No paper was found on knowledge distillation from a generalist+specialist ASR *ensemble back into each member* for speech LLMs in 2025–26.
- Prakash 2025 WER numbers and CTC-DID accuracy numbers were not retrieved.

## Q2. Self-training for diarization and LID specifically

### Takeaway
Diarization self-training has solid older evidence (EEND iterative PL: −37.4% rel. DER) and some 2025 evidence: audio-visual pseudo-labels in MISP 2025 (DER 15.52 → 8.18), and semi-supervised multichannel meeting diarization matching full supervision with 50% labels. I found **no 2025–26 paper reporting DER gains from pseudo-label adaptation of DiariZen, pyannote-3.x/Community-1 or Sortformer** on unlabeled meetings. One small 2026 study found that third-party pseudo-labels did **not** help pyannote fine-tuning, while plain supervised fine-tuning on a little in-domain data helped a lot. Evidence for LID/dialect self-training is thin and mostly indirect: SSL adaptive pretraining and CS-ASR self-training.

### Cited Findings
- **EEND iterative pseudo-labeling (Takashima et al., Interspeech 2021).** Iterative PL plus fine-tuning on unlabeled target audio gives −37.4% relative DER on CALLHOME. The committee variant fuses models. On DIHARD III, clustering- and EEND-derived pseudo-labels were both useful. — [arXiv 2106.04764](https://arxiv.org/pdf/2106.04764)
- **Semi-supervised multichannel meeting diarization (2023).** Labels are initialized from detected speech segments, then refined by iterative retraining and decoding. With 80% or 50% labeled data, performance is comparable to 100% labeled. The paper reports 57.01% relative DER reduction vs the clustering baseline on dev. — [arXiv 2307.08688](https://ar5iv.labs.arxiv.org/html/2307.08688)
- **MISP 2025 (audio-visual).** Visual cues generate diarization pseudo-labels, followed by retraining and iteration. Eval DER goes **15.52% → 8.18%** (−47.3% rel.). This shows that a second modality or module acting as the pseudo-labeler works well. — [arXiv 2506.02621](https://arxiv.org/html/2506.02621v1)
- **Pyannote fine-tuning, Bengali long-form (ShobdoSetu, 2026).** Fine-tuning only the segmentation sub-model of pyannote Community-1 on 10 labeled files gives DER **0.410 → 0.199** (vs raw pyannote 3.1). Adding Azure-generated pseudo-labeled files gives about 0.200, **no gain**. They deliberately mixed in only 1 pseudo-labeled file per round to limit its influence. — [arXiv 2603.19256](https://arxiv.org/pdf/2603.19256)
- **DiariZen vs pyannote under supervised fine-tuning (BUT, MLC-SLM 2025).** Fine-tuned DiariZen reaches **12.7% DER** vs 16.4% for fine-tuned pyannote, using supervised challenge data (not pseudo-labels). The DiariZen repo states that its benchmark numbers use no per-dataset domain adaptation. — [arXiv 2506.13414](https://arxiv.org/html/2506.13414v1); [DiariZen GitHub](https://github.com/BUTSpeechFIT/DiariZen)
- **Out-of-domain DiariZen.** DER is 25.4% on AMI-Mix but core error is only 9.7%, lower than pyannote. Much of the measured "error" comes from annotation conventions such as pause handling, which matters when pseudo-labels from one convention train another. — [arXiv 2609.11007](https://arxiv.org/pdf/2609.11007)
- **Segmentation-model domain adaptation (2025)** mostly helped on unseen DIHARD III, with only marginal changes in-domain. — [arXiv 2506.11605](https://arxiv.org/pdf/2506.11605)
- **M3SD (2025)** builds a large multi-scenario diarization dataset (including meetings) by pseudo-labeling unlabeled audio. It is a dataset paper, and no self-training DER deltas were found. — [arXiv 2506.14427](https://arxiv.org/html/2506.14427v2)
- **LID / dialect ID.** SSL adaptive pretraining of XLS-R for language and dialect ID reports up to **+40.1% relative accuracy** on FLEURS (Interspeech 2023). This uses unlabeled in-domain audio but is SSL continued pretraining, not pseudo-labeling. — [ISCA 2023](https://www.isca-archive.org/interspeech_2023/shahin23_interspeech.pdf)
- **Code-switching self-training evidence relevant to the LID/router:**
  - Tunisian Arabic–French–English CS ASR used self-training on 153 h of unlabeled TV audio. — [arXiv 2309.11327](https://arxiv.org/html/2309.11327v2)
  - Mandarin–English CS: noisy-student training with an LLM filter. — [arXiv 2407.04219](https://ar5iv.labs.arxiv.org/html/2407.04219)
  - Apple "Progressive Refinement" (2026) runs iterative PL over monolingual plus CS audio with two-stage bilingual training. — [arXiv 2607.05224](https://arxiv.org/pdf/2607.05224)
- **NADI 2025** covers both fine-grained Arabic dialect ID and dialectal ASR, and notes frequent code-switching with MSA and other languages. — [ACL Anthology](https://aclanthology.org/2025.arabicnlp-sharedtasks.103.pdf)

### Inferences
- For diarization, the evidence favours **(a) a small amount of supervised in-domain fine-tuning** (tens of labeled meetings give about 50% rel. DER cuts) over **(b) self-training on a single teacher's pseudo-labels**. Pseudo-labels pay off when they come from a *different, stronger view*: a fused committee (DOVER-Lap), another modality (visual in MISP), or ASR-informed lexical correction. This argues for building diarization pseudo-labels from DOVER-Lap(DiariZen, pyannote, Sortformer) plus DiarizationLM-style lexical correction, not from DiariZen alone.
- Measure diarization progress with **core error / per-convention scoring**, not only raw DER. Pause-annotation ambiguity can make a self-trained model look worse or better without real change (2609.11007).
- For Arabic dialect LID, the most label-efficient recipe suggested by the sources is: SSL continued pretraining on unlabeled in-domain audio (shahin23), then CTC-DID-style training on ASR-derived pseudo-labels (2601.12199), then active labeling of LID-ambiguous segments (Q4).

### Gaps
- No 2025–26 DER numbers for pseudo-label adaptation of DiariZen, pyannote Community-1 or NeMo Sortformer on unlabeled meetings.
- No published accuracy deltas for LID self-training on Arabic dialects or Arabic–English code-switching.
- No study was found on routing-error propagation, i.e. how LID mistakes sending Arabic audio to Qwen3 or English to Audar affect the loop.

## Q3. Confidence / uncertainty calibrated enough to drive loops; hallucination amplification

### Takeaway
Token-level confidence from ASR/speech LLMs is **systematically overconfident under noise**, which is exactly the far-field meeting condition here. For audio LLMs, semantic-level and verification-based uncertainty (semantic entropy, P(True)) beat token entropy on reasoning tasks, but **no method is reliably best for hallucination detection**. Cheap learned detectors such as attention-map probes trained on about 100 labels are promising. Cross-model agreement is the most widely used industrial filter (Q1). The sibling note's ASR-TRA paper warns that confidence or entropy *rewards* reinforce high-confidence errors.

### Cited Findings
- **Overconfidence under noise (2025).** Whisper's confidence under additive noise is studied, and some noisy-speech errors are "deceptively confident". Clean-speech calibration (e.g., temperature scaling) does not transfer automatically. — [arXiv 2509.07195](https://arxiv.org/html/2509.07195v1)
- **Uncertainty in audio-aware LLMs (Kuan, Huang, Lee, Apr 2026).** Benchmarks predictive entropy, length-normalized entropy, semantic entropy, discrete semantic entropy and P(True). Semantic-level and verification-based methods consistently beat token-level methods on general audio reasoning. On hallucination detection and unanswerable questions, effectiveness is **model- and benchmark-dependent** and conclusions do not carry over. The abstract has no AUROC numbers. — [arXiv 2604.25591](https://arxiv.org/abs/2604.25591)
- **Attention-map hallucination detector for SpeechLLMs (Findings ACL 2026).** Logistic regression over audio-attention features on Qwen2-Audio and Voxtral-3B gives up to **+0.23 PR-AUC** over uncertainty baselines on in-domain ASR, and needs only about **100 labeled examples**. — [arXiv 2604.19565](https://arxiv.org/html/2604.19565v1)
- **Confidence-guided LLM error correction (2025).** Injecting ASR word confidences into the LLM prompt gives 10% relative WER reduction vs naive LLM correction on the Speech Accessibility Project, and 47% on TORGO. This means post-processing should consume confidence, not just text. — [arXiv 2509.25048](https://arxiv.org/html/2509.25048v1)
- **TeLeS (2024).** A temporal lexeme similarity confidence score for end-to-end ASR, used both to pick the most uncertain samples for humans and to add high-confidence samples as pseudo-labels. — [arXiv 2401.03251](https://arxiv.org/pdf/2401.03251)
- **[bg] Uncertainty-driven pseudo-label filtering and calibration (2022).** ECE-based calibration and uncertainty filtering for speech pseudo-labels. — [arXiv 2211.07795](https://arxiv.org/pdf/2211.07795)
- **Interfacing LLMs with ASR via confidence (2024).** Confidence measures are used to decide when and how an LLM should touch ASR output. — [arXiv 2407.21414](https://arxiv.org/pdf/2407.21414)

### Inferences
- A filter stack that matches the evidence:
  1. Inter-model agreement (Qwen3 vs Audar on Arabic; Qwen3 vs a re-decode with augmentation for Chinese/English).
  2. Semantic-level consistency across N re-decodes or augmentations.
  3. A cheap learned hallucination probe (attention features, about 100 labels) per model.
  4. Length and repetition sanity checks.

  Token entropy alone should not gate data in far-field conditions (2509.07195).
- Calibrate per condition (near vs far-field, overlap vs not, MSA vs dialect). Calibration learned on clean read speech is unlikely to transfer.
- Diarization and LID also need confidence. Candidates are DOVER-Lap committee agreement for diarization and LID posterior margin plus ASR-script agreement for LID. No published calibrated-confidence study for these modules inside an LLM-ASR pipeline was found.

### Gaps
- No direct empirical evidence was found, for 2025–26 speech LLMs, that confidence-filtered pseudo-labels prevent *hallucination amplification* across multiple self-training rounds. The sibling note's ReHear and iterative-PL papers are the closest.
- No AUROC/ECE numbers were retrieved for Qwen3-ASR-family confidence.

## Q4. Active learning / human-in-the-loop: what to send to annotators

### Takeaway
ASR active learning consistently selects by **uncertainty plus diversity**: x-vector clustering then Bayesian batch AL; committee ∩ low-confidence; MC-dropout epistemic uncertainty over several rounds. Pure uncertainty sampling picks redundant clusters. No 2025–26 paper was found that does budgeted AL across a *whole pipeline*, prioritizing segments by module disagreement, overlap or LID ambiguity.

### Cited Findings
- **Two-stage AL for ASR (2024).** Unsupervised x-vector clustering for diversity, then supervised Bayesian batch AL for uncertainty. It also cites a method intersecting committee-selected samples with low-confidence ones, the closest ASR analogue of disagreement-based selection. — [arXiv 2406.02566](https://ar5iv.labs.arxiv.org/html/2406.02566)
- **Epistemic-uncertainty AL for accented ASR (ACL SRW 2025).** MC-dropout epistemic uncertainty drives multi-round selection to cut annotation cost for African-accented English, and the paper introduces an "Uncertainty WER" metric. — [arXiv 2306.02105](https://arxiv.org/html/2306.02105v7)
- **TeLeS-A.** Sends the most uncertain samples within budget to humans and uses high-confidence ones as pseudo-labels, i.e. a combined AL plus self-training loop. — [arXiv 2401.03251](https://arxiv.org/pdf/2401.03251)
- **BALD/disagreement vs plain uncertainty (NLP, background).** Bayesian AL by disagreement generally beats uncertainty sampling, while BatchBALD can pick redundant points. Diversity steps (farthest-point, committee + diversity) are recommended in recent non-speech work. — [arXiv 1808.05697](https://arxiv.org/pdf/1808.05697); [arXiv 2505.20956](https://arxiv.org/pdf/2505.20956v2)
- **Small labeled sets have very high value for diarization.** 10 labeled files halved pyannote DER (0.410 → 0.199), while pseudo-labels added nothing. — [arXiv 2603.19256](https://arxiv.org/pdf/2603.19256)
- **Hallucination probes need only about 100 labeled examples**, a cheap first use of annotation budget. — [arXiv 2604.19565](https://arxiv.org/html/2604.19565v1)

### Inferences
- **Pipeline-level acquisition score** (my design, consistent with the sources). Rank segments by a weighted sum of:
  - Qwen3–Audar disagreement (Arabic), or re-decode disagreement (Chinese/English)
  - DOVER-Lap diarizer disagreement or overlap ratio
  - LID posterior margin and ASR-script mismatch (code-switch ambiguity)
  - hallucination-probe score
  - domain-term (out-of-vocabulary) density

  Then take a diversity-constrained batch: cluster on speaker/x-vector or audio embeddings and cap per cluster.
- Spend the first labels where marginal value is highest per the evidence: (1) about 100 labels per model for a hallucination probe; (2) a few dozen fully annotated meetings per domain for diarization fine-tuning and calibration; (3) Arabic dialect/CS segments where generalist and specialist disagree.

### Gaps
- No budget-efficiency curves (WER/DER vs labeled hours) were found for 2025–26 speech-LLM pipelines or for disagreement-based AL across ASR models.

## Q5. Continual / online adaptation for deployed ASR (forgetting, LoRA per domain, rehearsal, vocabulary from logs)

### Takeaway
Rehearsal (experience replay) remains the most effective anti-forgetting tool for ASR. Regularizers such as EWC and MAS often fail across diverse domains. LoRA per domain is the standard parameter-efficient unit, and orthogonal-subspace LoRA offers a rehearsal-free option. No 2025–26 primary source was found on learning domain vocabulary from usage logs for LLM-ASR.

### Cited Findings
- **Clinical telephony on-device continual adaptation (2025/26 preprint).** Uses LoRA plus multi-domain experience replay: **17.1% relative target-WER improvement** and **55% less catastrophic forgetting** vs naive adaptation. An earlier version reports 47%, so figures differ across versions. The paper states that EWC and MAS "often fail" in diverse domains. — [arXiv 2512.16401](https://arxiv.org/html/2512.16401v4)
- **Rehearsal with singular-value tuning (Jan 2026).** Restates that rehearsal-based strategies remain among the most effective CL approaches, and cites ASR work in which rehearsal significantly outperforms regularization. — [arXiv 2601.18266](https://www.arxiv.org/pdf/2601.18266)
- **Rehearsal-free LoRA continual learning for Whisper (2024).** Uses prior LoRA subspaces to guide new learning (O-LoRA / orthogonal-gradient style) on Uyghur and Tibetan. — [arXiv 2408.10680](https://arxiv.org/pdf/2408.10680)
- **Diarization-guided Qwen3-ASR-1.7B adaptation (Interspeech 2026 MLC-SLM workshop).** Three stages: full SFT, then LoRA on synthetic TTS speech, then GRPO. SFT gave the largest gain, and LoRA-synthetic plus RL added robustness (details in Q6). — [arXiv 2607.08208](https://arxiv.org/abs/2607.08208)

### Inferences
- For Qwen3/Audar, use per-domain LoRA adapters (meetings/forums/lectures × near/far) on a frozen base. Every self-training round should mix replay of the original general set plus earlier-domain data, and should be gated by a fixed regression suite per domain and per language. That suite is the anti-drift guard.
- Vocabulary from logs: the Qwen3-ASR context-biasing prompt (see sibling note Q6) lets frequent entities mined from logs and post-processing corrections be injected **without weight updates**. This is a low-risk first loop before any fine-tuning. It is an inference; no source evaluates it.

### Gaps
- No primary 2025–26 source on mining domain vocabularies from deployment logs for LLM-ASR, and none of the AssemblyAI, Speechmatics or Deepgram "data flywheel" posts were retrieved with numbers in this pass.
- No head-to-head comparison of LoRA-only vs LoRA+replay was found for Qwen-family ASR.

## Q6. RL with pipeline-level reward (cpWER, attribution, entities) and reference-free rewards

### Takeaway
Pipeline-aware RL for speech LLMs appeared in 2025–26, mostly in MLC-SLM challenge systems. Qwen3-ASR-1.7B behind a diarization front end was trained with GRPO using WER/CER plus hallucination, repetition and length penalties, reaching **tcpMER 23.70 on dev, −6.83 absolute vs released Qwen-ASR-1.7B**. A speaker-attribution model trained with GRPO used a **cpWER-based attribution reward** plus speaker-count and order rewards. The reference-free reward evidence (cross-model consensus > self-consistency; confidence rewards reinforce confident errors) is in the sibling note.

### Cited Findings
- **Diarization-Guided Qwen-ASR Adaptation (arXiv 2607.08208, Interspeech 2026 MLC-SLM).**
  - Pipeline: VAD, CAMPPlus embeddings, two-speaker spectral clustering, RTTM segmentation, then Qwen3-ASR-1.7B.
  - Training: SFT → LoRA on 3-pipeline TTS augmentation → GRPO with a WER/CER reward and penalties for hallucination, repetition and length deviation.
  - Results: dev **tcpMER 23.70** (−6.83 absolute vs released model); final eval **17.97**.
  - SFT is the largest gain. No reward-hacking analysis.
  - — [arXiv 2607.08208](https://arxiv.org/abs/2607.08208)
- **HEAR "Who Said What" (arXiv 2608.29120, 2026).** The A2R model is optimized via GRPO. Its transcription reward includes a **cpWER-based attribution reward, a speaker-count reward and a speaker-order reward**, plus format and answer rewards, aiming to improve attribution while preserving general ability. — [arXiv 2608.29120](https://arxiv.org/pdf/2608.29120)
- **Seewo MLC-SLM 2025.** SFT then RL with verifiable rewards (Dr. GRPO), aimed at self-correction, for multilingual ASR and diarization+ASR tasks. — [arXiv 2506.13300](https://arxiv.org/html/2506.13300v2)
- **Target-speaker ASR with CoT + RL** in large audio-language models (cocktail-party mixtures). — [arXiv 2509.15612](https://arxiv.org/pdf/2509.15612v2)
- **MMW (smart glasses)** extends GRPO with a speaker-activity-consistency reward plus WER, a module-level consistency reward. — [arXiv 2507.05609](https://arxiv.org/pdf/2507.05609)

### Inferences
- Reward candidates for this pipeline:
  - **With references** (small labeled set): cpWER/tcpWER, entity or term accuracy, code-switch-point F1.
  - **Without references:**
    - Qwen3–Audar agreement on Arabic
    - diarizer–ASR consistency: speaker changes should align with ASR word timing gaps and lexical turn cues
    - LID–ASR script consistency
    - hallucination-probe score

  Only reference-based rewards have direct speech evidence (above). Every reference-free reward should be validated against a held-out labeled set each round to catch gaming.
- The penalties in 2607.08208 (repetition, length deviation, hallucination) are a concrete template for guarding WER-reward GRPO against degenerate outputs.

### Gaps
- No speech paper was found that uses a *reference-free* pipeline reward (cross-model agreement, LLM judge, or TTS round-trip) for diarization+ASR, and none reporting reward hacking under such rewards.
- The reward weights in 2607.08208 were not retrieved.

## Q7. Failure evidence: confirmation bias, collapse, metric gaming

### Takeaway
Direct speech-specific evidence of loop failure is sparse but consistent:
- Single-teacher pseudo-labels can add nothing (pyannote/Azure).
- Confidence can be deceptively high exactly where errors occur (noise).
- Industry moved from single-model confidence to cross-model agreement explicitly to avoid copying teacher errors.
- Uncertainty-method conclusions do not transfer to hallucination settings.
- Annotation-convention mismatches can make DER move without real change.

### Cited Findings
- AssemblyAI says the two-model agreement filter exists to keep the student from "replicating error patterns of the existing ASR models". — [arXiv 2404.07341](https://arxiv.org/pdf/2404.07341v2)
- Third-party pseudo-labels gave no DER gain over 10 labeled files for pyannote (0.200 vs 0.199), and the authors limited pseudo-label influence by design. — [arXiv 2603.19256](https://arxiv.org/pdf/2603.19256)
- "Deceptively confident" errors under noise. — [arXiv 2509.07195](https://arxiv.org/html/2509.07195v1)
- Uncertainty methods that work for audio reasoning do not reliably detect hallucinations. — [arXiv 2604.25591](https://arxiv.org/abs/2604.25591)
- DER inflated by pause-annotation conventions (DiariZen AMI-Mix: DER 25.4% vs core error 9.7%). — [arXiv 2609.11007](https://arxiv.org/pdf/2609.11007)
- Continual adaptation without replay causes forgetting that EWC/MAS often fail to prevent. — [arXiv 2512.16401](https://arxiv.org/html/2512.16401v4)
- DM-SUTA (model bank against error accumulation in continual TTA), ASR-TRA (confidence rewards reinforce confident errors) and proxy over-optimization in test-time scaling are documented in the sibling note (Q2/Q3).

### Inferences
- The guards that follow from the evidence:
  1. Heterogeneous verifiers: different architectures or modalities, not two near-copies.
  2. A frozen, never-trained-on labeled regression set per module and condition.
  3. Replay of original data in every round.
  4. Caps on the share of pseudo-labeled data per round.
  5. Tracking of core error vs convention-sensitive error.
  6. Monitoring of output-distribution drift: hypothesis length, repetition rate, language-mix ratio, speaker-count distribution.

### Gaps
- No 2025–26 paper documents multi-round **collapse** of a speech-LLM pipeline, or **cross-module error amplification** (e.g., LID errors poisoning ASR pseudo-labels that then re-train LID), with numbers.
