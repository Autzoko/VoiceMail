# Multi-talker, far-field meeting ASR: fixing "speaker-following" (SWAP) errors in speech-LLM ASR (2024 - Nov 2026)

Context reminder: in our own analysis, Qwen3-ASR (0.6B/1.7B) on far-field overlap often transcribes the louder interferer instead of the target ("SWAP", ~28% of overlapped utterances). The error disappears when the target is >=5 dB louder. With two simultaneous interferers the output is corrupted or words are deleted instead. Near-field audio is much less affected. These notes look for methods that condition the model on *which* speaker to follow, or that remove the interferer before ASR.

## Q1. Target-speaker ASR (enrollment- or diarization-conditioned): DiCoW, SE-DiCoW, speaker prompts for speech LLMs, SQ-Whisper

### Takeaway
Conditioning the ASR on the target speaker is the most direct fix for SWAP. Two forms have the strongest results:
- **Diarization conditioning.** DiCoW / SE-DiCoW (BUT) inject frame-level Silence/Target/Non-target/Overlap (STNO) masks into Whisper.
- **Reference-audio "voiceprint prompts" in a speech LLM.** Xiaomi-CocktailASR-1 prepends 1-4 s of the target's voice to a Qwen3-8B speech LLM. On AMI-SDM it reports 21.8% WER, against 38.2% for Qwen3-ASR-1.7B.

The main caveat is that gains shrink sharply with real rather than oracle diarization.

### Cited Findings
- **Xiaomi-CocktailASR-1 (arXiv 2609.11274, Sept 2026): design.**
  - Architecture: a 0.6B Data2Vec2-derived encoder, a linear adapter, and a Qwen3-8B base LLM.
  - Target-speaker input: reference speech (randomly 1-4 s), then 1 s of silence, then the mixed speech, concatenated. There is no separate speaker encoder and no separation module.
  - Negative samples: it is trained to output an empty string when the target is absent.
  - There is also a CoT mode, whose reasoning uses CAM++ speaker-similarity levels.
  - Training data: about 400k h of multi-speaker TS-ASR, about 600k h of single-speaker TS-ASR, and about 10k h of negatives.
  - No open weights or code are listed.
  - Source: [arXiv 2609.11274](https://arxiv.org/html/2609.11274)
- **Xiaomi-CocktailASR-1: reported numbers.**
  - Simulated mixtures (WER): LibriMix 2mix 4.11 and 3mix 12.29. Qwen3-ASR-1.7B scores 68.75 and 106.04 (it does not follow a target), and Gemini-2.5-pro scores 48.41 and 76.10.
  - Real far-field meetings (WER): AMI-SDM 21.81 and AliMeeting-Far 20.63. Qwen3-ASR-1.7B scores 38.18 and 39.64. Prior baselines are SQ-Whisper at AMI-SDM 22.0 and MC-TS-ASR at AliMeeting-Far 27.50.
  - Rejection when the target is absent: 68.5-79.6% for CocktailASR, against 0% for Qwen3-ASR.
  - The CoT mode improves LibriMix 2mix from 4.11 to 3.87.
  - NOTSOFAR and LibriCSS were not evaluated.
  - Source: [arXiv 2609.11274](https://arxiv.org/html/2609.11274)
- **DiCoW (Diarization-Conditioned Whisper, BUT; arXiv 2501.00114; journal version).**
  - Mechanism: Whisper is conditioned on frame-level STNO probabilities from diarization. FDDT (frame-level diarization-dependent transforms) applies learnable affine transforms in each encoder layer, mixed by the STNO weights. No enrollment audio is needed.
  - Recognition: Jury Award at CHiME-8 Task 2 (NOTSOFAR).
  - Sources: [arXiv 2501.00114](https://arxiv.org/pdf/2501.00114); [BUT FIT result page](https://www.fit.vut.cz/research/result/c199643/.en)
- **DiCoW: oracle vs. real diarization (secondary summary, not verified against the paper tables).**
  - NOTSOFAR-1 tcpWER: about 19.7% with oracle diarization, about 33.5% with DiariZen diarization.
  - AMI-SDM cpWER: 17.2% oracle, 23.6% real.
  - The model is trained only on ground-truth labels, so it cannot recover from diarization misses.
  - Source: [EmergentMind summary](https://www.emergentmind.com/topics/diarization-conditioned-whisper-dicow-encoder). Treat as unverified.
- **SE-DiCoW (Self-Enrolled DiCoW, arXiv 2601.19194, Jan 2026).**
  - Self-enrollment: within each 30 s window it selects the segment where the target's diarization probability is highest. That segment feeds every encoder layer through cross-attention, so no external enrollment audio is needed. The backbone is Whisper-large-v3-turbo.
  - Results (tcpWER): 52.4% relative reduction in macro-average tcpWER vs. the original DiCoW on the EMMA MT-ASR benchmark. Libri3Mix-clean with oracle diarization: DiCoW 39.5, DiCoW v3.3 16.0, SE-DiCoW 9.7. On NOTSOFAR-1 and AMI-SDM the gain over DiCoW v3.3 is only about 0.2 absolute.
  - Diarization dependence: performance degrades with real DiariZen diarization. DiariZen's two-speaker powerset limit consistently misses one speaker in 3-speaker mixtures.
  - Robustness of enrollment: an enrollment segment that itself contains overlap barely hurts (target-only 9.67, 25% overlap 9.61, 100% overlap 9.87).
  - Open code and models: [TS-ASR-Whisper (training)](https://github.com/BUTSpeechFIT/TS-ASR-Whisper), [DiCoW (inference)](https://github.com/BUTSpeechFIT/DiCoW), [HF SE_DiCoW](https://huggingface.co/BUT-FIT/SE_DiCoW), [HF DiCoW_v3_3](https://huggingface.co/BUT-FIT/DiCoW_v3_3), [EMMA leaderboard](https://huggingface.co/spaces/BUT-FIT/EMMA_leaderboard).
  - Source: [arXiv 2601.19194](https://arxiv.org/html/2601.19194)
- **"Grounding Spoken LLMs in Multi-Speaker Audio via Diarization Conditioning" (Polok, Cornell, Watanabe, Burget et al.; arXiv 2606.18134, 2026).** This applies the DiCoW-style diarization conditioning to a spoken LLM (code "Dixtral"; the cited resources include Voxtral and Ministral 3). The PDF body could not be parsed, so I have no numbers.
  - Sources: [arXiv 2606.18134](https://arxiv.org/pdf/2606.18134); [code](https://github.com/BUTSpeechFIT/Dixtral); [NSF-QA dataset](https://hf.co/datasets/popcornell/NSF-QA)
- **Other 2025-2026 target-speaker ASR variants found:**
  - TS-RNNT robust to enrollment audio that itself contains interferers: [arXiv 2501.15466](https://arxiv.org/pdf/2501.15466)
  - "Speaker Targeting via Self-Speaker Adaptation" (June 2025), which uses speaker activity instead of an enrollment profile: [arXiv 2506.22646](https://arxiv.org/html/2506.22646v1)
  - Unified TS-ASR that takes text or enrollment-speech cues: [arXiv 2609.33853](https://arxiv.org/pdf/2609.33853)
  - Wake-word-guided, enrollment-free target-speaker recognition with a speech LLM: [arXiv 2611.19844](https://arxiv.org/pdf/2611.19844)
  - Enrollment-conditioned gating in LLM ASR that *suppresses* a speaker's words, i.e. the inverse control knob: [arXiv 2609.30439](https://arxiv.org/pdf/2609.30439)

### Inferences
- CocktailASR-1 uses Qwen3 as its decoder and shows a 16-point absolute WER drop on AMI-SDM against Qwen3-ASR-1.7B. That is strong evidence that a Qwen3-ASR-style speech LLM can learn to follow a target speaker from a short audio prompt plus fine-tuning on simulated overlap. It needs no architecture change, only prompt or sequence formatting: [ref audio][silence][mixture].
- For our SWAP problem, the "negative-sample rejection" training is also relevant. The model learns to output nothing rather than transcribe the interferer.
- DiCoW-style conditioning (STNO masks) is the lightest way to add "which speaker" information to an encoder. It needs a diarizer, and errors there (missed speakers, the 2-speaker powerset limit) propagate directly.
- A self-enrollment segment taken from the same meeting (SE-DiCoW) gives an enrollment-free path. For our setting it can be built from the target's single-speaker regions of the recording.

### Gaps
- No verified DiCoW numbers on LibriCSS or AliMeeting, and the original-paper tables were not checked; the oracle/real numbers above are from a secondary summary.
- No numbers for the Dixtral diarization-conditioned spoken LLM (PDF unreadable).
- No paper that fine-tunes Qwen3-ASR itself for TS-ASR. The SQ-Former / SQ-Whisper paper itself was not fetched; only its AMI-SDM 22.0 WER as cited by Xiaomi.

## Q2. Separation/extraction front ends and challenge-winning systems (CHiME-7/8 DASR, NOTSOFAR-1, MLC-SLM)

### Takeaway
Winning far-field systems still use a heavy modular pipeline:
1. dereverberation (WPE),
2. strong diarization (TS-VAD / NSD-MS2S),
3. diarization-guided separation (GSS, or joint diarization-separation),
4. fine-tuned Whisper ensembles.

Multi-channel input roughly halves tcpWER relative to single-channel (NOTSOFAR-1 winner: 10.8% vs 22.2%). Diarization and speaker counting are the dominant error sources.

### Cited Findings
- **NOTSOFAR-1 (CHiME-8) winner: USTC-NERCSLIP, first in both tracks.** Eval tcpWER was 22.2% single-channel and 10.8% multi-channel.
  - Front end: WPE, then overlap detection. Overlapped regions go to 3 s continuous speech separation (CSS); non-overlapped regions get MVDR. These are followed by GSS / cACGMM and "JDS" (joint diarization and separation, 12.8 s window).
  - Diarization: spectral clustering on ResNet-221 embeddings, refined with NSD-MS2S.
  - ASR: an enhanced Whisper (WavLM features, MoE, relative positional encoding), with an ensemble of three Whisper models.
  - Source: [arXiv 2409.02041](https://arxiv.org/html/2409.02041)
- **USTC-NERCSLIP dev ablations (tcpWER / DER %):**
  - Adding CSS to diarization: DER 32.65 to 23.43. Missed speech fell from 25.36 to 15.17, while false alarms rose from 4.72 to 5.90.
  - Separation: CSS+MVDR 26.68, plus overlap detection 25.14, JDS (3 s) 20.62, JDS (8 s) with diarization masks 17.47.
  - ASR training data: original 16.57, plus GSS-processed data 12.07, plus all sets 9.87.
  - Cost: about 4 days of JDS training. Inference on dev-set-2 takes about 1 h diarization, 1 h JDS, and 6 h ASR.
  - Source: [arXiv 2409.02041](https://arxiv.org/html/2409.02041)
- **Other NOTSOFAR-1 multi-channel ranks:** STCON 13.8% (rank 2) and NTT 15.9% (rank 3), as shown in the challenge summary's ranking table (the track label is unconfirmed). NAIST placed 3rd multi-channel and 4th single-channel.
  - Sources: [NOTSOFAR-1 summary, arXiv 2501.17304](https://arxiv.org/pdf/2501.17304); [NAIST, CSL 2025](https://www.sciencedirect.com/science/article/pii/S0885230825000610)
- **NTT's array-geometry-independent CHiME-8 DASR system** reports results better than the challenge best on NOTSOFAR-1 among geometry-independent systems: [arXiv 2502.09859](https://arxiv.org/abs/2502.09859); [arXiv 2409.05554](https://arxiv.org/pdf/2409.05554)
- **CHiME-7/8 DASR review (arXiv 2507.18161), main findings:**
  - GSS is still the standard front end, used by all teams. Neural separation was always combined with GSS, and "still struggles" in complex, varied setups.
  - All top systems used TS-VAD-style diarization refinement. Speaker counting in the first diarization pass matters most.
  - Eval overlap ratio: CHiME-6 26.7%, DiPCo 24.9%, Mixer6 13.9%, NOTSOFAR-1 29.6%.
  - Test-time adaptation and LLM post-processing gave only marginal gains at significant overhead.
  - STCON had the best overall C8DASR result, with NTT close behind (only 3 teams took part in C8DASR).
  - Source: [arXiv 2507.18161](https://arxiv.org/html/2507.18161v2)
- **MLC-SLM 2025 (Interspeech 2025, multilingual two-speaker conversations).**
  - Task 1 (ASR) was won by TENP at 9.60 average WER/CER. Task 2 (diarization + ASR) was won by MegaAIS at 16.53 average tcpWER/tcpCER. 78 teams registered and 25 submitted.
  - BUT was 2nd in Task 2 at 16.75% with DiariZen + DiCoW, both fine-tuned on challenge data.
  - Seewo reported 17.67% in Task 2. NTU Speechlab (5th in Task 1) cut mix error rate from 20.2% to 10.6%.
  - Sources: [ISCA summary](https://www.isca-archive.org/mlcslm_2025/mu25_mlcslm.html); [BUT system](https://www.isca-archive.org/mlcslm_2025/polok25_mlcslm.pdf); [Seewo](https://arxiv.org/pdf/2506.13300); [NTU](https://arxiv.org/pdf/2506.13339); [TEA-ASLP](https://arxiv.org/pdf/2507.18051); [DKU](https://arxiv.org/pdf/2507.09499)
- **MLC-SLM 2026 (2nd edition).**
  - Data: about 2,100 h of two-speaker conversations, 14 languages / 21 variants. Recordings are on mobile phones in quiet indoor rooms, so they are not far-field arrays.
  - Transsion reports 15.41% tcpMER, 2nd place in Task 1.
  - Sources: [overview arXiv 2609.27514](https://arxiv.org/abs/2609.27514); [Transsion](https://arxiv.org/pdf/2609.20833)

### Inferences
- The +GSS training-data effect (16.57 to 12.07 tcpWER) suggests that training or fine-tuning the ASR on *front-end-processed* audio matters as much as the front end itself. If we add separation in front of Qwen3-ASR, we should expect to need adaptation to separation artifacts.
- Multi-channel GSS/MVDR effectively raises the SIR of the target stream. Given our finding that SWAP vanishes at >=5 dB target advantage, beamforming or extraction that delivers even a few dB of SIR gain should remove most SWAPs. This is an inference that needs our own validation.
- Pipeline cost is high (multiple hours per dev set, ensembles). The DiCoW-style single-model approach won the "Jury award" for practicality.

### Gaps
- Official per-dataset C8DASR macro tcpWER/DER tables were beyond the fetched portion of the review.
- No details on the MegaAIS system.
- No quantified separation-based results on DiPCo or CHiME-6 were extracted.

## Q3. Serialized output training (SOT), multi-talker speech LLMs, diarization-aware LLM ASR (Sortformer, etc.)

### Takeaway
SOT-style speech LLMs that transcribe *all* speakers sidestep "who is the target" at decode time, but they push the problem into speaker attribution. Reported AMI-SDM numbers are 20-33% cpWER/WER, which is not clearly better than target-conditioned systems.

NVIDIA Sortformer gives a permutation-resolved (arrival-order) speaker supervision that can be plugged into ASR models.

### Cited Findings
- **LLM-based SOT (Shi et al., arXiv 2408.17431, 2024):** pretrained speech encoder + projector + LLM, trained with SOT. 9.0% WER on LibriMix, and a claimed SOTA 20.4% cpWER on AMI: [arXiv 2408.17431](https://arxiv.org/pdf/2408.17431)
- **CMT-LLM (Interspeech 2025):** SOT LLM with contextual biasing. LibriMix 7.9% and AMI-SDM 32.9% (abstract numbers): [ISCA](https://www.isca-archive.org/interspeech_2025/he25_interspeech.pdf)
- **DiCoW adapted to end-to-end SOT multi-talker ASR:** trained on NOTSOFAR, AMI and LibriMix. Reported to beat SOT baselines on synthetic mixtures: [arXiv 2510.03723](https://arxiv.org/html/2510.03723v1)
- **Diarization-aware multi-speaker LLM ASR (arXiv 2506.05796, 2025):** integrates diarization into LLM transcription, aimed at high-overlap meetings: [arXiv 2506.05796](https://arxiv.org/html/2506.05796v1)
- **GRPO with constrained exploration for joint multi-talker ASR + diarization with unknown speaker count (arXiv 2611.22627):** built on Qwen-Audio/Omni-style speech LLMs and evaluated on LibriMix and Dynamic-Mix only. AMI/AliMeeting are left as future work: [arXiv 2611.22627](https://arxiv.org/pdf/2611.22627)
- **"Train Short, Infer Long":** a speech LLM doing zero-shot streamable joint ASR + diarization on long audio: [arXiv 2511.16046](https://arxiv.org/pdf/2511.16046)
- **Other multi-talker ASR work:**
  - Speaker-agnostic activity streams: [arXiv 2510.03630](https://arxiv.org/pdf/2510.03630)
  - Hypothesis clustering and merging with speaker tokens: [arXiv 2409.15732](https://arxiv.org/pdf/2409.15732)
- **Sortformer (NVIDIA, arXiv 2409.06656):** Sortformer-MS-Canary (1.12B parameters) reports AMI-test WER 18.04% and cpWER 26.71%, against 32.94% cpWER for the 170M MS-Canary baseline. Released in NeMo: [arXiv 2409.06656](https://arxiv.org/pdf/2409.06656)
- **Streaming Sortformer (speaker cache, arrival-time ordering, 2025):** [arXiv 2507.18446](https://arxiv.org/abs/2507.18446)
- **NVIDIA multitalker-parakeet-streaming-0.6b (up to 4 speakers) with the streaming Sortformer diarizer:** a community model card reports AMI-IHM cpWER of about 19.35-23.73%. These are not official NVIDIA numbers.
  - Sources: [HF card](https://huggingface.co/myflow-ai/multitalker-parakeet-streaming-0.6b-v1-gguf); [NeMo tutorial](https://github.com/NVIDIA-NeMo/Speech/blob/main/tutorials/asr/Streaming_Multitalker_ASR.ipynb)
- **Streaming multi-speaker ASR trade-off study (2026):** [arXiv 2609.10265](https://arxiv.org/pdf/2609.10265)
- **2026 evaluation of spoken LLMs for conversational ASR:** LLM-based systems are competitive with 2 speakers but degrade as speaker count and overlap rise, while modular pipelines stay more robust. Per a search summary, not opened: [arXiv 2603.22709](https://arxiv.org/html/2603.22709v1)

### Inferences
- The 2026 finding that LLM systems "degrade as speaker count and overlap increase" matches our own result that two interferers cause corruption or deletion.
- SOT may convert SWAP errors into attribution errors rather than eliminate them. For a target-speaker use case (VoiceMail), target conditioning (Q1) is more aligned than SOT.

### Gaps
- Sudo et al., "Serialized Output Prompting" (arXiv 2509.04488): details not retrieved.
- No official NVIDIA cpWER for the streaming multitalker models on AMI-SDM.

## Q4. Using near-field + far-field channels together, channel selection, SIR-based gating

### Takeaway
Little 2024-2026 work directly fuses near-field and far-field channels for ASR. Challenges mainly use envelope-variance (EV) channel selection and ensembling, and the CHiME review calls native multi-channel processing "under-explored." Overlap-gated extraction (route only overlapped regions to target-speaker extraction) is used in 2026 systems.

### Cited Findings
- **Channel selection in CHiME DASR:**
  - Baselines use EV channel selection. In the C7 baseline, keeping 80% of microphones was the best speed/accuracy trade-off.
  - Max-to-min SDR across microphones differs by about 10 dB on Mixer6 and is also large on CHiME-6. DiPCo and NOTSOFAR are more uniform.
  - Multi-channel information was mostly used through simple ensembling.
  - Source: [arXiv 2507.18161](https://arxiv.org/html/2507.18161v2)
- **ESPnet C8 baseline:** a local EEND (up to 3 speakers, 5 s) picks the best channel. DOVER-Lap fusion across arrays was worse and much slower on CHiME-6: [arXiv 2507.18161](https://arxiv.org/html/2507.18161v2)
- **SmartGlasses Challenge 2026 (tttAI, TSA-ASR task):** runs overlap detection first and sends only overlapped regions to target-speaker extraction, leaving clean regions untouched. Running per-speaker ASR directly on overlap produced duplicated hypotheses and more insertions: [arXiv 2607.17867](https://arxiv.org/pdf/2607.17867)
- **USTC (NOTSOFAR):** also overlap-gated. Overlapped segments get CSS and non-overlapped segments get MVDR only. Overlap detection gave a 26.68 to 25.14 gain: [arXiv 2409.02041](https://arxiv.org/html/2409.02041)
- **AliMeeting** provides both 8-channel far-field arrays and headset near-field recordings, with overlap ratios of 42.27% (train) and 34.76% (eval). It is a natural testbed for near+far experiments: [CUHK-Tencent M2MeT paper](https://arxiv.org/pdf/2202.01986)
- **Xiaomi-CocktailASR-1** reports AliMeeting-near (single-speaker) 6.57 vs AliMeeting-Far (TS-ASR) 20.63, which quantifies the near/far gap: [arXiv 2609.11274](https://arxiv.org/html/2609.11274)

### Inferences
- When a headset or near-field channel is available (as in AMI-IHM or AliMeeting-near), it can serve two roles:
  - as a near-clean enrollment / self-enrollment source for target conditioning;
  - as an SIR estimator. Comparing energy on the near vs far channel per frame gives a target-dominance estimate, which could gate when to trust far-field ASR.

  Because SWAP vanishes at >=5 dB target advantage in our data, an SIR estimate below about 5 dB could trigger fallback (near-field ASR, extraction, or TS-ASR). This is our own design inference, not found in the literature.

### Gaps
- No 2024-2026 paper found that explicitly gates far-field ASR by estimated SIR, or that fuses near+far channels for a speech LLM.

## Q5. Agentic/harness detection of overlap/SWAP at inference and re-decoding

### Takeaway
I found no paper that implements an agent loop of the form "detect SWAP → verify speaker → re-decode with extraction." The building blocks all exist in published systems: overlap detection, overlap-gated extraction, target-absent rejection, speaker-similarity reasoning (CoT), and diarization-consistency checks.

### Cited Findings
- **Overlap-gated extraction:** a detect-then-extract pattern used in a 2026 challenge system: [arXiv 2607.17867](https://arxiv.org/pdf/2607.17867)
- **Overlap detection jointly trained with CSS (USTC):** frame-level binary overlap detector, 800-frame window: [arXiv 2409.02041](https://arxiv.org/html/2409.02041)
- **Xiaomi-CocktailASR-1 CoT mode:** reasons about speaker similarity (CAM++ similarity quantized to 5 levels) before transcribing. Negative-sample training gives a "target absent → empty" behaviour (rejection rate 68-80%). Both are internal self-check mechanisms: [arXiv 2609.11274](https://arxiv.org/html/2609.11274)
- **Speaker-embedding consistency is fragile under overlap:** an embedding computed from overlapped speech differs strongly from the clean target's, so verification may falsely reject. A "voice filter" (extraction) is suggested before verification: [US patent 11568878](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11568878)
- **Diarization/ASR word-level mismatch in overlap:** the diarizer can identify one speaker while the ASR outputs words from the other. A patent describes reconciling these: [US patent 12555584](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12555584)
- **"Split-when-merged" consistency scoring for word-label boundaries (diarization correction, 2025):** [arXiv 2509.18377](https://arxiv.org/html/2509.18377v1)
- **CHiME review:** LLM post-processing and test-time adaptation gave only marginal gains at high overhead: [arXiv 2507.18161](https://arxiv.org/html/2507.18161v2)

### Inferences
A practical harness for Qwen3-ASR:
1. Run an overlap detector or diarizer (e.g. DiariZen or Sortformer).
2. For utterances with overlap, word-align the hypothesis (we already use Qwen3-ForcedAligner). Then compute speaker-embedding similarity of the aligned word spans to the target's clean enrollment or self-enrollment, using extracted, not raw, audio because raw overlap embeddings are unreliable.
3. If similarity is low or the SIR estimate is below 5 dB, re-decode with a target-conditioned path: TSE/GSS front end, a DiCoW-style model, or a CocktailASR-style reference-prompted LLM.

This is a novel combination; no direct prior art was found. The CHiME finding of marginal LLM post-editing gains suggests that re-decoding from audio beats text-only correction.

### Gaps
- No published SWAP-specific detection metric or detector.
- No quantification of how often speaker-verification checks catch wrong-speaker transcripts.

## Q6. Arabic / multilingual meeting data availability

### Takeaway
No public Arabic far-field multi-party *meeting* corpus with overlap annotation was found. MLC-SLM 2025/2026 (11 → 14 languages) does not include Arabic, and its data is close-talk phone conversation rather than far-field meetings. Arabic options are conversational or dialect corpora and simulated mixtures.

### Cited Findings
- **MLC-SLM 2026:** about 2,100 h, 14 languages (English, French, German, Italian, Portuguese, Spanish, Japanese, Korean, Russian, Thai, Vietnamese, Tagalog, Urdu, Turkish). No Arabic. Mobile-phone recordings in quiet rooms.
  - Sources: [Transsion 2026](https://arxiv.org/pdf/2609.20833); [MLC-SLM 2026 overview](https://arxiv.org/abs/2609.27514); [registration announcement](https://nexdata.medium.com/free-registration-free-dataset-and-20k-prize-pool-join-the-2nd-mlc-slm-challenge-2026-5a67dfe35b94)
- **Arabic resources:**
  - Mixed Arabic cocktail-party dataset (AUB): simulated 2-speaker mixtures from Common Voice MSA: [AUB ScholarWorks](https://scholarworks.aub.edu.lb/items/1b29ba83-2299-4a99-b231-d45870baf9be)
  - ZAEBUC-Spoken: Arabic-English, multi-speaker Zoom sessions, MSA/Gulf/Egyptian, code-switching: [arXiv 2403.18182](https://arxiv.org/html/2403.18182v1)
  - Saudi ASWAT (LREC 2026): large spontaneous Saudi speech corpus, with diarization named as a use case: [ACL Anthology](https://aclanthology.org/2026.lrec-1.124/)
  - Arab Voices survey: MGB-3 (Egyptian YouTube, 15.4 h); Iraqi conversational telephone (Appen, 50 h): [arXiv 2601.13319](https://arxiv.org/pdf/2601.13319)
- **Non-Arabic multilingual meeting / diarization benchmarks:**
  - Indic DiarBench, including about 27 h of far-field meetings: [arXiv 2607.23808](https://arxiv.org/pdf/2607.23808)
  - AliMeeting (Mandarin, far+near field): [arXiv 2202.01986](https://arxiv.org/pdf/2202.01986)

### Inferences
- For Arabic far-field overlap, the realistic route is simulation: remix Arabic solo speech (e.g. Saudi ASWAT, MGB-3, Common Voice) with room impulse responses and controlled SIR, i.e. our "study C" style generator. Fine-tune and evaluate on that, with AMI/AliMeeting/NOTSOFAR as real-world proxies.

### Gaps
- The full MLC-SLM 2026 language list was not verified against the official overview, only via a participant paper.
- Overlap annotations for OMAN-SPEECH and Saudi ASWAT were not confirmed.
- No Arabic meeting-array corpus was found; one may exist but not surface in search.
