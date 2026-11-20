# Audio preprocessing effects on diarization, LID, overlap/separation, Arabic, and downstream MT

Scope note: about 25 searches and fetches, as of 2026-11. Several key PDFs (Iwamoto et al. 2022 "How Bad Are Artifacts?", and parts of the CHiME-7/8 review) could not be text-extracted. Where a number is missing, it is listed under Gaps rather than guessed. "Measured" means a number reported in the cited source. "Inference" means my reasoning, applied to the project setup: single far-field MacBook mic, streaming Sortformer, Qwen3-ASR LID and routing, CAM++ on 150 Hz HPF audio, about 28% wrong-talker transcriptions in overlap, which mostly disappear at a target-to-interferer ratio (TIR) of 5 dB or more.

## 1. Enhancement or dereverberation before neural diarization; effects on embeddings and speaker counting

### Takeaway
No source found shows that generic single-channel denoising, used as a front end to modern neural diarization on real meetings, gives a stable DER gain. The positive evidence has three limits. It is old (an LSTM denoiser with an i-vector pipeline on DIHARD-I, 2018), or it involves WPE dereverberation and multi-channel GSS, or it requires the diarizer to be fine-tuned on enhanced audio. Challenge reviews instead say that speaker counting and diarization errors "propagate catastrophically" downstream. They also say DER is a weak predictor of final WER. For speaker embeddings, the published effect of dereverberation or denoising on ECAPA EER is small and dataset-dependent, and it is sometimes negative.

### Cited Findings
- **DIHARD-I (2018).** The USTC/iFlytek system used an LSTM-based speech denoising front end and reached DER 24.56% on Track 1 and 36.05% on Track 2, placing second in both tracks. — [Sun et al., Interspeech 2018](https://www.isca-archive.org/interspeech_2018/sun18b_interspeech.pdf)
- **Same group (ICASSP 2018).** It reports "consistent and significant reductions of DER" over unprocessed noisy speech and over traditional enhancement. This was an i-vector-era pipeline, and the ablation numbers were not retrieved. — [Sun et al., ICASSP 2018](https://sprateam-ustc.github.io/Publications/sunlei2018icassp.pdf)
- **DIHARD-II/III.** The USTC enhancement became the official DIHARD-II enhancement baseline, with WebRTC SAD run on enhanced audio. The DIHARD-III USTC-NELSLIP system reached DER 11.30% (Track 1) and 16.78% (Track 2), with its models *fine-tuned on the enhanced* dev set and tested on enhanced eval. The gain is coupled to matched-condition retraining, which is not possible with frozen weights. — [DIHARD review, arXiv 2101.09624](https://arxiv.org/pdf/2101.09624); [arXiv 2103.10661](https://arxiv.org/pdf/2103.10661.pdf)
- **CHiME-7 DASR (NTT).** NTT applied WPE dereverberation before EEND-VC diarization. Its DIA1 system "significantly reduces the DER over the baseline" on the three datasets but "still has relatively high speaker confusion errors, especially for CHiME-6." This was multi-channel. — [NTT CHiME-7](https://www.chimechallenge.org/workshops/chime2023/papers/CHiME_2023_DASR_kamo.pdf)
- **CHiME-7/8 DASR review.** "Accurate diarization and, especially, speaker counting are critical": errors "propagate catastrophically through the subsequent separation and recognition stages." All best systems used target-speaker diarization refinement (TS-VAD). Neural speech separation and enhancement gave "marginal improvements over GSS" and was used only together with GSS. DER "between different microphones can vary by more than 10%." — [Cornell et al., arXiv 2507.18161](https://arxiv.org/html/2507.18161v2)
- **NOTSOFAR-1 summary.** DER is a weak predictor of tcpWER. USTC had the *highest* DER (14.19%, dev-set-2) but the best tcpWER. STCON, NTT and BUT/JHU had DERs of 7.9%, 9.72% and 10.4%. For BUT/JHU, single-channel versus multi-channel DER was 10.9% versus 10.4%. — [NOTSOFAR-1 summary, arXiv 2501.17304](https://arxiv.org/html/2501.17304v1)
- **NAIST in NOTSOFAR.** WPE dereverberation reduced dev-set-2 tcpWER from 30.3% to 28.9%. This is end-to-end WER, not DER. — [arXiv 2501.17304](https://arxiv.org/html/2501.17304v1)
- **"Once more diarization" (2024).** Re-doing speaker attribution *after* enhancement and separation significantly reduces speaker-confusion errors from the first-pass diarization. The authors argue DER is a poor metric for this, so they use cpWER. Ordering matters: in this case diarization ran on the raw signal and enhancement came afterwards. — [arXiv 2406.03155](https://arxiv.org/pdf/2406.03155); [CHiME 2020 note that enhancement is usually placed after diarization](https://www.isca-archive.org/chime_2020/boeddeker20_chime.pdf)
- **Embeddings (ECAPA-TDNN + WPE).** On VOiCES-19 dev, EER was 1.17% unprocessed versus 1.09% with WPE-single (a small gain). On the FFSVC-2020 eval set the result was mixed or slightly worse, and "all TSO-based DUNet front-ends failed" on FFSVC. Gains are dataset-dependent. — [arXiv 2112.13569](https://arxiv.org/pdf/2112.13569)
- **Embeddings (DNN enhancement before commercial speaker verification).** Enhancement gave benefits on real noisy utterances. — [Eskimez et al. 2018](https://labsites.rochester.edu/air/publications/eskimez2018frontend.pdf)
- **Embeddings (Diff-Filter, Interspeech 2023).** A diffusion-based enhancer gave gains for ECAPA only when *jointly optimized* with the verifier using an EER-based loss. The authors stress that enrollment must receive the same preprocessing. — [arXiv 2307.02244](https://arxiv.org/pdf/2307.02244)
- **pyannote.** No GitHub issue was found with measured DER for denoise-then-pyannote. A maintainer states that external VAD before the pipeline is not needed. — [pyannote discussion #1295](https://www.github.com/pyannote/pyannote-audio/discussions/1295)

### Inferences
- For a frozen streaming Sortformer, the 2018 to 2021 DIHARD gains do not transfer cleanly. Those systems were retrained or fine-tuned on enhanced audio, or they used older i-vector back-ends. A denoiser placed in front of a frozen Sortformer creates a train/test mismatch. The strongest risk is the one the reviews highlight most: speaker-count and confusion errors. Enhancement can remove the low-level spectral cues (channel and reverb tail) that let a diarizer separate two similar voices. This is speculation, with no direct measurement found.
- CAM++ voiceprints: whatever preprocessing is applied to the live audio must also be applied to the enrollment audio. Otherwise cosine scores shift ([Diff-Filter thesis point](https://hal.univ-lorraine.fr/tel-04257423/document)). The current 150 Hz HPF is mild and linear, which is the low-risk end of the spectrum.
- If dereverberation is tried, WPE is the best-supported choice. It is linear, so it introduces no nonlinear artifacts, but its effects are small (about 0.08 to 1.4 points absolute in the cited cases).
- Because DER correlates weakly with tcpWER, any experiment should report cpWER or tcpWER and speaker-count accuracy, not only DER.

### Gaps
- No 2023 to 2026 paper measured DER for real meetings with a single-channel neural denoiser in front of a *frozen* EEND, Sortformer or pyannote-3.x model.
- No NeMo or Sortformer issue thread with measurements was found.
- No published CAM++-specific study of denoising front ends was found.
- No published numbers were found on how enhancement changes speaker-count error.

## 2. Spoken language identification under noise and after enhancement, including segments under 2 s

### Takeaway
Evidence is thin. No study was found that directly measures Whisper LID, ECAPA-LID or MMS-LID accuracy *after* a speech-enhancement front end. The one 2026 paper on noisy LID explicitly argues that front-end enhancement "does not guarantee" a better language embedding, and it moves the correction into embedding space instead. Short segments are the dominant error source independent of noise.

### Cited Findings
- **Odyssey 2026 (Cao et al.).** Front-end enhancement "may improve perceptual quality but does not guarantee" a better downstream language embedding. Prior SE-for-LID approaches "often require costly retraining of the LID model". The paper proposes mapping noisy embeddings toward clean ones instead. Its numeric tables were not retrieved. — [cao26_odyssey](https://www.isca-archive.org/odyssey_2026/cao26_odyssey.pdf)
- **VoxLingua107, length dependence.** A noise-trained system had a 12.3% error rate on 0 to 5 s clips versus 6.1% on 5 to 20 s clips, about a 2x increase for short clips. — [VoxLingua107, arXiv 2011.12998](https://arxiv.org/pdf/2011.12998)
- **Short utterances generally.** LID is high-accuracy on utterances over 3 s, but on utterances of 1 s or less it "is still far from satisfactory". BERT-LID reported gains of about 6.5% on long segments and 19.9% on short segments over its baseline. — [arXiv 2203.00328](https://arxiv.org/pdf/2203.00328)
- **Noise robustness by model (dialect ID, Voxlect).** Whisper-Large degrades relatively little at SNR 15 to 25 dB. At 5 dB SNR, MMS-LID-256 drops more sharply than Whisper-Large. On short clips (6 s or less), Whisper-Large macro-F1 was 0.659 versus 0.971 on long clips for one Indic language. — [Voxlect, arXiv 2508.01691](https://arxiv.org/pdf/2508.01691)
- **Whisper LID in clean conditions.** Zero-shot Whisper LID on FLEURS underperforms supervised SOTA by 13.6%. It is also fragile to adversarial perturbation, and fooling the language detector degrades downstream multilingual ASR. — [Whisper paper](https://cdn.openai.com/papers/whisper.pdf); [Olivier & Raj, Interspeech 2023](https://www.isca-archive.org/interspeech_2023/olivier23_interspeech.pdf)
- **Arabic dialect ID, short utterances.** Conventional utterance-level dialect ID "typically degrade[s] for shorter utterances". A CTC-based streaming dialect-ID approach is proposed. — [CTC-DID, arXiv 2601.12199](https://arxiv.org/pdf/2601.12199)

### Inferences
- For Qwen3-ASR's internal LID, which is frozen, utterance length and context (aggregating the decision over several segments of the same speaker) are probably bigger levers than denoising. Denoising with nonlinear artifacts adds an out-of-distribution input to a model trained on noisy web data. Speculation.
- In overlap, LID can be "captured" by the louder interferer if the two talkers speak different languages. This is the same mechanism as the project's 28% wrong-talker effect. No published measurement was found. It is worth logging LID flips in the project's own overlap test set.

### Gaps
- No measured result was found for Whisper, Qwen3 or MMS LID accuracy with versus without a denoiser (DeepFilterNet, RNNoise, Demucs, etc.) on short segments.
- No zh/en/ar-specific LID noise study was found.

## 3. Single-channel separation and CSS before ASR versus target-speaker ASR; streaming feasibility

### Takeaway
On real single-channel meetings (NOTSOFAR-1 SC), separation-based pipelines do help, but only as joint diarization-plus-separation systems, and they are still about 2x worse than multi-channel. The best SC system reached 22.2% tcpWER versus 41.4% for the CSS baseline. Simple CSS front ends were outperformed by diarization-first designs. Separation artifacts are a documented failure mode: at high SIR the *unprocessed* mixture often transcribes better than the separated output. Target-speaker ASR (DiCoW, SE-DiCoW) keeps the raw audio and conditions on diarization; it is competitive, but it requires that specific Whisper-based model. No measured real-time separation benchmark on Apple Silicon was found.

### Cited Findings
- **NOTSOFAR-1 results.** The best single-channel tcpWER was 22.2% versus 10.8% for the best multi-channel system (about 51% relative), both by USTC. The best tcorcWER was 17.7% (SC) versus 9.5% (MC). — [NOTSOFAR-1 summary](https://arxiv.org/html/2501.17304v1)
- **NOTSOFAR-1 pipeline comparison.** Diarization, then separation, then ASR beat CSS, then ASR, then diarization in both tracks. The best CSS-based MC system (NPU-TEA) reached 18.7% versus 10.8% for the winner, with the baseline at 28.3%. The best SC approach was USTC's neural joint diarization and separation (one T-F mask per speaker), and the top-3 SC systems all used fine-tuned Whisper. — [arXiv 2501.17304](https://arxiv.org/html/2501.17304v1)
- **Separation artifacts on real data.** STCON reported *higher* recognition error on neurally separated speech despite high restored-audio quality. A neural target-speaker extractor beat GSS plus beamforming only after fine-tuning with an ASR loss and using GSS output as a reference. — [arXiv 2501.17304](https://arxiv.org/html/2501.17304v1)
- **Separation is not a standalone win.** CHiME-7/8 DASR teams "still heavily rely on GSS". Neural speech separation and enhancement was "still unable to reliably deal with complex scenarios". — [arXiv 2507.18161](https://arxiv.org/html/2507.18161v2)
- **DCF-DS (Fano Labs/USTC), single-channel NOTSOFAR-1 eval-small.** The CSS baseline scored 41.4% tcpWER. A spectral-clustering prior gave 38.02%. The DCF-DS 12.8 s window gave 34.93%. With MIMO-SE it gave 33.81 to 34.51%, and with re-clustering 31.72%. The table labels are garbled, so the exact configuration is uncertain. The paper claims single-channel SOTA on LibriCSS at 4.43% tcpWER. — [arXiv 2411.06667](https://arxiv.org/html/2411.06667v1)
- **ToTaTo (PixIT separation plus fine-tuned Whisper, no separate diarization).** It scored 41.2% tcpWER on NOTSOFAR eval, roughly at baseline. — [ToTaTo, CHiME 2024](https://www.isca-archive.org/chime_2024/kalda24_chime.pdf)
- **"Should we always separate?" and "Learning to enhance or not" (NTT, simulated CSJ plus CHiME-4 noise, CER).**
  - At SIR 0 dB, enhanced output wins by a wide margin: CER 83.9 to 31.2 at SNR 0, and 80.7 to 8.4 at SNR 20.
  - At SIR 10 dB and SNR 0, the observed mixture wins (20.4 versus 24.9).
  - At SIR 20 dB, the observed mixture wins at every SNR (14.8 vs 23.2, 6.0 vs 8.9, 5.3 vs 6.3).
  - Average CER: observed 35.8%, enhanced 14.1%, rule-based switching 12.8%, learned switching 12.4%, learned plus soft switching 11.4%, oracle soft switching 9.2%. Switching classification accuracy was 83.0%.
  - Sources: [arXiv 2201.03881](https://arxiv.org/html/2201.03881); [arXiv 2106.00949](https://arxiv.org/abs/2106.00949)
- **Artifacts versus residual noise.** Decomposing SE error shows that the *artifact* component, not residual noise, is "the main cause of performance degradation". "Observation adding", which mixes a scaled observed signal back into the enhanced one, monotonically increases signal-to-artifact ratio and improved ASR on simulated and real recordings. The numbers were not retrieved. — [Iwamoto et al., arXiv 2201.06685](https://arxiv.org/abs/2201.06685)
- **Persistence and metrics.** Artifact errors still hurt SSL-based or E2E-trained ASR back-ends ([arXiv 2311.11599](https://arxiv.org/pdf/2311.11599)). PESQ and STOI stop tracking WER once denoising is already strong ([NAaLoss, arXiv 2308.12615](https://arxiv.org/abs/2308.12615)).
- **Target-speaker ASR alternatives.**
  - DiCoW conditions Whisper on frame-level STNO masks (silence, target, non-target, overlap) taken from diarization. It does not alter the audio. With real diarization it reached 33.5% tcpWER on NOTSOFAR eval-small, and much of the error came from diarization misses. — [DiCoW, arXiv 2501.00114](https://arxiv.org/pdf/2501.00114)
  - SE-DiCoW adds self-enrollment segments, chosen from diarization where the target speaker is most active, through cross-attention. This fixes the case where fully overlapped speakers have near-identical STNO masks. It reports a 52.4% relative reduction in macro tcpWER versus DiCoW on the EMMA MT-ASR benchmark, with *oracle* diarization. — [SE-DiCoW, arXiv 2601.19194](https://arxiv.org/html/2601.19194v1)
  - Caveat: SE-DiCoW's NOTSOFAR numbers use tcpWER with a 5 s collar on a challenge subset, so they are not directly comparable to corpus-level scoring. — [arXiv 2609.39344](https://arxiv.org/pdf/2609.39344)
  - BUT/JHU's target-speaker ASR in NOTSOFAR, which skips separation, had weaker tcpWER but better tcorcWER than most CSS systems. — [arXiv 2501.17304](https://arxiv.org/html/2501.17304v1)
  - SA-DiCoW, which decodes all speakers jointly, reached 20.8% cpWER on NOTSOFAR versus 18.4% for independent-decoding DiCoW. — [arXiv 2510.03723](https://arxiv.org/pdf/2510.03723)
- **Streaming and low-latency separation.**
  - Causal Conv-TasNet loses quality without future context (8.96 dB SI-SDRi in one study) and has latency as low as about 5 ms. — [arXiv 2504.02302](https://arxiv.org/html/2504.02302v1); [TasNet](https://ar5iv.labs.arxiv.org/html/1711.00541)
  - TF-GridNet-Causal is the most accurate, but its parameters and compute "pose challenges for deployment in low-latency causal applications". TFACM (attention-cache TF-GridNet) approaches TF-GridNet-Causal at lower cost. — [arXiv 2505.13094](https://arxiv.org/html/2505.13094)
  - TF-GridNet "is not real-time even in its original form" and had to be modified to run on hearable accelerators. — [Itani et al.](https://homes.cs.washington.edu/~gshyam/Papers/neuralaids.pdf)
  - RTF below 1.0 for causal Conv-TasNet-style models was measured on an Intel Xeon CPU, not on Apple Silicon. — [arXiv 2504.02302](https://arxiv.org/html/2504.02302v1)

### Inferences
- **The project's TIR finding matches the literature.** The ≥5 dB effect parallels the NTT SIR table, where separation helps hugely at SIR 0 dB and hurts at SIR 10 to 20 dB. A separation front end applied always-on would likely *worsen* the majority of non-overlapped or high-TIR segments. This is the "switching" lesson: gate any separation on detected overlap with low estimated TIR. The project's Sortformer already gives per-frame overlap, which can drive that gate at no extra cost. This is an inference, untested in this setup.
- **Observation adding as a cheap mitigation.** Feeding ASR a mix of α·separated + (1−α)·raw, rather than separated audio alone, is the cheapest documented artifact mitigation, and it adds no ASR compute.
- **Model-weight constraint.** DiCoW and SE-DiCoW require a different, Whisper-based model, which violates the "weights cannot change" constraint for Qwen3-ASR and Audar. The audio-side analogue is to use Sortformer's speaker activity to choose which segments to send to ASR, and to use separated audio only within overlap regions.
- **Real-time separation on Apple Silicon.** A small causal Conv-TasNet (a few M parameters) is plausibly real-time on M-series CPU or ANE, given sub-1.0 RTF on a Xeon. TF-GridNet-class models are unlikely to fit the "no extra latency or compute" budget. Unmeasured.
- **Separation runs alongside the existing models.** Two-speaker separation then requires *two* ASR passes over overlap regions, which conflicts with the compute constraint unless it is restricted to overlap segments only.

### Gaps
- No single-channel LibriCSS per-overlap-ratio WER table was retrieved (the original LibriCSS paper's numbers were not fetched).
- No CoreML or MLX benchmark of any separation model was found.
- No leakage-rate measurement was found, such as the fraction of separated outputs that still contain the interferer's words, which is directly analogous to the project's 28% metric.
- No result was found for separation plus a *frozen* Qwen3-ASR or any LLM-based ASR.

## 4. Arabic and dialectal speech (Gulf, MSA, code-switching) and enhancement

### Takeaway
No study was found that examines how speech enhancement interacts specifically with Arabic phonetics (pharyngeals, emphatics, high-frequency fricative energy) or with Arabic-English code-switching. The Arabic noise-robustness literature is small, uses narrow tasks (digits, single corpora), and reports modest gains. Dialect-coverage research and noise-robustness research have stayed largely separate.

### Cited Findings
- **Self-supervised SE for Arabic ASR.** WER was 2.19% clean and 15.40% with seen noise, improving to 14.64% after SE (about 5% relative). Unseen noise gave 18.4%, improving to 16.71% after SE (about 9% relative). — [IIETA, Traitement du Signal 38(2)](https://www.iieta.org/journals/ts/paper/10.18280/ts.380212)
- **Arabic digits (ARADIGIT) with NMF/CASA denoising plus a CNN.** Babble noise at −10 to 15 dB, up to 98% accuracy. Isolated digits, not dialectal conversation. — [Springer CSSP 2025](https://link.springer.com/10.1007/s00034-025-03418-w)
- **Whisper zero-shot on Arabic.** It outperforms fine-tuned XLS-R on all tested datasets but "deteriorates significantly" on five unseen dialects, including the UAE (Gulf) dialect. — [Talafha et al., Interspeech 2023, arXiv 2306.02902](https://arxiv.org/abs/2306.02902)
- **Comparability problems.** A 2026 systematic review of Arabic and Saudi ASR and dialect ID says cross-study comparison is "restricted by heterogeneous corpora, varying label granularity, and inconsistent evaluation practices". — [Springer review 2026](https://link.springer.com/article/10.1007/s44443-026-00609-6)
- **Jais-ASR (vendor claim).** It claims greater noise robustness on 171 very-noisy internal recordings. This is a self-report, not independent. — [Jais-ASR blog](https://ifm.ai/blog/jais-asr/)
- **Arabic call-domain benchmark (telephone channel, many dialects).** — [arXiv 2403.04280](https://arxiv.org/pdf/2403.04280)
- **Arabic dialect ID robustness.** — [arXiv 2306.03789](https://arxiv.org/pdf/2306.03789)

### Inferences
- **Phonetic speculation, not measured.** Arabic pharyngeals (ħ, ʕ) and emphatics are characterized mainly by low-to-mid formant shifts (F1 raised, F2 lowered). Fricatives (s, ʃ, sˤ, θ, ð) and ħ/h carry energy in the high bands where mask-based denoisers apply the most attenuation in noisy, low-SNR bins. A denoiser that over-suppresses aperiodic, high-frequency noise-like energy could plausibly blur h/ħ/x and s/sˤ contrasts. No paper testing this was found.
- **High-pass filter cutoff.** The 150 Hz HPF should be harmless for these contrasts, since emphatic cues sit above 150 Hz. Raising the HPF cutoff toward about 300 Hz could start to affect F1 and male F0 harmonics. Speculation.
- **Code-switching speculation.** Code-switching Gulf Arabic and English means LID decisions on short segments are the binding constraint (see Q2). Enhancement is unlikely to fix this and may destabilize it.

### Gaps
- No measured enhancement-by-phoneme-class analysis for Arabic was found.
- No SADA, Mixat, Casablanca or MGB paper with an enhancement ablation was found.
- No code-switched noisy-speech study was found.

## 5. Does the ASR front end change downstream speech translation and MT quality?

### Takeaway
There is one directly relevant measured result, on simulated data. In ESPnet-SE++, SE front ends feeding a frozen speech-translation model changed BLEU substantially, and in both directions depending on the enhancer. A good enhancer recovered part of the gap; a weak one did nothing or made BLEU worse. No study was found that measures enhancement, then ASR WER, then MT BLEU in a cascade on real far-field speech.

### Cited Findings
- **ESPnet-SE++, simulated noisy-reverberant LibriTrans, ST model not fine-tuned.**
  - Clean input: 16.7 BLEU. Unprocessed noisy input: 7.3 BLEU.
  - Single-channel DCCRN: 10.1. Conv-TasNet: 7.4, essentially no gain.
  - Multi-channel iNeuBe: 13.2 to 14.6.
  - Multi-channel FaSNet: 7.7 on one set but 3.6 on the other, which is *worse than unprocessed*.
  - Sources: [arXiv 2207.09514](https://arxiv.org/pdf/2207.09514) (numbers via search snippet of the full paper; the abstract page itself confirms the benchmark setup only)
- **Cascades with noisy ASR text.** MT trained on clean text degrades noticeably on noisy ASR output. The fix proposed is on the MT side (contrastive learning), not the acoustic front end. — [EMNLP 2023](https://preview.aclanthology.org/fix_video/2023.emnlp-main.560.pdf)

### Inferences
- In a cascade, MT quality inherits ASR errors. The project's wrong-talker transcription is a *content substitution* error, not a minor WER error. It would translate fluently but wrongly, which is worse for users than garbled output. Fixing attribution (gating, TIR-aware handling) is therefore likely to matter more for MT than marginal WER gains from denoising. Inference.

### Gaps
- No study was found on real far-field meetings that measures front-end choice through to MT or ST quality (BLEU or COMET).
- No study was found on zh/en/ar translation from enhanced versus raw audio.
