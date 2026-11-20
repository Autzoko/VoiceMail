# Single-channel speech enhancement / denoising before large pretrained ASR: does it help or hurt?

Scope note: I collected evidence through 2026-11. Peer-reviewed (P) and preprint (pre) status is marked where I know it. Most 2025–2026 "enhancement hurts Whisper" papers are arXiv preprints. No source I found tests Qwen3-ASR, OWSM or Canary directly with an SE front-end, so evidence for LLM-AED models is by analogy to Whisper-large-v3, which is the closest architecture (an AED trained on very large weakly supervised noisy data).

## Q1. Do off-the-shelf denoisers reduce or increase WER for Whisper-class / large ASR on real noisy speech?

### Takeaway
For large ASR models trained on noisy data (Whisper-large-v3, Google Chirp 3, Azure, Gemini, Parakeet), the evidence from 2024–2026 mostly says that blind, full-strength single-channel SE raises WER on both real and simulated noise. Not one of 5 SEs × 8 ASRs improved WER on real far-field Dutch recordings, and 40/40 configurations got worse in a medical study. The exceptions are a few strong enhancers on simulated VoiceBank+DEMAND (small gains) and older or weaker backends (wav2vec2 CTC, Kaldi hybrid, a WRConformer trained on clean data), where SE helps.

### Cited Findings
**Real recordings (most relevant to Vocali)**
- DRES (de Groot, Zhang, Martinez, Scharenborg, arXiv 2603.09725, pre, 2026) is real semi-spontaneous Dutch from 80 speakers in 4 public indoor spaces (exhibition venue, babble lunch area, open study area, creative space), recorded far-field at about 1.0–1.5 m on a 4-mic linear array. SNR was not reported. Five SEs were tested: spectral subtraction (SS), spectral noise gating (SNG), MetricGAN-OKD, and SGMSE+ trained on WSJ0-CHiME3 (SGW) or on VB-DEMAND (SGV). The authors write "none of the SE algorithms improved ASR performance, irrespective of the ASR model." Average WER %, shown as original → SS / SNG / GAN / SGW / SGV — [arXiv html](https://arxiv.org/html/2603.09725v1):
  - Whisper-large-v3: 15.8 → 16.3 / 20.1 / 17.8 / 16.5 / 18.7
  - Google Chirp 3: 11.2 → 11.9 / 16.5 / 13.1 / 12.4 / 15.3
  - Azure: 20.8 → 22.2 / 36.2 / 23.1 / 20.7 / 24.2. SGW at 20.7 is the only result that ties the original.
  - MMS: 28.5 → 31.0 / 41.5 / 32.5 / 31.0 / 33.6
  - NeMo-nl: 33.4 → 36.9 / 50.1 / 41.0 / 36.1 / 41.5
  - CGN Conformer: 21.6 → 27.1 / 94.7 / 27.7 / 24.0 / 28.2
  - The least harmful methods were the mild ones: SS, and the diffusion SGMSE+ trained on CHiME-type noise. The worst was noise gating, an aggressive time-frequency gate.
- SAM-Audio (Small, text-prompted separation used as a denoiser) feeding Whisper tiny through large-v3 (arXiv 2603.04710, pre, 2026) raised WER and CER in all 10 model×dataset configurations, even though PSNR improved (32.28 → 35.99 dB). Real Bengali YouTube vlogs (traffic, overlapping speech, music): Whisper-large-v3 WER 0.658 → 0.774 and CER 0.241 → 0.347. Simulated English MS-SNSD: tiny 0.161 → 0.266, medium 0.187 → 0.274. Caveat: some raw WERs exceed 1.0 and the paper does not explain them. Model-size trends are inconsistent. — [arXiv html](https://arxiv.org/html/2603.04710)
- Medical speech study, "When De-noising Hurts" (arXiv 2512.17562, pre, Dec 2025). MetricGAN+ (VoiceBank-trained) was applied before Whisper, Parakeet, Gemini Flash 2.0 and Parrotlet-a on 500 medical recordings under 9–10 noise conditions. Original noisy audio had lower semantic WER in all 40 configurations. Enhancement degraded semWER by 1.1 to 46.6 points absolute, and Whisper was the most sensitive model. — [arXiv abs](https://arxiv.org/abs/2512.17562); [code](https://github.com/eka-care/when-denoising-hurts)

**Simulated VoiceBank+DEMAND (additive noise at 2.5–17.5 dB SNR plus extra −7.5 to 0 dB, no reverb)**
- "Where Speech Enhancement Hurts Recognition" (arXiv 2607.11157, pre, 2026). Whisper-large-v3 average WER: clean 1.9%, noisy 6.6%. After SE: FRCRN 6.7%, MossFormerGAN 6.0% (the only enhancer that helped), MetricGAN+ 10.6%, a simple magnitude mask 11.4%, MeanFlowSE 8.2%, FlowSE 8.9%. Most enhancers improved SI-SDR, PESQ and STOI while raising Whisper WER. — [arXiv html](https://arxiv.org/html/2607.11157v1)
- In the same paper, wav2vec2-large CTC behaves the opposite way. Its noisy baseline is 17.13%, and WER falls steadily as enhancement strength rises (about 10.2% with the best global setting). Whether SE helps therefore depends on the recognizer: weaker or less noise-robust models benefit and Whisper does not. — [arXiv html](https://arxiv.org/html/2607.11157v1)

**Cases where SE helps (older or less robust backends, or a decoupled design)**
- Iwamoto et al. (Interspeech 2022, P). Denoising Conv-TasNet with a Kaldi LF-MMI hybrid ASR trained multi-condition. On real CHiME-3 et05, the raw enhanced signal did not beat the noisy observation. Observation adding did help (see Q2). — [ar5iv](https://ar5iv.labs.arxiv.org/html/2201.06685); [ISCA](https://www.isca-archive.org/interspeech_2022/iwamoto22_interspeech.html)
- Yang, Pandey & Wang (arXiv 2403.06387; CSL 2025, P). An ARN front-end trained on LibriSpeech plus a WRConformer back-end trained on clean WSJ reaches 3.32% (simu) / 4.44% (real) WER on single-channel CHiME-4 with LM and speaker adaptation. Here the back-end is not noise-robust, so SE is essential. The arXiv v1 evaluates no Whisper back-end. A search snippet claims the journal version reports consistent gains with Whisper, but I could not verify any numbers. — [arXiv html](https://arxiv.org/html/2403.06387v1); [CSL pdf](https://pnlwang.github.io/papers/YPW.csl25.pdf)
- A bridging module plus observation addition with Whisper (arXiv 2406.12699, pre). The abstract says neural SE "often introduces artifacts ... and harms ASR performance" when SE and ASR are trained independently. A lightweight network predicts how much of the original signal to add back. Search snippets (not verified in the full text) report about 2.6% relative WER gain with CMGAN and about 6.5% with DEMUCS for Whisper large-v3, and note that the highest PESQ/STOI "may not guarantee better recognition." — [arXiv abs](https://arxiv.org/abs/2406.12699)
- Demucs denoiser (Défossez et al., Interspeech 2020, P). On simulated noise at 0 dB SNR it recovered up to 51% of the WER lost to noise for a pre-Whisper ASR. — [arXiv 2006.12847](https://arxiv.org/pdf/2006.12847)

**Evaluation-methodology caution**
- "Too Good to Be True" (arXiv 2605.12107, pre, 2026). ASR-WER as an SE metric depends heavily on which ASR and which text normalization are used. Modern ASR with large-scale noisy training and an internal LM correlates more with human WER, but its noise robustness and use of context make it "uninformative" for judging acoustic enhancement. — [arXiv abs](https://arxiv.org/abs/2605.12107)

### Inferences
- Qwen3-ASR-1.7B and Audar-ASR-Turbo are LLM-based AEDs trained on huge noisy corpora, so they behave like Whisper-large-v3 or Chirp. The prior for "DeepFilterNet/RNNoise/FRCRN-style SE at full strength improves WER on real far-field rooms" is negative or neutral. The best case in the real-data study was "ties baseline."
- DRES (real, far-field, about 1–1.5 m, public indoor spaces) is the closest public analogue to MacBook-in-meeting-room. It found no gain on any of 8 systems.
- None of the denoisers studied separates competing talkers. They target stationary or non-speech noise, which is the minor error source in your own data (+9.5 vs +37.4 per 100 words). Even a perfect denoiser could recover only part of the smaller term.

### Gaps
- I found no peer-reviewed or preprint study that directly measures DeepFilterNet (2/3), RNNoise, Apple Voice Isolation, Krisp/NVIDIA Broadcast or other commercial NS in front of Whisper/Qwen-ASR on real meeting data with WER numbers.
- I found no SE→ASR results for Qwen3-ASR, OWSM or Canary. The DRES paper may cite Canary, but its 8 systems do not include Canary.
- I did not find CHiME-7/8 (NOTSOFAR, DASR) ablations isolating single-channel denoising with Whisper/NeMo back-ends. Top systems rely on separation/GSS plus diarization, not denoising. The USTC NOTSOFAR single-channel system reports 22.99% tcpWER with Whisper-large-v3 but has no enhancement on/off ablation — [arXiv 2409.02041](https://arxiv.org/pdf/2409.02041).
- No Arabic or Chinese SE→ASR studies turned up.

## Q2. Why does it hurt, and what mitigates it (observation adding, partial/ASR-aware enhancement, gating, ensembles)?

### Takeaway
The dominant cause is processing artifacts or over-suppression, not residual noise. Remixing part of the raw signal back in (observation adding, OA) is the most robust, cheapest and training-free fix. On real CHiME-3 it gave about 20% relative WER reduction, and on VB-DEMAND a mild 25% magnitude-correction setting moved Whisper from 6.44% to 5.57%. Training an adapter against the frozen ASR's own loss is the principled alternative.

### Cited Findings
- Iwamoto et al. split SE error, using orthogonal projection, into a noise component and an artifact component. "We experimentally identify the artifact component as the main cause of performance degradation". Scaling down artifacts improves ASR far more than scaling down residual noise. — [arXiv 2201.06685](https://arxiv.org/abs/2201.06685)
- Observation adding is y = ŝ + ω·x. It provably raises the signal-to-artifact ratio (SAR) monotonically under mild conditions. With ω_obs swept from 0 to 1.5, OA beat both the noisy input and the plain enhanced output. On real CHiME-3 et05 it gave "about 20% relative WER improvement" for ω in [0.3, 0.8] (Kaldi hybrid back-end). — [ar5iv](https://ar5iv.labs.arxiv.org/html/2201.06685). A journal follow-up by Ochiai, Iwamoto et al. extends the analysis — [arXiv 2404.14860](https://www.arxiv.org/pdf/2404.14860)
- The polar-projection diagnosis (arXiv 2607.11157) projects the mask M = A·e^{jφ} to A^α·e^{jγφ}:
  - Magnitude strength α is what matters. Estimated phase gives no ASR benefit, and reusing the noisy phase (γ=0) is as good or better in nearly all settings.
  - Whisper-large-v3 has a U-shaped WER curve: noisy 6.44%, best α=0.25 → 5.57% (p<0.001), and full α=1 is indistinguishable from noisy (p=0.92).
  - A per-utterance oracle α gives 3.7% for Whisper (vs 6.44%). Choosing the strength per input is therefore highly valuable, and the oracle is an upper bound for any gate.
  - At low SNR, Whisper loses mainly from over-suppression and wav2vec2 from under-suppression.
  - Calibrating α on a small validation set lowers WER without retraining. This was tested on only one corpus.
  - Source: [arXiv html](https://arxiv.org/html/2607.11157v1)
- Mechanisms proposed by the medical-ASR study: Whisper's 680k h of noisy training already gives internal robustness, and SE adds "spectral smearing, temporal discontinuities and unnatural formant transitions." — [arXiv 2512.17562](https://arxiv.org/pdf/2512.17562)
- Distribution mismatch between perceptually enhanced audio and what the recognizer saw in training is offered as a plausible but not established explanation. — [arXiv 2603.04710](https://arxiv.org/html/2603.04710)
- Learned OA weight: the bridging module (arXiv 2406.12699) predicts the observation-addition ratio from signal-level features, needs no knowledge of the ASR, and generalizes across datasets. — [arXiv abs](https://arxiv.org/abs/2406.12699)
- ASR-loss-trained front-end: Dissen, Yonash, Cohen, Keshet (Interspeech 2024; TASLP 2025, P). A small adaptation network in front of frozen Whisper is trained with Whisper's cross-entropy plus an L1 term, because ASR loss alone converges unstably. It reduced WER across domains and languages for packet loss. The TASLP version extends this to noise. I did not extract the noise numbers. — [arXiv 2406.18928](https://arxiv.org/pdf/2406.18928); [TASLP pdf](https://keshet.technion.ac.il/wp-content/uploads/2025/06/DissenYonashCohenKeshet25.pdf)

### Inferences
- If any denoiser is tried for Vocali, it should be run as `y = x + β·(ŝ − x)` with β ≈ 0.25–0.5 (equivalently OA ω ≈ 0.3–0.8), keeping the noisy phase. A small validation sweep on your own far-field remix set should choose β. Full-strength output is the configuration most likely to hurt.
- The per-utterance oracle gap (3.7% vs 5.6% global α for Whisper) suggests that an SNR- or quality-predicted gate can add value beyond a fixed β. That is a predictor-quality question, though, and I found no published gate validated with Whisper on real data.
- ROVER or ensembles over raw and enhanced inputs double ASR compute, which violates the "no worse compute" constraint. I found no recent Whisper-specific numbers for them.

### Gaps
- I found no published SNR-gated or "enhance-only-when-predicted-helpful" system evaluated with Whisper/Qwen on real data.
- I found no 2023–2026 ROVER over raw and enhanced inputs for LLM-based ASR.
- The OA real-data numbers come from a Kaldi hybrid. For Whisper there are only simulated (VB-DEMAND) partial-correction numbers.

## Q3. Do gains hold at moderate/high SNR, or only at very low SNR? Is there any stable improvement?

### Takeaway
There is no stable improvement for Whisper-class models with full-strength SE at moderate or high SNR. The Whisper gains that exist are small (about 0.6–0.9 absolute WER points on VB-DEMAND) and come from mild or partial enhancement. Classical SE gains are concentrated at very low SNR (≤0 dB) and in weak back-ends.

### Cited Findings
- VB-DEMAND (−7.5 to 17.5 dB): the only enhancer that beat noisy input for Whisper-large-v3 was MossFormerGAN (6.6 → 6.0%). FRCRN was neutral (6.7%), and 4 others hurt by 1.6–4.8 points. — [arXiv 2607.11157](https://arxiv.org/html/2607.11157v1)
- At low SNR, Whisper's errors come from over-suppression, so stronger SE is not the answer even there. The best strength stays mild (α≈0.25). — [arXiv 2607.11157](https://arxiv.org/html/2607.11157v1)
- On real far-field Dutch audio at unknown but realistic SNR, 0/40 SE×ASR combinations improved WER. — [arXiv 2603.09725](https://arxiv.org/html/2603.09725v1)
- Older ASR at 0 dB simulated: Demucs recovered up to 51% of noise-induced WER loss. — [arXiv 2006.12847](https://arxiv.org/pdf/2006.12847)

### Inferences
- Vocali's measured noise sensitivity is +9.5 per 100 words at −5 dB noise-only, which already indicates a noise-robust model. That leaves little headroom for denoising at the ≥0 dB SNRs typical of meeting rooms. Any gain is most plausible below 0 dB with non-stationary noise, and only with partial mixing.

### Gaps
- I found no per-SNR breakdown on real recordings for Whisper-class models. DRES does not report SNR.

## Q4. Hallucination and insertion effects on noise-only segments: do denoising or VAD change them?

### Takeaway
Whisper-large-v3 hallucinates on about 40% of non-speech inputs. A good neural VAD (Silero) cuts that to near zero and has by far the largest WER effect. A neural denoiser (DCCRN) did not reduce hallucinations. Gating out non-speech, not cleaning it, is what controls insertions.

### Cited Findings
- Barański et al. (ICASSP 2025, P; arXiv 2501.11378) — [arXiv html](https://arxiv.org/html/2501.11378v1):
  - Whisper-large-v3 hallucinated on 40.3% of 301,317 non-speech inferences.
  - Top outputs were "thank you" (24.8%), "thanks for watching" (10.3%) and "so" (3.8%). The top 10 outputs make up more than half of hallucinations.
  - By clip length: 1 s 52.1%, 10 s 11.6%, 30 s 62.3%.
  - Speech padded with 30 s of non-speech sound produced 44.5–46.1% detected hallucinations. With 1 s of padding the rate was 0.3–0.4%. Noise overlapping speech raised hallucinations only slightly.
  - Beam size 1 gave fewer hallucinations than beam 5 (19.5% vs 28.2% non-overlapped).
  - Mitigation results (detected hallucinations / WER, non-overlapped):

    | Configuration | Hallucinations | WER |
    |---|---|---|
    | No processing | 21.3% | 104.8% |
    | WebRTC VAD | 12.5% | 68.3% |
    | Silero VAD | 0.2% | 8.0% |
    | De-looping + "bag of hallucinations" removal | 0.0% | 17.1% |
    | Silero + de-looping + BoH | 0.0% | 6.5% |

  - DCCRN denoising did not help with hallucinations.
- Calm-Whisper (Interspeech 2025): Whisper-large-v3 hallucinates on nearly all UrbanSound8K clips. Fine-tuning 3 decoder heads cut non-speech hallucination by more than 80% with under 0.1% LibriSpeech WER degradation. It requires a weight change, so it is not applicable to Vocali. The paper notes that VAD has limits: Silero v5 has only 61% utterance-level accuracy on ESC-50. — [arXiv 2505.12969](https://arxiv.org/pdf/2505.12969)
- Hallucination Space Projection (arXiv 2609.04561, pre, 2026) and an SAE steering method (arXiv 2606.07473, pre) report non-speech hallucination rates for large-v3 of 40.3% and 86.88% respectively, the latter reduced to 27.33%. I did not verify the details. — [2609.04561](https://arxiv.org/pdf/2609.04561); [2606.07473](https://arxiv.org/pdf/2606.07473v1)

### Inferences
- For Qwen3-ASR (also an AED with an LLM decoder, which is prone to the same "fill silence with fluent text" behaviour), the high-value non-weight levers are:
  - a strong VAD or segment gate,
  - avoiding long non-speech padding inside windows,
  - text-level blacklists or de-looping of known hallucination phrases.
- Denoising has no demonstrated benefit here.
- The VAD's false-reject rate on far-field soft speech must be checked, because the Calm-Whisper paper shows VAD accuracy on non-speech classes is imperfect.

### Gaps
- I found no published hallucination measurements for Qwen3-ASR, or for Arabic or Chinese.
- No study combined SE with VAD specifically for hallucinations.

## Q5. Simple non-neural steps (loudness/peak normalization, AGC, high-pass, pre-emphasis, resampling quality): do they matter for large ASR?

### Takeaway
I found no controlled peer-reviewed WER ablation of loudness normalization, AGC, high-pass filtering, pre-emphasis or resampler quality for Whisper or Qwen-class models. Recommendations exist only as practitioner blogs. The simple step with documented benefit is mild spectral subtraction, which was the least harmful SE on real data, though still not helpful.

### Cited Findings
- In DRES, the classical methods split sharply. Spectral subtraction changed Whisper-large-v3 from 15.8 to 16.3% and Chirp from 11.2 to 11.9% (small harm). Spectral noise gating raised Whisper to 20.1%, Azure to 36.2% and the CGN Conformer to 94.7% (large harm). — [arXiv 2603.09725](https://arxiv.org/html/2603.09725v1)
- Anecdotal (not peer-reviewed): a developer blog recommends loudness-normalizing to about −16 dBFS after denoising and claims "measured WER improvements" without showing numbers — [dev.to](https://dev.to/stevecase430/clean-audio-before-whisper-how-noise-removal-improves-transcription-accuracy-with-code-pko). A StellaScript doc recommends level normalization for consistency, also without numbers — [StellaScript](https://berangerthomas.github.io/StellaScript/concepts/quality.html).
- A call-center ASR paper uses a 250 Hz high-pass in an equalization chain but reports no on/off WER (the search snippet does not identify it clearly, so I have no citation).
- Text normalization, not audio normalization, changes reported WER substantially. Whisper's normalizer lowers WER by 5.1 points absolute for English and 3.2 for Finnish, and by more for Indic languages, where the gain is partly an artifact of reference tokenization. This matters when comparing Arabic or Chinese conditions. — [arXiv 2409.02449](https://arxiv.org/pdf/2409.02449)

### Inferences
- Whisper-style log-Mel front-ends apply per-utterance log compression and normalization, so moderate global gain changes are largely absorbed. The real risks are clipping (from AGC or peak normalization overshoot) and very low levels near quantization. This is reasoning, not a sourced measurement.
- Aggressive gating or expansion, the non-neural equivalent of over-suppression, is clearly harmful (the SNG result).
- These steps are cheap to test on your own remix set. That would be the first real evidence for this question.

### Gaps
- I found no published WER ablation for AGC, pre-emphasis, high-pass cutoff or resampler (sinc vs linear) with Whisper/Qwen/OWSM/Canary.
- I found no data on Apple Voice Isolation or macOS voice-processing I/O (AUVoiceProcessing) feeding ASR. Only anecdotal discussions likely exist, and I did not find any with numbers.
