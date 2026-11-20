# On-device real-time audio preprocessing for Apple Silicon, and how to keep only stable gains

Scope: deployable denoise/dereverb models at Vocali's budget (16 kHz mono, built-in MacBook mic, MLX GPU already busy with ASR/MT, ANE busy with the Nemotron diarizer). Also covers selective/gated enhancement, dual-path ASR fusion, significance testing, and what production systems do. Research date: 2026-11-20. Tool budget was limited. Numbers come from READMEs/abstracts unless noted, and items marked "unverified" are from background knowledge and need checking.

## 1. Candidate models and their real-time cost (params, MACs, latency, Apple ports, licenses)

### Takeaway
Only the ultra-light class (RNNoise, GTCRN, ULCNet-type, DTLN) has negligible compute next to a 1.7B ASR. DeepFilterNet3 is the strongest small model that has Apple Silicon ports with measured RTF (about 0.03 on MLX GPU, about 35x real time on Core ML). It runs at 48 kHz, though, and adds about 30–40 ms of fixed delay. Commercial SDKs (Maxine, Krisp) and Apple Voice Processing are either not available on macOS, not measurable, or bundled with AEC/AGC, so they don't fit a toggleable harness.

### Cited Findings
**GTCRN**
- 48.2K parameters and 33.0 MMACs/s per the README. The paper originally reported 23.7K params and 39.6 MMACs/s; the README says the higher parameter count includes the ERB module's non-learnable parameters. The streaming version has RTF 0.07 on an Intel i5-12400. MIT license. A streaming version is in the `stream` folder, and sherpa-onnx supports it — [GTCRN repo](https://github.com/Xiaobin-Rong/gtcrn)
- There is an MLX conversion of the sherpa-onnx `gtcrn_simple` streaming model for Apple Silicon. It has no published RTF — [agentable/gtcrn-simple-mlx](https://huggingface.co/agentable/gtcrn-simple-mlx). sherpa-onnx lists gtcrn_simple and DPDFNet among its speech-enhancement models — [sherpa-onnx SE docs](https://k2-fsa.github.io/sherpa/onnx/speech-enhancement/index.html)
- A third-party Rust port lists 48.8K params and 33.0 MMACs/s — [vpermilp/gtcrn](https://huggingface.co/vpermilp/gtcrn)
- The same group's follow-up, UL-UNAS, targets about 30 MMACs — [UL-UNAS arXiv 2503.00340](https://arxiv.org/html/2503.00340v2)

**DeepFilterNet (1/2/3)**
- DFN v1: single-thread RTF 0.19 on an i5-8250U. A look-ahead of 2 frames gives 40 ms overall latency. The slides list 1.8M params and 0.35 GMACs — [DeepFilterNet arXiv 2110.05588](https://arxiv.org/pdf/2110.05588); DFN2 RTF 0.04 on a notebook Core i5 — [DFN2 arXiv 2205.05474](https://arxiv.org/pdf/2205.05474.pdf)
- DFN3 runs at 48 kHz. Its paper reports VoiceBank PESQ 3.17 and STOI 0.944, with no ASR results — [DeepFilterNet3 paper, arXiv 2305.08227](https://ar5iv.labs.arxiv.org/html/2305.08227)
- **DFN3 MLX port:** 2,131,824 params (48 kHz, FFT 960, hop 480). On an Apple M5 Pro, forward-pass RTF is about 0.028–0.029 on GPU and 0.036–0.040 on CPU for 5–20 s clips — [DeepFilterNet3-MLX](https://huggingface.co/aufklarer/DeepFilterNet3-MLX); see also [moonbox-dev/DeepFilterNet3-MLX](https://huggingface.co/moonbox-dev/DeepFilterNet3-MLX)
- **DFN3 streaming Core ML port:** 0.264 ms of compute per unpaced hop. A 52.13 s clip processed in 1.494 s (34.9x real time) on an Apple Silicon Mac. The port has a fixed 30 ms delay that is separate from compute — [iky1e/DeepFilterNet3-Streaming-CoreML](https://huggingface.co/iky1e/DeepFilterNet3-Streaming-CoreML); also [aufklarer/DeepFilterNet3-CoreML](https://huggingface.co/aufklarer/DeepFilterNet3-CoreML/blob/main/README.md)
- Other Core ML SE ports: DPDFNet-8 48 kHz HR runs 1.55x real time on a 60 s clip, which is too slow — [DPDFNet8 Core ML](https://huggingface.co/iky1e/DPDFNet8-48kHz-HR-CoreML). There is also MossFormerGAN SE 16k on Core ML (not real-time-class, no RTF given) — [MossFormerGAN Core ML](https://huggingface.co/starkdmi/MossFormer_GAN_SE_16K_CoreML)

**RNNoise / ULCNet / Fast-ULCNet / DTLN**
- RNNoise: 0.06M params, 0.04 G/s MACs, VoiceBank WB-PESQ 2.34 ("performance is modest") — [UL-UNAS comparison table, arXiv 2503.00340](https://www.arxiv.org/pdf/2503.00340)
- ULCNet: 688K params, 0.097 GMACs, RTF 12.7% on a single Cortex-A53 core — [ULCNet arXiv 2312.08132](https://arxiv.org/pdf/2312.08132); [project page](https://fhgainr.github.io/ULCNet). Fast-ULCNet (FastGRNN in place of GRU) matches it while cutting model size by more than half and latency by 34% on average — [arXiv 2601.14925](https://arxiv.org/html/2601.14925v1). Align-ULCNet adds joint AEC and noise reduction — [arXiv 2410.13620](https://arxiv.org/html/2410.13620v1). The hybrid AEC+NR variant runs at RTF 13.1% on a 1.43 GHz A53 — [arXiv 2408.15746](https://arxiv.org/pdf/2408.15746)
- DTLN: about 987K params (2x2 LSTM, 128 units), one frame in and one frame out. It beat the DNS baseline by 0.24 MOS — [DTLN arXiv 2005.07551](https://arxiv.org/abs/2005.07551)

**Commercial / platform**
- NVIDIA Maxine AFX: denoise, dereverb, AEC, and super-resolution. BNR is a residual CRNN. It targets NVIDIA GPUs (Windows/Linux), and the docs give no audio latency figure — [Maxine AFX SDK docs](https://docs.nvidia.com/deeplearning/maxine/audio-effects-sdk/), [BNR NIM overview](https://docs.nvidia.com/nim/maxine/bnr/overview.md). The only ASR evidence is an Avaya beta with Riva captions, with no WER published — [Avaya case study PDF](https://developer.download.nvidia.com/devblogs/proviz-developer-avaya-success-story-2496642-r2-web.pdf)
- Apple Voice Processing (AUVoiceIO / `setVoiceProcessingEnabled`): it requires the input and output nodes together, and developers report crashes on macOS (err -10876 "couldn't create the aggregate device") — [Apple dev forum 128518](https://developer.apple.com/forums/thread/128518). Developers also report reduced mic/speaker gain that stays even with AGC off — [forum 733733](https://developer.apple.com/forums/thread/733733). There is no standalone noise-suppression switch, so it comes bundled with AEC/AGC (dictation developer report, cited in [forasoft write-up](https://www.forasoft.com/ship-log/spatial-audio-vpio)). macOS 14 added ducking and muted-talker APIs — [WWDC23 10235 notes](https://wwdcnotes.com/notes/wwdc23/10235)

### Inferences
- **Budget math.** GTCRN (33 MMACs/s) and RNNoise (40 MMACs/s) cost on the order of 1e-4 of the ASR's compute. They can run on the CPU (Accelerate/ONNX Runtime), so they don't compete with MLX GPU or ANE work. This is the safest place for "compute must not get worse."
- **Latency.** Algorithmic delay adds directly to caption latency. The window/hop sets it: DFN3 is about 30–40 ms, and GTCRN is likely about 32 ms (unverified: 512-point FFT at 16 kHz). Against a VAD minimum silence of 0.35 s and utterance-level ASR, 30–40 ms is small. It still has to be counted, because the constraint is "must not get worse," not "small."
- DFN3 runs at 48 kHz, so it either needs 48 kHz capture before downmix/resample or an up/down-sample around it. Vocali's capture is already 16 kHz mono. GTCRN, ULCNet, and DTLN are native 16 kHz.
- Avoid Apple VPIO in the harness. It can't be toggled per stage, it changes gain/AGC, and it may resample (one report says it only accepts 44.1 kHz, [OpenAI forum](https://community.openai.com/t/audio-notes-for-openai-realtime-on-apple-platforms/1108404)). It also can't be replicated in the PyTorch evaluation replica.
- **Enhancement also changes the VAD input.** Silero gating on enhanced audio vs raw audio is a separate, possibly larger effect. The harness should toggle "SE for VAD only" and "SE for ASR" independently.

### Gaps
- No GTCRN, RNNoise, or DTLN RTF numbers on M-series chips were found. You will need to measure them yourself.
- FRCRN params/MACs were not retrieved. From background knowledge (unverified), it is a ModelScope model of about 14M params, with complexity well above GTCRN. It was used in the polar-projection study below.
- Krisp SDK: no public latency/params or ASR numbers were found. Its license is commercial and proprietary.
- License checks needed. RNNoise is BSD-3 (unverified), DeepFilterNet is MIT/Apache-2.0 (unverified), and DTLN is MIT (unverified). GTCRN is MIT (verified above).
- No dereverb-specific real-time model with Apple ports was found. WPE is classical and could be a candidate, but no Mac RTF was found.

## 2. Evidence for selective / gated enhancement

### Takeaway
Gating works in research. Switching between raw and enhanced audio per utterance using SNR/SIR or a learned predictor, or blending them ("observation adding", OA), consistently beats "always enhance." Gains are often about 10–25% relative. With Whisper-class ASR, always-on enhancement is frequently harmful.

### Cited Findings
- Sato et al. (ICASSP 2022): a DNN predicts per input whether ASR does better on enhanced or observed audio. Learned switching was "comparable to rule-based oracle switching." Soft switching (a posterior-weighted mix of the two signals) gave up to 23% relative CER reduction vs the conventional method. The rule-based rationale is that the observed mix is better under high SIR and low SNR — [arXiv 2201.03881](https://arxiv.org/abs/2201.03881)
- Iwamoto et al. (Interspeech 2022): they split SE error into residual noise and artifacts. Artifacts are the main cause of ASR degradation. Adding back the scaled noisy observation monotonically raises SAR, and WER improves along with it — [arXiv 2201.06685](https://arxiv.org/abs/2201.06685), [ISCA](https://www.isca-archive.org/interspeech_2022/iwamoto22_interspeech.html). A follow-up proves OA monotonically improves SAR — [arXiv 2404.14860](https://arxiv.org/html/2404.14860v1)
- Polar-projection diagnosis (2026): FRCRN on VoiceBank+DEMAND. wav2vec2.0 WER goes from 17.13 to 10.20 near full correction (α=0.85). Whisper goes from 6.44 to 5.57 only at weak correction (α=0.25), and full correction (α=1) gives 6.46, which is no gain. Different utterances prefer different strengths, and a per-utterance oracle beats the best fixed α "substantially" (numbers not in the abstract) — [arXiv 2607.11157](https://arxiv.org/abs/2607.11157)
- Intelligibility-guided OA (2026): fusion weights come from the backend ASR's own intelligibility/confidence (posterior entropy). It is training-free and needs no change to SE or ASR. It improves over OA baselines across SE–ASR pairs, but the abstract gives no numbers — [arXiv 2602.20967](https://arxiv.org/abs/2602.20967)
- Dual application of SE: time-frequency reliability masks for original vs enhanced features — [arXiv 2011.03840](https://arxiv.org/pdf/2011.03840)
- Always-on SE harms modern ASR. MetricGAN+ raised semWER in all 40 configurations, by 1.1–46.6 points absolute (Whisper, Parakeet, Gemini Flash 2.0, Parrotlet; 500 medical recordings, 9 noise conditions) — [arXiv 2512.17562](https://arxiv.org/abs/2512.17562). SAM-Audio preprocessing raised Whisper WER/CER, and more so for larger Whisper models — [arXiv 2603.04710](https://arxiv.org/abs/2603.04710)

### Inferences
- For Qwen3-ASR (Whisper-like, large-data, noise-robust), the default expectation is that full-strength SE is neutral to harmful. The most promising knob is OA with a small α (for example, out = α·enh + (1−α)·raw, with α ≤ 0.3). It costs nothing beyond the SE itself, needs no ASR re-run, and is fully reproducible in the PyTorch replica. Make α a harness dimension.
- Gating signals available for free in Vocali: Silero speech probability, frame energy/estimated SNR (for example, noise-floor tracking during VAD-off segments), and the ASR's own token log-probs. DNSMOS costs an extra model. A simple SNR-threshold rule is a cheap baseline to beat.
- Gating on ASR confidence after decoding forces a second decode, which breaks the compute constraint. Prefer pre-decode predictors such as SNR or VAD statistics.

### Gaps
- Full results tables for Sato 2022, Iwamoto 2022, and 2602.20967 weren't extracted because the PDF text was not readable with the tools available. Pull the exact WER tables before citing absolute gains.
- No paper was found that uses DNSMOS specifically as an ASR gate with reported WER gains.

## 3. Dual-path ASR (raw + enhanced) with ROVER / confidence selection / LLM fusion

### Takeaway
Combining hypotheses from multiple channels or enhanced versions gives about 1.7–7% WER reductions in older CHiME-3 work. It roughly doubles ASR compute, and ROVER needs ≥3 hypotheses to vote. Under Vocali's constraint, a second ASR pass is not acceptable on the live path.

### Cited Findings
- Quality-estimation-ranked ROVER on CHiME-3 (multiple channels, including enhanced): WER reduction of 1.7% vs the best enhanced channel and 7.3% vs random-order ROVER. Hypothesis order matters, and combining all hypotheses is not optimal — [arXiv 1706.07238](https://arxiv.org/pdf/1706.07238)
- MOVER (CHiME-8, speaker-aware system combination): 9.55% / 8.51% relative tcpWER improvement over SOTA. This combines full systems, not raw vs enhanced — [arXiv 2508.05055](https://arxiv.org/html/2508.05055v1)
- LLM-based ASR error correction is a related post-hoc fusion approach — [arXiv 2409.09554](https://arxiv.org/pdf/2409.09554)

### Inferences
- Cost: 2x ASR GPU time per utterance, plus an extra source of caption rewrites if the fused output differs from the first-pass caption. Both violate the hard constraints. Possible uses: (a) an offline upper bound in the harness (oracle min-WER of raw vs enhanced, per utterance) to measure the headroom a gate could get; (b) a "final transcript" re-pass that runs only when the device is idle, separate from live captions.
- Signal-level fusion (OA, soft switching) gets part of the benefit at zero ASR cost. Prefer it.

### Gaps
- No paper was found that reports raw+enhanced dual decoding with Whisper-class models together with a compute cost.

## 4. Evaluation methodology for small effects (0.5–1 point WER)

### Takeaway
Use paired tests over the same utterances with blockwise (session/speaker) bootstrap resampling. Bisani & Ney's per-sentence bootstrap understates variance when utterances are dependent. Expect to need roughly tens of thousands of words, ideally dozens of independent sessions, to resolve 0.5 points. Only accept a step whose paired delta has a CI excluding zero on the held-out split of several corpora, with no SNR bucket regressing.

### Cited Findings
- Bisani & Ney (ICASSP 2004): bootstrap-t CIs for WER and a paired bootstrap ("probability of improvement"). Sentence-wise bootstrap gives considerably narrower CIs than speaker-level, and most reported improvements were small relative to the standard error — [BootLog page](https://www-i6.informatik.rwth-aachen.de/~bisani/bootlog.html), [DOI record](https://api.crossref.org/works/10.1109%2FICASSP.2004.1326009)
- Blockwise bootstrap (Meta, 2020): vanilla bootstrap "fails in dealing with dependent data" such as same-speaker utterances. Resample blocks instead — [Meta AI](https://ai.meta.com/research/publications/statistical-testing-on-asr-performance-via-blockwise-bootstrap/), [arXiv 1912.09508](https://arxiv.org/pdf/1912.09508). Modeling utterance dependence in ASR evaluation — [arXiv 2209.05281](https://arxiv.org/pdf/2209.05281)
- Rule of thumb: a 5,000-word set probably won't show significant differences between competitive engines at 95%, while 10,000 words might — [Speechmatics accuracy benchmarking](https://docs.speechmatics.com/tutorials/accuracy-benchmarking). With 731 utterances, differences under about 1.5 WER points couldn't be resolved by paired bootstrap — [arXiv 2610.09934](https://arxiv.org/pdf/2610.09934)
- Approximate randomization tests are preferred when test and train domains differ, and they are as powerful as parametric tests for large samples — [arXiv 2503.23576](https://arxiv.org/pdf/2503.23576)
- Simulated vs real mismatch: SE helps wav2vec2 much more than Whisper on simulated VoiceBank+DEMAND ([2607.11157](https://arxiv.org/abs/2607.11157)), and it hurts in real medical noise ([2512.17562](https://arxiv.org/abs/2512.17562)). In CHiME-8 NOTSOFAR, the single-channel track (22.2% tcpWER) was far worse than multi-channel (10.8%) for the winner — [USTC arXiv 2409.02041](https://arxiv.org/html/2409.02041v2). The CHiME-7/8 review covers front-end trends — [arXiv 2507.18161](https://arxiv.org/pdf/2507.18161)

### Inferences (proposed harness acceptance rule)
- **Statistic.** For each utterance i, compute errors e_i and reference words n_i for the baseline (A) and the candidate (B). ΔWER = Σ(e_B−e_A)/Σn. Bootstrap by resampling sessions (meetings), not utterances, about 10k times. Report the 95% CI and P(Δ<0). Remix data inherits the source meeting's session ID, which matches the existing split policy in commit 4643e1b.
- **Power estimate.** The paired-delta variance comes mostly from the utterances where A and B differ. With weak SE (OA α small), most utterances are identical, so the paired test is much more powerful than comparing two unpaired WERs. Estimate the variance empirically from a pilot: compute the session-level SD of Δ, then N_sessions ≈ (2·1.96·SD/0.5)². Speechmatics' 10k-word figure is for unpaired engine differences, and paired small-Δ tests will usually need more words but fewer sessions than that suggests. Treat that as a check, not a rule.
- **Multiple comparisons.** The search loop tests many configs. Select on the "select" split, then confirm once on the frozen "test" split, using Holm or Bonferroni correction across the final shortlist. Never re-test repeatedly on the test split.
- **Stability.** Require (a) CI excluding 0 on pooled held-out data, (b) Δ ≤ 0 point estimate on every corpus (AMI, NOTSOFAR, CHiME-6, DiPCo, AliMeeting, LibriCSS, Arabic), or leave-one-corpus-out selection, (c) no SNR/overlap bucket with a significant regression, and (d) also check the opposite effect on clean audio (for example, LibriCSS 0S), since "must not get worse" applies there too. Include latency and caption-rewrite metrics as hard non-inferiority gates.
- **Sim-to-real.** Weight decisions by real far-field corpora, not simulated noisy speech. Run final verification on Mac capture (built-in mic replay), since the device's own DSP and mic response differ from corpus mics.

### Gaps
- No published closed-form sample-size table for paired WER deltas of 0.5–1 point was found. It must be estimated from pilot variance.
- No meeting-corpus study was found that reports CIs specifically for SE-on vs SE-off.

## 5. What production meeting / ASR systems do

### Takeaway
No public source shows Zoom, Teams, Otter, or Meet putting a denoiser in front of their transcription ASR. Their public denoising is for human listening in calls. The large-ASR approach (Whisper) gets robustness from diverse noisy training data rather than from a front-end.

### Cited Findings
- Teams: ML noise suppression trained on clean speech plus more than 100 noise types. Microsoft notes that the ground truth for noise suppression is clean speech, while for ASR it is the words. Real-time suppression can't go to the cloud — [VentureBeat 2020](https://venturebeat.com/2020/04/09/microsoft-teams-ai-machine-learning-real-time-noise-suppression-typing/). It is on by default on Windows and has music detection — [Microsoft 365 blog 2022](https://www.microsoft.com/en-us/microsoft-365/blog/2022/06/13/how-microsoft-teams-uses-ai-and-machine-learning-to-improve-calls-and-meetings), [Teams blog](https://techcommunity.microsoft.com/t5/microsoft-teams-blog/expanding-background-noise-suppression-across-platforms-and/ba-p/3300366). Teams also has personalized voice isolation — [Teams blog](https://techcommunity.microsoft.com/blog/microsoftteamsblog/voice-isolation-in-microsoft-teams-enables-personalized-noise-suppression-for-ca/4096077/replies/4160069). A Microsoft-commissioned 2025 comparison with Zoom and others used P.835/DNSMOS, so treat its ranking with caution — [Teams blog 2025](https://techcommunity.microsoft.com/t5/microsoft-teams-blog/microsoft-teams-leads-in-audio-quality-echo-cancellation-and/ba-p/4416095)
- Whisper: 680k hours of weakly supervised web data. Its robustness to accents and background noise comes from scale and diversity. "Diversity in audio quality can help train a model to be robust" — [Whisper paper arXiv 2212.04356](https://arxiv.org/pdf/2212.04356)
- Meeting-ASR challenge systems that do use front-ends use separation/GSS/WPE (multi-channel), not single-channel denoising. Examples are NTT with a GSS front-end and fine-tuned Whisper ([arXiv 2409.05554](https://arxiv.org/html/2409.05554v1)), and the NOTSOFAR baseline with CSS and MVDR followed by Whisper ([USTC 2409.02041](https://arxiv.org/html/2409.02041v2))
- A macOS dictation developer replaced Apple voice processing with RNNoise before Whisper after Spatial Audio dropped segments. This is an anecdote — [forasoft write-up](https://www.forasoft.com/ship-log/spatial-audio-vpio)

### Inferences
- The industry pattern is: denoise for the human ear, and use raw or lightly processed audio plus a noise-robust ASR for transcripts. This fits Vocali's situation, since it uses a noise-robust ASR whose weights it can't change. Gains, if any, will likely come from weak/OA enhancement, VAD-path enhancement, or gated use in low-SNR buckets, not from always-on SE.

### Gaps
- No Zoom, Otter, Google Meet, or Apple Live Captions engineering source was found describing their ASR front-end. Treat their practice as unknown, not as "no denoise."
- Krisp markets "noise cancellation for transcription," but no independent WER evidence was found.
