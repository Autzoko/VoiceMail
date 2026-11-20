# Far-field front-end processing for meeting-room ASR: dereverberation, laptop arrays, platform voice processing

Scope note: these notes were researched in Nov 2026 with ~20 search/fetch calls. Several primary PDFs (CHiME-8 ISCA system papers) could not be text-extracted in this environment, so where a number comes from a search snippet rather than the read paper it is flagged "(snippet-level)". Project context assumed: MacBook 3-capsule array (~1–1.5 cm), mono average, 16 kHz, frozen Qwen3-ASR-1.7B / Audar-ASR-Turbo, streaming Sortformer diarization, raw AUHAL capture.

## Q1. WPE / NARA-WPE and neural dereverberation before modern ASR: real far-field gains, Whisper-class models, online-WPE cost

### Takeaway
WPE has a solid record with **multi-channel** input and conventional or ASR-adapted back ends: it is part of almost every CHiME-6/7/8 and NOTSOFAR multi-channel pipeline, and online RLS variants match offline gains. It does **not** have a record of helping a frozen large pretrained model (Whisper-class) on **single-channel** audio. The one direct far-field test with a Whisper-large-v3 model found single-channel NaraWPE made WER clearly worse (far-field 49.5% to 56.1%, near-field 21.6% to 35.9%). The only positive single-channel-track WPE number (NAIST, about 1.4 points absolute tcpWER) came with a WavLM/Zipformer back end trained on that pipeline, not a frozen foundation model.

### Cited Findings
- **LOTUSDIS (Thai far-field meetings, 2025), NaraWPE in front of the fine-tuned Pathumma-whisper-th-large-v3.** WPE "consistently degraded performance." Overall WER went from 38.33 to 48.00, near-field from 21.59 to 35.92, and far-field from 49.54 to 56.12. Per microphone: lavalier 22.77 to 37.14, condenser 20.40 to 34.69, JBL speaker at 2 m 26.42 to 37.00, BT at 3 m 58.15 to 63.04, BT at 10 m 64.04 to 68.32. MMSE-LSA spectral subtraction also hurt (near 24.92, far 54.55, overall 42.89). The authors attribute this to processing artifacts. — [LOTUSDIS, arXiv 2509.18722](https://arxiv.org/html/2509.18722v1)
- **LOTUSDIS, data instead of a front end.** Fine-tuning on distance-diverse data cut far-field WER from 81.6% to 49.5%, which is far more than any front end achieved. — [LOTUSDIS](https://arxiv.org/html/2509.18722v1) (snippet-level for the 81.6 figure)
- **NAIST, CHiME-8 NOTSOFAR-1.** Block-wise WPE (nara-WPE) before Conformer CSS, with a WavLM-large + Zipformer transducer back end. Reported relative tcpWER reduction vs. baseline: 11.6% in the single-channel (SC) track and 18.7% in the multi-channel (MC) track. The system runs 2–6× faster than the baseline. — [NAIST CHiME-8 paper page](https://www.isca-archive.org/chime_2024/hirano24_chime.html)
- **NAIST WPE ablation, as reported in the NOTSOFAR-1 summary.** dev-set-2 tcpWER went from 30.3% to 28.9% with WPE; the summary does not say which track. NAIST's final scores were MC 23.0 / 21.1 and SC 36.6 / 33.3 (tcpWER / tcORC WER). — [NOTSOFAR-1 summary, arXiv 2501.17304](https://arxiv.org/html/2501.17304)
- **Caroselli et al., Interspeech 2017 (Google), adaptive multi-mic WPE.** WPE improves ASR even when the acoustic models had multi-style training on noisy, reverberant speech, and the gains hold on large real-world sets. An adaptive (online) version for multichannel ASR gives "similar gains as the non-causal version", and applying dereverberation in training as well adds further gains. The abstract gives no numbers. — [Caroselli et al. 2017](https://www.isca-archive.org/interspeech_2017/caroselli17_interspeech.html)
- **Online WPE.** Offline WPE needs several iterations over the whole utterance, which is unsuitable for low latency. Heymann et al. propose frame-online DNN-WPE, using a DNN to estimate the speech PSD inside RLS-WPE. They evaluate offline, block-online (2 s blocks) and frame-by-frame settings. Performance drops as latency constraints tighten but stays better than no enhancement. — [Heymann et al., IWAENC 2018](https://groups.uni-paderborn.de/nt/pubs/2018/IWAENC_2018_Heymann_Paper.pdf)
- **Online WPE design choices.** The online forms are RLS (or Kalman) reformulations, with the PSD taken from recursive smoothing of the reverberant periodogram or from a DNN. Related work includes Nakatani & Kinoshita's frame-by-frame recursive spatio-temporal covariance updates (Interspeech 2019), joint WPE + acoustic-model training with an ASR loss ([Heymann ICASSP 2019](https://ris.uni-paderborn.de/download/12875/15334/ICASSP_2019_Heymann_Paper.pdf)), and an RLS "switching WPE" that reportedly beats plain online WPE (IEEE SPL 2021). — summarized from search results, snippet-level
- **CHiME-6 baselines.** The official CHiME-6 array front end was WPE + weighted delay-and-sum (BeamformIt), with dev/eval WER of 51.8 / 51.3. One CHiME-6 entry (CW-XMU) found WPE outperformed a CDR-based dereverberator; its WPE-only baseline was 52.04%. — [CHiME-6 overview](https://www.isca-archive.org/chime_2020/watanabe20b_chime.pdf), [CW-XMU](https://www.isca-archive.org/chime_2020/yang20_chime.pdf) (snippet-level)
- **NeMo CHiME-7/8 baseline.** Applies MIMO-WPE before diarization in 40 s windows with 2 s overlap, so it is block-wise rather than frame-online. — [CHiME-7/8 DASR review, arXiv 2507.18161](https://arxiv.org/html/2507.18161)
- **Neural single-channel enhancement before Whisper, mixed record.**
  - Demucs enhancement degraded Whisper ([Dissen et al., TASLP 2025](https://keshet.technion.ac.il/wp-content/uploads/2025/06/DissenYonashCohenKeshet25.pdf)).
  - A baseline SE front end raised Whisper WER while improving perceptual metrics ([arXiv 2507.07631](https://arxiv.org/html/2507.07631)).
  - Single-channel SE "often distorts speech more than it removes noise". Observation addition (mixing enhanced and noisy signals) helps; experiments use Whisper base ([arXiv 2501.02452](https://arxiv.org/html/2501.02452v1)).
  - Samsung's Whisper-Fest, a learned single-channel far-field enhancer, reports −64.7% relative WER on VOiCES far-field and −30.3% on AMI with Whisper small.en. That figure is vendor-blog-level and used a small model ([Samsung Research blog](https://research.samsung.com/blog/WHISPER-FEST-Single-Channel-Far-Field-Enhanced-Speech-to-Text-without-Parallel-Data)).
  - Background on why artifacts hurt ASR: [Iwamoto et al., arXiv 2201.06685](https://arxiv.org/pdf/2201.06685).

### Inferences
- For a frozen Whisper-class model like Qwen3-ASR-1.7B on 1-channel input, the prior should be that single-channel WPE is neutral to harmful. Single-channel WPE can only exploit temporal prediction, and the delayed linear prediction tends to over-suppress or add artifacts on low-reverb or close speech (LOTUSDIS near-field degraded most). If tried, the safest designs are a weak/partial WPE (short taps, larger delay, mixing back a fraction of the original, as in "observation addition") with an on/off A/B on Vocali's own far-field set. A gain from multi-channel WPE + adapted ASR in CHiME should not be expected to transfer to a frozen model.
- Compute is not the obstacle: online RLS-WPE on 1–3 channels with about 10 taps × 257 bins is cheap on a CPU. Latency is about one STFT frame. Block-online (2 s) or offline WPE as used in CHiME pipelines is incompatible with Vocali's streaming constraint unless applied to the final (non-streaming) pass.

### Gaps
- I found no peer-reviewed ablation of single-channel WPE on/off in front of an unmodified Whisper-large (or comparable LLM-ASR) on AMI-SDM, NOTSOFAR-SC, CHiME-6 or DiPCo. This is the key unknown and should be measured in-house.
- I found no published RTF or latency numbers for online RLS-WPE on a laptop CPU.
- REVERB-challenge numbers with modern models were not retrieved.

## Q2. Front ends of top NOTSOFAR-1 (CHiME-8) and CHiME-7/8 DASR systems, especially single-channel

### Takeaway
In multi-channel tracks, **GSS** (guided source separation with a cACGMM spatial mixture plus MVDR) is the universal front end, usually preceded by WPE and channel selection. Neural TSE or CSS was added only as initialization or refinement of GSS, and only helped after ASR-loss fine-tuning. In the **single-channel** NOTSOFAR track GSS is impossible. The best system (USTC, 22.2% tcpWER) used CSS plus neural overlap detection and a mask-based JDS model with a heavily adapted Whisper. Others dropped separation entirely (Fano Labs; BUT/JHU target-speaker Whisper). The best MC system was about 51% relatively better than the best SC system (10.8 vs 22.2 tcpWER). Fine-tuning ASR on separated far-field audio was among the largest levers, and a frozen model cannot use it.

### Cited Findings
- **NOTSOFAR-1 best scores (tcpWER / tcORC WER).** SC: USTC 22.2 / 17.7, baseline 41.4 / 35.5. MC: USTC 10.8 / 9.5, baseline 28.3 / 24.6. MC is 51% relatively better on tcpWER and 46% on tcORC. The summary's Table 6 labels SC and MC the other way round from Tables 2 and 3; the assignment above matches the text's 51% claim. — [NOTSOFAR-1 summary, arXiv 2501.17304](https://arxiv.org/html/2501.17304)
- **Per-system front ends** — [NOTSOFAR-1 summary](https://arxiv.org/html/2501.17304):
  - USTC MC: Conformer CSS + overlap detection for diarization, then neural JDS, then GSS (with T-F initialization), with an MVDR variant.
  - STCON MC (13.8 / 11.6): amplitude normalization, WPE, MicRank channel selection, GSS (WPE + cACGMM + MVDR), and a neural TSE "GSS enhancer".
  - NTT MC (15.9 / 12.6): channel selection, WPE, GSS (cACGMM), SP-MWF.
  - NAIST: nara-WPE + Conformer CSS (MC 23.0 / 21.1; SC 36.6 / 33.3).
  - NPU-TEA: Conformer CSS (MC 18.7 / 17.0; SC 30.0 / 25.7).
  - BUT/JHU: GSS in MC (24.9 / 15.6). In SC they used no separation stage, only a Whisper-based target-speaker ASR (40.1 / 27.2).
  - USTC SC: CSS + neural overlap detection + neural JDS T-F mask model; GSS is not used because it needs spatial input.
- **USTC ablations.** T-F-initialized GSS was only marginally better than plain GSS. Replacing GSS with MVDR on JDS masks was about 3–4% relatively worse. ASR adaptation on Dev-set-2 took tcpWER from 16.57% to 9.87%, a 40% relative gain. — [NOTSOFAR-1 summary](https://arxiv.org/html/2501.17304)
- **USTC system paper.** Reports SC tcpWER of 22.989% on dev-set-2 and 22.2% on eval. ASR is a posterior fusion of three Whisper models (enhanced large-v2, enhanced large-v3, and large-v3 with more simulated data); all were fine-tuned. A 25.14 tcpWER configuration in the SC pipeline reportedly involves CSS + overlap detection + MVDR (snippet-level; how MVDR applies to 1 channel was not verified). — [USTC-NERCSLIP, arXiv 2409.02041](https://arxiv.org/html/2409.02041v2)
- **Fano Labs, SC track.** Claims "first place in the realistic single-channel track" with DCF-DS (snippet-level). It **removed the CSS module** for compute reasons and used Whisper large-v3. This ranking claim conflicts with USTC's first-place claim, perhaps because of a sub-track or constraint category; verify against the official leaderboard. — [Fano Labs, CHiME-2024](https://www.isca-archive.org/chime_2024/broughton24_chime.pdf) (PDF not text-extracted; from search snippet)
- **NJU-AALab** reports 3rd in SC with TF-GridNet CSS. Inference tweaks (non-speech filtering, merging short segments, verbatim prompting) took dev tcORC WER from 37.6% to 31.0%. — [NJU-AALab](https://www.isca-archive.org/chime_2024/hu24_chime.pdf); [NOTSOFAR-1 summary](https://arxiv.org/html/2501.17304)
- **Summary-paper learnings** — [NOTSOFAR-1 summary](https://arxiv.org/html/2501.17304):
  - GSS dominated; all three top MC systems used it.
  - CSS-based systems trailed Dia-Sep-ASR systems, and CSS fine-tuning on real meetings was unexplored.
  - ASR adaptation to separated far-field audio was among the biggest levers.
  - DER did not predict tcpWER.
  - Transient noise and laughter hurt both tracks most, and debate-style overlap hurt SC far more than MC.
- **CHiME-7/8 DASR review** — [Cornell et al., arXiv 2507.18161](https://arxiv.org/html/2507.18161):
  - "All participating teams relied on guided source separation (GSS)." DNN front ends served only as GSS initialization or refinement and gave "marginal improvements".
  - STCON's TSE refinement helped "only after fine-tuning with the ASR loss".
  - STCON tried CSS and found it "too brittle… prone to introduce speaker confusion errors."
  - Envelope-variance channel selection was used almost universally; keeping about 80% of microphones was the optimum for C7DASR. NTT added a Brouhaha C50 reverb estimate to channel selection.
  - Best CHiME-6 tcpWER still exceeds 30%.
  - The ESPnet baseline found it "critical" to include GSS-enhanced data in ASR training, so the gain depends on matched training.

### Inferences
- What is "kept" across years: WPE + channel selection + GSS/MVDR for multi-channel, distributed arrays, plus ASR adapted to that output. What was "dropped or weak": CSS (brittle, speaker confusion) and pure neural enhancement or TSE without ASR-loss fine-tuning.
- For single-channel, the competitive approaches either use separation **and** fine-tune the ASR on separated output, or skip separation and use speaker-conditioned/target-speaker ASR. Both need model changes. With frozen weights, Vocali's realistic levers are diarization quality, segmentation and decoding/prompting tricks (cf. NJU's 37.6 to 31.0 from inference tweaks), not CSS.
- This matches the project observation that far-field overlap errors are mostly the model transcribing the competing talker. That is a separation or target-speaker problem, which a dereverberation front end will not fix.

### Gaps
- There is no clean SC-track ablation of "CSS vs no CSS" with a fixed, unadapted Whisper.
- The official final SC ranking (USTC vs Fano Labs claims) was not verified.
- The DiPCo and AMI-SDM front-end ablations specific to Whisper-class models were not found. One snippet claimed AMI conformer SDM ≈47.5% vs MDM-BeamformIt ≈22.2%; it is unverified and implausibly large, so do not use it.

## Q3. Very small arrays (laptop/phone, cm spacing): beamforming vs. channel averaging; MacBook evidence

### Takeaway
I found **no published ASR measurement for a MacBook's built-in array** and no WER study for arrays of about 1–2 cm with 3 mics in a reverberant meeting room. Theory and the small-array literature say differential or superdirective beamformers *can* give directional gain at small apertures, but at the cost of white-noise amplification at low frequencies. Data-independent beamforming helps only modestly (≤11% relative on AMI, with a larger array). Apple already runs its own beamforming inside Voice Isolation / VPIO.

### Cited Findings
- Differential beamforming "can achieve the maximal directional gains with a given number of microphones and a small aperture" (MEMS wearable array paper). — [ResearchGate, wearable MEMS array](https://www.researchgate.net/publication/297646068_Wearable_speech_enhancement_system_based_on_MEMS_microphone_array_for_disabled_people) (snippet-level)
- A 7-mic MEMS array with 34 mm diameter was judged embeddable, with superdirective beamforming suited to small arrays. — [Beamforming with small diameter microphone array](https://www.researchgate.net/publication/325705402_Beamforming_with_small_diameter_microphone_array) (snippet-level)
- Training-free (data-independent) beamforming as input to multichannel E2E ASR gave up to 11% WER reduction on AMI vs a multichannel baseline without beamforming; AMI's MDM array is a 10 cm radius circle, far larger than a MacBook. — [arXiv 2509.10234](https://arxiv.org/html/2509.10234)
- Smart glasses (small array, wearer speech): beamforming + channel selection + a "differential" multi-channel ASR model gave up to 18.0% relative WER reduction vs beamforming alone. The target is the near wearer, not distant talkers, and it needs a trained model. — [arXiv 2509.14430](https://arxiv.org/pdf/2509.14430)
- The AGADIR work notes that geometry-specific beamformers break when even one mic moves, which motivates array-agnostic directional ASR (trained). — [AGADIR, arXiv 2401.10411](https://arxiv.org/html/2401.10411v1)
- MC-Whisper adds a multichannel side branch to a pretrained Whisper and reports substantial AMI distant-mic gains, but it requires training. — [MC-Whisper](https://www.researchgate.net/publication/383448174_MC-Whisper_Extending_Speech_Foundation_Models_to_Multichannel_Distant_Speech_Recognition) (snippet-level)
- A real-time 3-mic DOA implementation exists on a Pixel 3's built-in mics, showing tiny phone arrays carry usable spatial cues for localization. — [PMC8681871](https://pmc.ncbi.nlm.nih.gov/articles/PMC8681871/)
- CHiME-7/8: "Most systems relied on simple ensembling techniques to fuse information across microphones", and native multichannel processing is under-explored. Even with large distributed arrays, channel selection plus GSS was the robust choice. — [CHiME-7/8 review](https://arxiv.org/html/2507.18161)

### Inferences
- At 1–1.5 cm spacing and 16 kHz, inter-mic delays are under about 45 µs, under 1 sample. Below about 1–2 kHz, a superdirective/MVDR beam needs large gains on tiny differences, so it amplifies self-noise and calibration mismatch. Delay-and-sum at this aperture is nearly identical to averaging. Averaging is therefore close to the best *data-independent* combiner for diffuse noise or reverb with matched capsules, buying about ≤10·log10(3) ≈ 4.8 dB of uncorrelated-noise (self-noise) reduction and little against reverb or competing talkers. This is first-principles reasoning, not a measured result.
- The plausible higher-value use of the 3 channels is **spatial cues** (TDOA/DOA for the diarizer or for speaker-change or overlap detection) or a mild adaptive null toward a fixed interferer such as a PA loudspeaker. Speech-quality beamforming is unlikely to pay off, and any front end must be A/B tested since it alters input statistics for a frozen ASR.
- First check whether the AUHAL multi-channel stream is truly raw per-capsule. Some Mac models expose only a pre-beamformed or processed "MacBook Pro Microphone" device. Verify capsule correlation and inter-channel delay empirically.

### Gaps
- There is no public measurement of MacBook built-in array geometry, per-capsule access, or the ASR effect of beamforming on it.
- There is no WER study of MVDR or superdirective vs averaging for arrays with ≤2 cm aperture in rooms.

## Q4. Apple platform options: VPIO, mic modes (Voice Isolation / Wide Spectrum), developer control, measured ASR effect

### Takeaway
Apple's voice processing (AUVoiceIO/VPIO or AVAudioEngine `setVoiceProcessingEnabled`) bundles AEC, noise suppression and AGC. Opting into it hands the **user**, not the app, control of the mic mode (Standard / Voice Isolation / Wide Spectrum). Apps can read the mode and open the system picker, but cannot set it via public API. The only app-controllable knobs documented in the WWDC23 session are other-audio ducking and muted-talker detection. **I found no measurement, formal or anecdotal, of VPIO or Voice Isolation effects on third-party ASR WER.** Vocali's raw-AUHAL choice is consistent with the general finding that perceptual enhancement can hurt frozen ASR (Q1), but that is an inference.

### Cited Findings
- **Apple, WWDC23 "What's new in voice processing" (session 10235).**
  - Quote: "best-in-class audio signal processing, including echo cancellation, noise suppression, automatic gain control, among others."
  - It is available as AUVoiceIO ("also known as AUVoiceProcessingIO") or via AVAudioEngine voice-processing mode, and both "provide the same voice processing capabilities."
  - Quote: "Choosing Apple's voice processing APIs also grants users full control over the mic mode settings for your app, including Standard, Voice Isolation, and Wide Spectrum."
  - Other-audio ducking is controlled by `mEnableAdvancedDucking` (off by default) and `mDuckingLevel` (Default/Min/Mid/Max).
  - Muted-talker detection reached macOS 14. macOS also offers HAL voice-activity detection via `kAudioDevicePropertyVoiceActivityDetectionState`.
  - The session text does not document a public AGC on/off or NS on/off toggle.
  - [WWDC23 10235](https://developer.apple.com/videos/play/wwdc2023/10235/)
- **Apple Support.** Voice Isolation filters out background sounds, Wide Spectrum includes the sounds around you, and Standard is the default. The modes are selected by the user from Control Center. — [Use Mic Modes on your Mac](https://support.apple.com/guide/mac-help/use-mic-modes-on-your-mac-mchle82b42f0/mac)
- **Developer reports (anecdotal / GitHub issue).** Mode access:
  - `AVCaptureDevice.MicrophoneMode` and the `preferredMicrophoneMode` / `activeMicrophoneMode` class properties are read-only; apps can observe them via KVO and call `AVCaptureDevice.showSystemUserInterface(.microphoneModes)`.
  - Mic modes need macOS 12+ on Apple Silicon or T2 Intel Macs.
  - Chromium/getUserMedia does not expose the mode, while Firefox 128 creates a VoiceProcessingIO unit.
  - Whether a VPIO session is required for the system to apply Voice Isolation to an app is listed as an open question.
  - No measurements are reported.
  - Source: [Concord-Voice/Desktop issue #27](https://github.com/Concord-Voice/Desktop/issues/27)
- **Developer forum (anecdotal).** On iOS, mic-mode support requires adopting the AUVoiceIO audio unit. — [Apple Developer Forums thread 690152](https://developer.apple.com/forums/thread/690152)
- **Developer forum.** Thread asking how to get a device's Voice Isolation status via Core Audio (not read in full). — [Apple Developer Forums 757948](https://developer.apple.com/forums/thread/757948)
- A GitHub issue titled "Mac mic: Mic Mode picker and Voice Isolation (research and measured WER)" exists in backant-io/shepetal #75, but it returned HTTP 404 (private or deleted), so its numbers could not be verified. — [link (404)](https://github.com/backant-io/shepetal/issues/75)

### Inferences
- Because the mode is user-controlled and opaque, enabling VPIO would make Vocali's input distribution depend on a user setting and on Apple's DSP version. That is bad for reproducibility with frozen ASR models. If evaluated, record from the same session via raw AUHAL and VPIO in parallel, or use playback-rerecord, and compare WER and diarization DER per mode.
- Voice Isolation is tuned for one near talker (video calls). In a 3–6-person meeting around a laptop it plausibly attenuates distant participants, which would raise deletions and hurt diarization. This is untested and is the main risk to measure.
- VPIO's AEC is the only Apple-provided path for cancelling the Mac's *own* playback. It does not cancel an external PA system it has no reference for.
- Apple's Speech framework (SFSpeechRecognizer / SpeechAnalyzer) is a separate recognizer, not a front end. Sound Analysis is a classifier (e.g., speech, laughter, music) and could serve as a cheap non-speech filter. Neither was researched quantitatively here.

### Gaps
- There is no WER measurement (paper, blog or forum) of VPIO, Voice Isolation or Wide Spectrum vs raw capture for any third-party ASR.
- There is no public documentation of an AGC/NS-only toggle on macOS VPIO. `kAUVoiceIOProperty_BypassVoiceProcessing` and AVAudioInputNode `isVoiceProcessingAGCEnabled` / `isVoiceProcessingBypassed` exist in the SDK to my knowledge, but I did not verify them against Apple docs in this session; check the headers.

## Q5. Automatic gain control / level effects on far-field ASR; echo cancellation for PA loudspeakers

### Takeaway
The evidence on AGC is thin and mostly pre-foundation-model. Level consistency matters for older engines (up to about 10% relative WER from a gain-control strategy; 26% relative from energy normalization in an older system), and AGC patents argue for fast, artifact-free gain control in far-field VUIs. For log-Mel models with internal normalization (Whisper-style, and likely Qwen3-ASR), the main real risks are clipping and very low levels (quantization, VAD and diarization thresholds), not moderate gain differences. For PA-amplified lectures and forums, the PA signal *is* the talker's speech (delayed and reverberated), so classic reference-based AEC does not apply unless there is a far-end reference. I found no ASR study for that scenario.

### Cited Findings
- A gain control strategy ("MOST") improved WER of a Mandarin ASR system "by an average of up to 10%" at a −9 dB gain setting (non-Whisper). — [Sensors 2022, 22(8):3027](https://www.mdpi.com/1424-8220/22/8/3027)
- An older energy-normalization approach based on AGC principles, compensating mic-gain and loudness differences, reported a 26% relative ASR improvement (older, non-neural system). — [ResearchGate](https://www.researchgate.net/publication/4084650_Using_noise_reduction_and_spectral_emphasis_techniques_to_improve_ASR_performance_in_noisy_conditions) (snippet-level)
- A patent application states that back-end ASR engines are "generally sensitive to input speech level" and need consistent level, and proposes fast AGC for far-field VUIs (patent, not peer-reviewed). — [US 20200279575](https://patents.justia.com/patent/20200279575)
- In a hearing-aid gain study, an ASR system that normalized input by maximum energy made uniform gain changes irrelevant. The authors flag loudness-based normalization as unexplored. — [JASA-EL 148(3) EL227](https://pubs.aip.org/asa/jasa/article/148/3/EL227/916882/Improving-hearing-aid-gains-based-on-automatic)
- STCON's top NOTSOFAR MC system included explicit "amplitude normalization" as the first front-end step. — [NOTSOFAR-1 summary](https://arxiv.org/html/2501.17304)
- Apple VPIO includes AGC and AEC, and the AEC uses the device's own playback as reference. — [WWDC23 10235](https://developer.apple.com/videos/play/wwdc2023/10235/)

### Inferences
- For Vocali, a slow, speech-gated level normalizer, or simply a fixed headroom check with peak limiting before 16 kHz resampling, is a low-risk, near-zero-compute step. Fast-acting AGC that pumps during pauses can raise background and reverb tails and should be avoided or tested. Far talkers in a 3–6-person room can be 10–20 dB below near talkers, which matters mostly for the diarizer and VAD thresholds.
- For lectures or forums with a PA: when the laptop hears both the live talker and PA loudspeakers, the result is effectively extra reverberation and echo of the *same* speech. Mitigations are placement and, if a mixer feed is available, using the feed directly. Neither AEC nor WPE is a clear fix. This is reasoning, not a measurement.

### Gaps
- There is no study of AGC or loudness normalization on Whisper-class or LLM-ASR WER in far-field meetings.
- There is no study of ASR accuracy with PA-loudspeaker reinforcement or of AEC in lecture capture.
- I found no data on Apple VPIO's AGC time constants or target level.
