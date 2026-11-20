# Arabic ASR (MSA, dialects, Arabic–English code-switching): what works, 2023 – Nov 2026

Scope note: about 18 tool calls. Primary sources are arXiv abstracts or HTML, HF blogs, and the leaderboard GitHub. Several numbers come from vendor or community pages and are marked as such. I did not open the live HF leaderboard Space, so Qwen3-ASR's official leaderboard numbers are **not** in these notes (see Gaps).

## 1. State of the art on Arabic benchmarks

### Takeaway
On the Open Universal Arabic ASR Leaderboard (mid-2026), Arabic-specialized models lead: Cohere Transcribe Arabic averages 25.87 WER, OmniASR-LLM-7B 28.32, and Whisper-large-v3 36.86. All models are good on MSA and broadcast (MGB-2 about 14–16, Common Voice about 6–18) and poor on dialects (SADA 37–60, Casablanca 50–72). Dialects remain the bottleneck by roughly 3× in WER.

### Cited Findings
- The leaderboard's test sets are SADA (10 dialects, 10.7h), Common Voice 18.0 (12.6h), MASC clean (10.5h), MASC noisy (8.9h), MGB-2 (9.6h), and Casablanca (8 dialects, 7.7h). — [Leaderboard GitHub](https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard)
- Models added over time: Qwen3-ASR-0.6B/1.7B and VibeVoice-ASR-7B (2026/01/30), 8 omnilingual-asr CTC and LLM models (2025/11/13), Qwen3-Omni-30B-A3B (2025/09/30), Voxtral Small/Mini (2025/09/22), cohere-transcribe-03-2026 (2026/04/16), Gemma-4 E2B/E4B (2026/04/21), and NVIDIA Parakeet-CTC-XXL-1.1B (2025/01/11). — [Leaderboard GitHub](https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard)
- Original paper (Wang et al., Interspeech 2025): the top model was NVIDIA Conformer-CTC-large-Arabic + 4-gram LM at 25.71 average WER and 10.02 average CER, followed by Whisper large and SeamlessM4T. All models are best on MSA and decline sharply on Egyptian and Khaliji. Whisper-large-v3 on SADA scored 51.45 (clean), 58.47 (noisy), and 58.71 (music). — [arXiv 2412.13788](https://arxiv.org/abs/2412.13788v1); [Interspeech 2025](https://www.isca-archive.org/interspeech_2025/wang25_interspeech.html)
- Leaderboard snapshot of 2026-06-23, WER (Avg / SADA / CV / MASC-clean / MASC-noisy / MGB-2 / Casablanca). Source is a vendor blog. — [HF blog, Cohere Transcribe Arabic](https://huggingface.co/blog/CohereLabs/cohere-transcribe-arabic-07-2026-release); [Cohere blog](https://cohere.com/blog/transcribe-arabic)
  - Cohere Transcribe Arabic: 25.87 / 37.47 / 5.82 / 15.54 / 27.07 / 15.54 (possibly a typo duplicating MASC-clean) / 49.71
  - OmniASR-LLM-7B (Meta Omnilingual): 28.32 / 41.61 / 9.75 / 19.69 / 29.29 / 14.13 / 56.46
  - Cohere Transcribe (multilingual): 30.67 / 60.11 / 8.17 / 8.66 / 19.01 / 25.33 / 62.71
  - Whisper-large-v3: 36.86 / 55.96 / 17.83 / 24.66 / 34.63 / 16.26 / 71.81
- Cohere's internal per-dialect test (not public) gives WER for Cohere-Arabic / OmniASR-LLM-7B / Whisper-v3. Code-switched AR-EN: 27.84 / 35.37 / 36.90. Gulf: 24.36 / 36.45 / 46.14. Najdi: 26.18 / 47.74 / 44.97. Hijazi: 16.24 / 16.47 / 30.60. Egyptian: 19.16 / 21.38 / 28.25. Levantine: 39.78 / 74.50 / 58.64. North African: 36.94 / 47.17 / 71.54. — [HF blog](https://huggingface.co/blog/CohereLabs/cohere-transcribe-arabic-07-2026-release)
- Cohere Transcribe Arabic's training recipe: Arabic + Arabic-accented English + L1 English, resampled across dialect groups so none dominates, real bilingual AR-EN code-switched speech, varied devices and noise, plus **TTS-generated samples for underrepresented conditions**. It requires a language tag and works best when the tag is the matrix language. — [HF blog](https://huggingface.co/blog/CohereLabs/cohere-transcribe-arabic-07-2026-release)
- Casablanca (UBC-NLP, Talafha et al. 2024): 48h of fully human-transcribed speech from 717 TV episodes covering 8 dialects, with code-switching labels. License is CC-BY-NC-ND-4.0. — [arXiv 2410.04527](https://arxiv.org/abs/2410.04527v1); [HF dataset](https://huggingface.co/datasets/UBC-NLP/Casablanca/blob/main/README.md)
- NADI 2025 (ASR on Casablanca): Munsit ranked first. It used weakly supervised pretraining on **15,000h** of weakly labeled MSA + dialect speech, then continual supervised fine-tuning on filtered weak data + a small gold set. — [arXiv 2508.08912](https://arxiv.org/abs/2508.08912)
- NADI 2026 (21 teams, 48 submissions) covered ASR, spoken DID, TTS, SLT, and SLU. Out-of-domain generalization was the main bottleneck. — [arXiv 2609.27086](https://arxiv.org/abs/2609.27086)
  - Subtask 1.1, low-bandwidth Casablanca (audio 4kHz→16kHz): Whisper baseline 81.37 WER / 45.89 CER. Winner **Wifaq 43.24 / 19.75**, using 4× w2v-BERT 2.0 CTC + 5-gram KenLM + ROVER. 2nd Resonate 50.29, from full adaptation of one Arabic encoder-decoder. 3rd Thakaa 50.42, a dialect-aware ROVER of 8 systems. KAND CA 51.11 used fine-tuned Whisper + an mT5 multidialect **spelling-correction** post-editor. — [arXiv 2609.27086 HTML](https://arxiv.org/html/2609.27086v1)
  - Subtask 1.2, mixed-dialect (6 countries, 15+ subdialects, no labels): baseline 57.91 / 23.66. Salesteq 44.22 / 18.05. **Namaa 44.65 / 17.16 with zero-shot Cohere Arabic model** (no fine-tuning). Itgan 46.71 used LoRA-Whisper + ROVER. KAND CA 48.07 used full fine-tuning + extra data + replay. — [same](https://arxiv.org/html/2609.27086v1)
  - Subtask 1.3, code-switched Tunisian (FR/EN, 1h50m, talks and podcasts): the top three were within 0.12 WER of each other, and all used model combination. Winner Abjad AI used **synthetic speech augmentation** + weight averaging or ROVER. Hamza normalization changed top-5 WER by only about 0.30. Exact winner WER is in Table D.1, which I did not read. — [same](https://arxiv.org/html/2609.27086v1)
- Almieyar (2026): 17 dialects in 6 families, 13.7h of newly recorded, culturally grounded speech, 12 systems evaluated zero-shot. Overall WER: GPT-4o-transcribe 35.0, Voxtral-Mini-4B 41.1, Fanar-STT-LF 45.9, Whisper-large-v3 49.5. No model is uniformly best across dialect groups. — [arXiv 2609.35564](https://arxiv.org/abs/2609.35564)
- Qwen3-ASR officially covers Arabic among 52 languages and dialects. Community fine-tunes on dialect sets (not comparable across pages): **zero-shot Qwen3-ASR-0.6B scored WER 0.756 / CER 0.454 on a 932-clip dialect test set, and a fine-tune reached 0.676**. On the same page, Whisper-large-v3-turbo scored 0.344 and cohere-transcribe-arabic 0.357. A 1.7B dialect fine-tune reached 0.478 WER on about 2,600 clips vs Whisper-large-v3 0.320. Maghrebi was hardest for every model. — [oddadmix 0.6B](https://friendli.ai/models/oddadmix/qwen3-asr-0.6b-arabic-dialectal); [oddadmix 1.7B v2](https://friendli.ai/models/oddadmix/qwen3-asr-1.7b-arabic-dialectal-v2)
- Self-reported "Lemura" Qwen3-ASR-1.7B Arabic fine-tune: 22.6 average WER over six benchmarks, MGB-2 12.9, Casablanca-Moroccan 43.2. Unverified. — [Lemura model page](https://friendli.ai/models/lemuralabs/lemura-arabic-asr-qwen3)
- Fanar 2.0 (QCRI) Aura speech family adds a long-form ASR model for hours-long audio. The abstract gives no WER numbers. — [arXiv 2603.16397](https://arxiv.org/abs/2603.16397)

### Inferences
- Out-of-the-box Qwen3-ASR appears notably weaker than Whisper-large-v3 and Arabic-specialized models on dialectal Arabic. In community tests it is about 2× Whisper's WER at 0.6B. A specialized model (Cohere Transcribe Arabic, OmniASR-LLM-7B) or a fine-tune is the bigger lever than harness tricks for the NYUAIRLAB-127 baseline.
- Winning shared-task recipes are almost always **ensembles (ROVER / weight averaging) + n-gram LM + Arabic-specialized backbone**, not LLM correction.

### Gaps
- Official leaderboard WER for Qwen3-ASR-0.6B/1.7B, VibeVoice, Voxtral, Omnilingual CTC variants, MMS, SeamlessM4T-v2, and ArTST. These are in the HF Space, which I did not fetch.
- FLEURS-Arabic, MGB-3/5, and QASR SOTA numbers for 2025–26 were not found in this pass.
- Commercial (ElevenLabs Scribe, Gemini, Azure) Arabic dialect numbers on public sets, except GPT-4o-transcribe on Almieyar.

## 2. Arabic-specific evaluation issues

### Takeaway
Scores are only comparable with an explicit normalizer. Typical steps: strip diacritics and tatweel, unify alef/hamza variants, alif-maqsura→ya, Eastern→Western digits, and NFKC. Dialect spelling variation and script choice for English words make WER overstate errors. In one code-switching study WER inflated gaps by about 3×. Report CER alongside WER, and consider spelling-variant-aware or semantic metrics.

### Cited Findings
- Cohere's Arabic normalizer (adapted from Whisper's): removes diacritics, collapses hamza and alef variants, folds alif maqsura and Persian/Urdu letters, converts Eastern Arabic numerals to Western, removes tatweel, normalizes punctuation, applies Unicode NFKC. — [HF blog](https://huggingface.co/blog/CohereLabs/cohere-transcribe-arabic-07-2026-release)
- NADI 2026 Tunisian code-switching: hamza normalization lowered the top five teams' WER by about 0.30 abs without changing the ranking. — [arXiv 2609.27086](https://arxiv.org/html/2609.27086v1)
- Commercial CS benchmark (Egyptian-EN, Saudi-EN, Persian, German; 5 providers; 300 samples per pair): WER exaggerates quality gaps by about 3× because it penalizes semantically correct transliteration choices. BERTScore and WER give the same ranking for Arabic pairs (Kendall τ = 1.0). ElevenLabs Scribe v2 had 13.2 WER and 0.936 BERTScore overall. — [arXiv 2605.19069](https://arxiv.org/abs/2605.19069)
- PolyWER (Findings EMNLP 2024): a code-switching metric that accepts transliterations and translations. It added these annotations to an Arabic-English CS dataset. — [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.356.pdf)
- WERd (ASRU 2017): matches spelling variants mined from social text. It was judged more adequate than WER for Egyptian dialect ASR, because dialects have no regulated orthography. — [arXiv 1709.07484](https://arxiv.org/pdf/1709.07484)
- Multi-reference and segmentation-based WER variants for dialects. — [Ali et al., Interspeech 2019](https://www.isca-archive.org/interspeech_2019/ali19_interspeech.html)
- AraDiaWER (2023): adds LM-embedding-based explanation to WER for Egyptian. — [ACL Anthology 2023.fieldmatters-1.8](https://aclanthology.org/2023.fieldmatters-1.8)
- Almieyar recommends joint WER + CER. wav2vec2 models showed large WER-CER gaps (partially correct words). — [arXiv 2609.35564](https://arxiv.org/abs/2609.35564)
- Reference-side standardization of transcripts reduced WER by about 40% in the ArVoice dataset work. This was seen in a search snippet, not verified in the paper. — [arXiv 2505.20506](https://arxiv.org/pdf/2505.20506)
- Cohere human evaluators checked whether English terms were written in Latin script, which shows that the script convention for code-switched words is a scoring decision. — [HF blog](https://huggingface.co/blog/CohereLabs/cohere-transcribe-arabic-07-2026-release)

### Inferences
- For NYUAIRLAB-127: fix one normalizer (Cohere/Whisper-style plus ta-marbuta/ha folding as an optional variant), always report WER + CER, and for code-switched segments add a transliteration-tolerant score (PolyWER-style) or a semantic score. Otherwise "gains" from LLM post-editing may just be orthographic re-spelling.

### Gaps
- No 2025 paper found that quantifies the WER inflation from dialect spelling variants alone on modern models. Ta-marbuta normalization effect sizes were not found.

## 3. LLM generative error correction (GER) for Arabic

### Takeaway
Evidence for LLM post-correction on Arabic is thin and mostly weak-venue. No peer-reviewed study was found using Jais, ALLaM, Fanar, or AceGPT for ASR GER. The stronger documented harness gains come from **context-aware decoding / proxy-guided n-best selection**: 22% relative on MSA, but only about 9% on dialects. Small seq2seq spelling correctors (mT5) also show up in shared tasks. Risk: LLMs push dialect toward MSA.

### Cited Findings
- Zero-shot context-aware decoding (Talafha, Abu Alhassan, Abdul-Mageed; arXiv 2511.18774, ACL Findings 2026) needs no parameter updates. For Whisper it prompts the decoder with first-pass hypotheses and prefixes retrieved speech-text exemplars, plus speaker-matched synthetic exemplars for informal or multi-speaker speech. For CTC it uses **proxy-guided n-best selection**, choosing the n-best hypothesis closest to external proxy hypotheses. Average relative WER reduction over 10 conditions: **MSA 22.29%, accented MSA 20.54%, dialectal 9.15%**. CTC proxy selection gave 15.6% relative on MSA. — [arXiv 2511.18774](https://arxiv.org/abs/2511.18774v2); [ACL Findings 2026](https://aclanthology.org/2026.findings-acl.1296.pdf)
- Whisper-Large + GPT post-processing across Arabic dialects beat other systems, with the largest gains on dialects. This is a regional journal, and the numbers were not extracted. — [IU Journal of Applied Sciences](https://jesc.iu.edu.sa/Main/Article/289)
- NADI 2026: KAND CA used a fine-tuned mT5 multidialect spelling corrector after Whisper and placed 4th on 1.1 (51.11 WER). Top systems relied on ROVER and ensembles instead. — [arXiv 2609.27086](https://arxiv.org/html/2609.27086v1)
- NADI 2025: MarsadLab's zero-shot cascade of Whisper-large-v3 + MARBERT ranked 7th with 104.9 WER (>100, which suggests a harmful cascade or insertion blow-up). This is snippet-level and was not verified. — [ACL Anthology 2025.arabicnlp-sharedtasks.103](https://aclanthology.org/2025.arabicnlp-sharedtasks.103.pdf)
- General finding: post-hoc LLM correction needs confidence-based filtering to avoid damaging correct transcripts (not Arabic-specific). — [arXiv 2407.21414](https://arxiv.org/html/2407.21414v1)
- ArzEn-LLM (2024) used Llama and Gemma with Whisper for Egyptian-English code-switched ASR→MT. It reported 56% (EN) and 9.3% (AR) translation improvements, and is not a GER study. — [arXiv 2406.18120](https://arxiv.org/abs/2406.18120v2)

### Inferences
- Expect harness-level LLM correction to give single-digit relative WER gains on dialects, with a real risk of MSA-ization (the LLM "fixes" dialect morphology into MSA, which counts as substitutions against a dialectal reference). Mitigations: constrain to n-best or lattice selection, use dialect-tagged prompts, gate by confidence, and evaluate with CER.
- Retrieval of in-domain speech-text exemplars is a cheap test-time method directly applicable to Qwen3-ASR's context or prompt input.

### Gaps
- No found results for Jais, ALLaM, Fanar, AceGPT, or Qwen LLMs as Arabic ASR correctors, and no quantified MSA-normalization-risk study.

## 4. Dialect identification + routing, dialect prompting, contextual biasing

### Takeaway
Out-of-domain dialect ID is still weak (best 56% accuracy at NADI 2026), so hard routing to dialect-specific models is risky. Pooled multi-dialect models match dialect-specific ones, and dialect-aware ensembles help more than routing.

### Cited Findings
- NADI 2026 SDID: out-of-domain data from 6 countries and 18 city dialects. The ECAPA-TDNN baseline scored 81.19% in-domain and **28.70% out-of-domain**. Best accuracy was Thakaa at 56.15% (ensemble weighted to Ara-BEST-RQ, 30% text-branch weight, i.e. multimodal ASR-text + audio). Several teams fine-tuned Cohere's Arabic ASR model for DID. — [arXiv 2609.27086](https://arxiv.org/html/2609.27086v1)
- Thakaa's dialect-aware ROVER of 8 systems was 3rd on low-bandwidth ASR (50.42). — [same](https://arxiv.org/html/2609.27086v1)
- Whisper multi-dialect fine-tuning (Interspeech 2025): dialect-pooled models perform comparably to dialect-specific ones, and small amounts of MSA fine-tuning data substantially help smaller models. — [arXiv 2506.02627](https://arxiv.org/abs/2506.02627v1)
- Talafha et al.: large drops on unseen dialects. — [arXiv 2411.05872](https://arxiv.org/pdf/2411.05872v3)
- Cohere Transcribe Arabic uses a language tag (ar/en). Setting the matrix language correctly matters for code-switching. — [HF blog](https://huggingface.co/blog/CohereLabs/cohere-transcribe-arabic-07-2026-release)
- CV-18 NER (2026): Common Voice Arabic augmented with named-entity annotation, relevant for NE-aware evaluation. Seen only via search. — [arXiv 2604.02209](https://arxiv.org/pdf/2604.02209)

### Inferences
- Prefer soft dialect conditioning (prompt or tag with an ensemble fallback) over hard routing. Error in dialect ID (about 44% OOD) would propagate.

### Gaps
- No Arabic-specific contextual-biasing (hotword or NE list) paper with numbers was found in this pass.

## 5. Self-training, pseudo-labeling, synthetic TTS data, weak supervision

### Takeaway
Weak supervision at scale (15k hours) won NADI 2025. Confidence-filtered pseudo-labeling gives solid gains for low-resource dialects: Sudanese went from 64.1 to 57.1 WER, about 11% relative. TTS-only data helps little or hurts, and works only mixed with real data. Synthetic speech did contribute to winning code-switching systems.

### Cited Findings
- Sudanese (arXiv 2601.06802), Whisper-Medium. Zero-shot 84.3 (eval) / 96.8 (holdout). Full fine-tuning on MSA+Sudanese (7.25h): 64.1 / 57.5. Teacher on 3.93h clean data: 62.5 / 54.0. **Student + pseudo-labels from 17,973 unlabeled clips at confidence ≥0.7 (23.76h): 57.1** / 54.1. Confidence ≥0.9: 57.2 / 52.1. **TTS augmentation (Klaam) + gold: 99.5 / 57.0. TTS only: 71.6 / 79.9.** Best holdout came from gold + pseudo(0.7) + TTS: 57.9 / **51.6**. The paper has internal inconsistencies between the abstract and the tables. — [arXiv 2601.06802](https://arxiv.org/html/2601.06802v1)
- Munsit (NADI 2025, ranked 1st): 15,000h weakly labeled MSA + DA pretraining, then filtered weak + gold fine-tuning. — [arXiv 2508.08912](https://arxiv.org/abs/2508.08912)
- NADI 2025 participant: dialect-specific adaptation of Whisper reduced WER from 93.89 to 39.78 (snippet). — [ACL Anthology 2025.arabicnlp-sharedtasks.103](https://aclanthology.org/2025.arabicnlp-sharedtasks.103.pdf)
- NADI 2026 Tunisian CS winner (Abjad AI) used synthetic speech augmentation + model averaging or ROVER. — [arXiv 2609.27086](https://arxiv.org/html/2609.27086v1)
- Cohere Transcribe Arabic used TTS samples for underrepresented conditions and dialect-balanced resampling. — [HF blog](https://huggingface.co/blog/CohereLabs/cohere-transcribe-arabic-07-2026-release)
- MGB-2: 1,200h of lightly supervised Al Jazeera broadcast (63% conversation, 19% interview, 18% report). QASR has more than 2,000h. — [Casablanca paper context](https://arxiv.org/html/2410.04527v1); [Leaderboard paper](https://arxiv.org/html/2412.13788v1)

### Inferences
- For an RSI loop on Qwen3-ASR: (1) pseudo-label in-domain unlabeled meeting audio with the strongest available teacher (an ensemble or Cohere-Arabic / OmniASR) and confidence-filter it. (2) Use TTS only to add domain terms or code-switched words mixed with real speech, never alone. (3) Include ROVER of diverse teachers to raise label quality. Expect roughly 10% relative on matched dialects.

### Gaps
- No RL (e.g., GRPO/WER-reward) results for Arabic ASR found.

## 6. Arabic conversational, meeting, and lecture datasets

### Takeaway
True Arabic meeting and lecture corpora are scarce and small. ZAEBUC-Spoken (about 12h Zoom, MSA/Gulf/Egyptian + English) and ESCWA (2.8h UN meetings) are the closest. Most large data is broadcast (MGB-2, QASR) or TV (Casablanca).

### Cited Findings
- ZAEBUC-Spoken: about 12h of multi-speaker Zoom meetings, MSA + Gulf + Egyptian, about 30% dialectal, with Arabic-English code-switching. — [arXiv 2403.18182](https://arxiv.org/html/2403.18182v1)
- ESCWA: 2.8h of UN meetings with dialect/EN/FR code-switching. — [arXiv 2403.18182 related-work](https://arxiv.org/html/2403.18182v1)
- Casablanca: 48h, 8 dialects, TV, CS labels, NC-ND license. — [arXiv 2410.04527](https://arxiv.org/abs/2410.04527v1)
- SADA (Saudi TV, 10 dialects) and MASC are in leaderboard test sets. — [Leaderboard GitHub](https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard)
- BULBUL (2026): new dialectal ASR dataset with conversational speech, collected Jan 2025–Apr 2026. Size was not confirmed. — [arXiv 2608.21950](https://arxiv.org/pdf/2608.21950)
- Almieyar: 13.7h, 17 dialects, elicited descriptions (eval only). — [arXiv 2609.35564](https://arxiv.org/abs/2609.35564)
- Tunisian code-switched talks and podcasts set (NADI 2026 1.3, about 1h50m), and the commercial CS benchmark (Egyptian-EN, Saudi-EN, public on HF). — [arXiv 2609.27086](https://arxiv.org/html/2609.27086v1); [arXiv 2605.19069](https://arxiv.org/abs/2605.19069)
- ArzEn: about 12h of informal Egyptian-English code-switched speech. — [Survey arXiv 2510.07037](https://arxiv.org/pdf/2510.07037)

### Inferences
- For meeting and lecture adaptation, the realistic path is in-house unlabeled audio + pseudo-labeling, using ZAEBUC-Spoken/ESCWA/ArzEn as small eval or dev sets.

### Gaps
- No Arabic lecture corpus with domain vocabulary (e.g., university lectures) was found. Forum and panel datasets were not found.
