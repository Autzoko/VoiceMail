# Harness-level (inference-time scaffolding) methods for ASR accuracy: meetings, panels, lectures (2023 – Nov 2026)

Scope note: the notes cover the components wrapped around a fixed ASR model: LLM correction, contextual biasing, ensembling and agents, long-form handling, and automatic optimization. Every number below was checked against the abstract or full-text page cited, unless it is marked "[prior knowledge, not re-verified this session]". Many 2026 items are arXiv preprints. Where I saw only the abstract, I say so.

## Q1. LLM generative error correction (GER): how large are the gains, and when does correction hurt?

### Takeaway
GER over N-best lists gives a few to about 35% relative WER reduction on benchmark data (HyPoradise/GenSEC). The largest gains come from fine-tuned correctors, or from prompts optimized for one domain. Zero-shot LLM correction of already-good transcripts often over-corrects and can double WER. Gating edits by ASR confidence, or restricting them to entity spans with multi-hypothesis evidence, is the main mitigation. Direct evidence on real far-field meeting audio is thin.

### Cited Findings
- **GenSEC challenge (SLT 2024)** had three post-ASR tasks: transcription correction (N-best to ground truth, from Whisper beam-search hypotheses), speaker tagging and emotion recognition. On HyPoradise, the Task-1 baselines were: fine-tuned LLaMA-13B correction WER about 8.62/8.63, LLaMA2-7B 8.71/8.33, T5-750M 9.21/9.05. The two columns are probably with and without normalization; I could not confirm this. — [Yang et al., arXiv 2409.09785](https://arxiv.org/html/2409.09785v3); [IEEE version](https://ieeexplore.ieee.org/document/10832176/)
- FlanEC was a GenSEC Task-1 entry that fine-tuned Flan-T5 encoder-decoders to map N-best lists to one sentence. I did not retrieve its final numbers. — [arXiv 2501.12979](https://arxiv.org/html/2501.12979)
- **Robust GER / RobustHP (ICLR 2024):** a language-space noise embedding extracted from N-best diversity conditions a fine-tuned LLM. It reports up to 53.9% GER WER improvement on noisy test sets. The sources are CHiME-4, VB-DEMAND, LS-FreeSound and NOIZEUS, i.e. noise-corrupted data, not real meetings. — [Hu et al., arXiv 2401.10446](https://arxiv.org/abs/2401.10446)
- **Denoising GER (Sep 2025)** is a multimodal (audio-aware) extension. Its motivation is that a text-only LLM "may misinterpret acoustic information as language noise". It reports significant gains on synthesized noisy speech and generalization to unseen noise and to clean speech. I did not see its CHiME-4 tables. — [Liu et al., arXiv 2509.04392](https://arxiv.org/html/2509.04392)
- **Over-correction evidence (disordered speech, Sep 2025):** Parakeet-TDT 110M plus a LoRA-tuned LLaMA-3.1-8B corrector. Naive correction raised TORGO WER from 10.83% to 20.01%, and SAP-unshared from 9.94 to 10.56. Putting per-word Tsallis-entropy confidence in brackets in the prompt ("focus on low-confidence words") gave 10.58 on TORGO and 9.48 on SAP-unshared. Harmful edits on high-confidence utterances were 1.1–5.0%. Helpful edits were concentrated in low-confidence utterances (63.8% vs 17.5% on SAP-shared). In-domain (SAP-shared) the naive corrector went 15.64 to 4.69, so gains are large in-domain and negative out-of-domain. — [Hernandez et al., arXiv 2509.25048](https://arxiv.org/html/2509.25048)
- **Specialized vs. general LLM correction:** LLMs hallucinate on already-correct ASR output. A specialized correction model (ECLM) built to leave correct text unchanged lowers average WER from 10.9% to 9.5% across conversational and noisy data. — [arXiv 2405.15216](https://arxiv.org/pdf/2405.15216)
- **Confidence-filtered prompting of LLMs on Whisper output (2024):** only lower-confidence transcripts go to the LLM. LLMs help most for weaker acoustic models. — [arXiv 2407.21414](https://arxiv.org/pdf/2407.21414)
- **Zero-shot vs. fine-tuned agentic correction (dysarthric, Jan 2026):** a Judge–Editor LLM agent over top-k hypotheses keeps reliable spans and rewrites uncertain ones. Zero-shot Judge–Editor "yields little improvement"; fine-tuning boosts it greatly. — [arXiv 2601.21347](https://arxiv.org/html/2601.21347v1)
- **Child speech (2025):** GenSEC correctors trained on adult speech generalize poorly. After fine-tuning they give 13.3% relative WER reduction but struggle with insertions and disfluencies. — [CHSER, arXiv 2505.18463](https://arxiv.org/html/2505.18463)
- **Multi-system LLM fusion for long-form talks (KIT IWSLT 2025):** an LLM merged multi-ASR hypotheses across chunks. It did **not** beat the individual systems on ASR. — [arXiv 2505.13036](https://arxiv.org/pdf/2505.13036)
- HyPoradise (NeurIPS 2023) introduced the N-best to transcription benchmark (about 334k pairs) and showed fine-tuned LLMs can sometimes recover tokens absent from the N-best list. [prior knowledge, not re-verified this session] — [Chen et al., arXiv 2309.15701](https://arxiv.org/abs/2309.15701)

### Inferences
- For far-field meetings, a text-only GER is risky. N-best lists from a strong model (Whisper/Qwen3-ASR) are often low-diversity, and the LLM's "fluency prior" rewrites disfluent but correct conversational speech. Use confidence or entity gating and an audio-aware verifier.
- Gains reported on HyPoradise/CHiME-4 are mostly simulated noise or read speech. Expect smaller and less reliable gains on AMI/NOTSOFAR-style real overlap.
- Composition rule from the evidence: GER should run **after** biasing and ensembling, edit only low-confidence or entity spans, and use a guardrail (edit-distance cap, phonetic-similarity check).

### Gaps
- I found no 2025–26 paper reporting GER on AMI/NOTSOFAR/LibriCSS real far-field meeting transcripts with in- vs. out-of-domain comparison.
- I did not retrieve the official GenSEC final leaderboard or the winners' margins over baseline.
- Denoising GER's per-dataset numbers were not retrieved.
- I have no measured latency or cost per hour of audio for GER from any paper in this set.

## Q2. Contextual biasing, hotwords and domain vocabularies (prompt biasing, TCPGen-style, retrieval, slides/agenda/names)

### Takeaway
Biasing is the highest-yield harness component for domain terms. Classic decoder-side methods (CTC word spotting, GPU phrase-boosting tree) and Whisper-based biasing cut biased-word WER by up to about 88% relative while leaving other words "largely unaffected". Prompting speech LLMs with term lists works on read speech but degrades on spontaneous speech, with many distractors, and is sensitive to word order. Large lists therefore need a retrieval or filter stage. Slides and lecture notes as context give about 30–35% relative gains on terms in lecture settings.

### Cited Findings
- **Head-to-head comparison (Aug 2026):** two Whisper-based context-biasing methods vs. three speech LLMs (including Qwen3-ASR-1.7B) prompted with context words, on read and non-read speech. Biasing methods cut biased WER by up to 88% relative with unbiased words "largely unaffected". Speech LLMs did well on read speech but generalized less well to non-read speech. They were sensitive to the number of distractors (up to 250 added) and to the **order of words in the prompt**. — [Huber & Waibel, arXiv 2608.05759](https://arxiv.org/abs/2608.05759)
- **Qwen3-ASR (Jan 2026):** the 1.7B and 0.6B models cover 52 languages. SFT includes **context-biasing data**, which is the source of the "context" field behaviour. The 0.6B model reaches TTFT about 92 ms and 2000 s of audio per 1 s at concurrency 128. Qwen3-ForcedAligner-0.6B is an NAR LLM-based timestamp predictor for 11 languages. The abstract gives no hotword benchmark numbers. — [Shi et al., arXiv 2601.21337](https://arxiv.org/abs/2601.21337); Alibaba Model Studio guidance on improving accuracy with context: [help.aliyun.com](https://help.aliyun.com/en/model-studio/improve-asr-accuracy)
- **NVIDIA NeMo, TurboBias / GPU-PB (Aug 2025)** on CSTalks:
  - Greedy CTC: no boosting WER 13.7%; GPU-PB WER 11.9% (F 64.8, 1991 RTFx); CTC-WS WER 10.9% (F 79.8, about 906 RTFx).
  - RNN-T: GPU-PB 10.7% (F 70.4, 1753 RTFx) vs. CTC-WS 10.1% (F 80.0, 639 RTFx).
  - CTC-WS needs the full file (no streaming) and runs its word spotter on the CPU.
  - GPU-PB supports CTC, RNN-T/TDT and AED (Canary), greedy and beam.

  — [arXiv 2508.07014](https://arxiv.org/html/2508.07014v2); [NeMo word-boosting docs](https://docs.nvidia.com/nemo/speech/latest/asr/asr_customization/word_boosting.html)
- **BR-ASR (Interspeech 2025):** a bias-retrieval front-end for speech LLMs that scales to 200k-entry lists. It reports 93% retrieval accuracy and prunes 99.99% of irrelevant entries. On LibriSpeech test-other with 2,000 bias words, B-WER is 7.1%, about 45% relative better than prior work. — [arXiv 2505.19179](https://arxiv.org/pdf/2505.19179)
- **Prompt biasing in speech LLMs (Interspeech 2024):** about 10% relative B-WER reduction, but it **degrades as list size grows due to hallucination**. A biasing-fusion network follow-up reports 35% relative B-WER reduction on LibriSpeech. — [Gong et al., ISCA](https://www.isca-archive.org/interspeech_2024/gong24b_interspeech.html); [summary](https://lacuna.tiptreesystems.com/work/contextual-biasing-speech-recognition-in-speech-enhanced-large-language-model/wrk_616e30aa82e7a9cf6a804cbd650c6e28)
- **Two-stage rare-word filtering for multi-talker LLM-ASR:** 7.9% WER on LibriMix and 32.9% on AMI SDM at a biasing-list size of 1,000. — [arXiv 2506.12059](https://arxiv.org/html/2506.12059v1)
- **CMT-LLM (Interspeech 2025):** builds AMI biasing lists by OCR of meeting slides. It improves recognition of rare personal names from slides in multi-talker LLM-ASR. Exact numbers were not extracted because the PDF was unreadable to my fetch. — [He et al., ISCA](https://www.isca-archive.org/interspeech_2025/he25_interspeech.pdf)
- **Hotword retrieval + GRPO RL for LLM-ASR (Dec 2025):** retrieved hotwords are injected as structured prompts, trained with distractors and a reward of bias-word match plus (1 − WER). It reports substantial keyword error rate reductions while keeping general-benchmark accuracy. — [arXiv 2512.21828](https://arxiv.org/html/2512.21828)
- **Common-word cues + bias-word position prediction (Apr 2026):** B-WER from 5.8% to 4.4% (up to 24.1% relative). — [arXiv 2604.12398](https://arxiv.org/html/2604.12398)
- **Oracle vs. noisy context exposure bias (Mar 2026):** speech LLMs trained with oracle context are hurt by noisy, realistic context lists. — [arXiv 2603.24034](https://arxiv.org/pdf/2603.24034)
- **LLM rescoring with biasing lists (2023):** biasing lists plus few-shot prompts in second-pass LLM rescoring give 17.8% and 9.6% relative WER gains. Dynamic prompting with multi-task training gives 11.3–20.0% on entity-heavy sets and SLUE-VoxPopuli. — [arXiv 2309.00723](https://arxiv.org/html/2309.00723)
- **Lightweight prompt biasing for E2E ASR:** 4.60% WER with an exact list, 4.80% with a large list, and 4.71% after entity filtering, which shows the distractor penalty. — [arXiv 2506.06252](https://www.arxiv.org/pdf/2506.06252)
- **Lectures, slides as context:**
  - "Do Slides Help?" (Oct 2025) on conference talks: adding slide information gives about 34% relative WER reduction overall and about 35% on domain terms vs. baseline. This was a trained model with slide-augmented data, not zero-shot. — [Sinhamahapatra & Niehues, arXiv 2510.13979](https://arxiv.org/abs/2510.13979)
  - The MIT OpenCourseWare NE study (about 45 h, NER-MIT-OpenCourseWare) uses LLM revision with phonetic-semantic context from lecture material and gets up to about 30% relative WER reduction on named entities. — [Trinh, He & Whitehill, arXiv 2506.10779](https://arxiv.org/abs/2506.10779)
- **Industry APIs (inputs and limits):**
  - **AssemblyAI** `keyterms_prompt`: up to 100 terms in streaming; up to 1,000 words or phrases (max 6 words each) pre-recorded. AssemblyAI recommends putting keyterms inside the free-text `prompt` rather than using both fields. — [AssemblyAI docs](https://www.assemblyai.com/docs/streaming/keyterms-prompting); [guide](https://assemblyai.com/docs/guides/combine-keyterms-and-prompt)
  - **Deepgram** Keyterm Prompting is for Nova-3 and Flux; older models use "Keywords". — [Deepgram docs](https://developers.deepgram.com/docs/keyterm)
  - **OpenAI** `prompt` limits (third-party doc): 1,024 chars for gpt-4o-transcribe and gpt-4o-mini-transcribe, 8,192 chars for gpt-transcribe, and a `keywords` field only on gpt-transcribe. I did not verify this against OpenAI's own docs. — [Vapi docs](https://docs.vapi.ai/customization/custom-keywords)
- Whisper prompting (`initial_prompt` / previous-text conditioning) can steer spelling and style but can also trigger hallucination or looping. Prompt-based biasing of Whisper was studied in "Prompting the Hidden Talent of Web-Scale Speech Models". [prior knowledge, not re-verified this session] — [Peng et al., arXiv 2305.11095](https://arxiv.org/abs/2305.11095)
- TCPGen (tree-constrained pointer generator) and "deep biasing" are pre-2023 trained-component baselines. Later work (e.g. BR-ASR, TurboBias) compares against this lineage. [prior knowledge] — [Sun et al., arXiv 2109.00627](https://arxiv.org/abs/2109.00627)

### Inferences
- For meetings and panels, a practical harness has three stages: (1) build a candidate vocabulary from the agenda, participant names, slides OCR and documents; (2) retrieve or filter per segment to keep about 10–100 terms, because distractors hurt speech LLMs; (3) inject as a prompt (Qwen3-ASR context / Whisper prompt), or as decoder boosting for CTC/TDT models. Measure B-WER, U-WER and false alarms separately.
- The false-alarm risk is mostly from oversized lists in prompt-biased speech LLMs, which the Interspeech 2024 work attributes to hallucination. Decoder-side boosting with word-spotter verification (CTC-WS) is the safest on U-WER, but it is offline only.
- Prompt word order sensitivity (Huber & Waibel) means the term ordering is itself a tunable harness parameter.

### Gaps
- I found no published Qwen3-ASR hotword benchmark (B-WER or false-alarm numbers) in the abstract-level material. The full report or GitHub may contain one.
- Few papers report a false-alarm rate on unbiased words as a separate metric. Most report U-WER only.
- There is no quantitative public benchmark of AssemblyAI, Deepgram or OpenAI keyterm prompting on meeting data. Vendor claims were not retrieved.

## Q3. Agentic ASR frameworks, ensembling, confidence and re-transcription

### Takeaway
Explicit "ASR agent" work so far is mainly post-hoc entity correction. The best-reported example is RECOVER (EMNLP 2026 Industry): multi-hypothesis evidence, LLM-Select, guardrails and a critic loop, with up to 49% relative entity-WER reduction while keeping overall WER. Multi-model ROVER ensembles give large WER gains at about 3x cost. LLM-based fusion of multiple ASR outputs is inconsistent. I found no peer-reviewed system in which an LLM agent orchestrates VAD, diarization or separation and re-decodes uncertain segments end to end.

### Cited Findings
- **RECOVER (Mar 2026, v2 Nov 2026; EMNLP 2026 Industry):**
  - Base ASR is Whisper-large-v3; correction is post-ASR and agentic.
  - Five hypothesis strategies are compared: 1-Best, Entity-Aware Select, ROVER, LLM-Select and Adaptive Routing. LLM-Select was best.
  - The pipeline is: retrieve entities, LLM proposes constrained edits, deterministic guardrails filter, LLM critic judges each edit, with an optional targeted feedback round.
  - Results across 5 datasets: up to 49% relative E-WER reduction, up to +14 pp entity recall, overall WER "preserved".
  - The abstract notes that text-only correction cannot recover entities absent from the 1-best, which motivates the multiple hypotheses.

  — [Kumar & Sachdeva, arXiv 2603.16411](https://arxiv.org/abs/2603.16411)
- **Judge–Editor LLM agent (Jan 2026):** keep or rewrite spans across top-k hypotheses. Fine-tuning is needed; zero-shot gives little. — [arXiv 2601.21347](https://arxiv.org/html/2601.21347v1)
- **ROVER over Whisper + Canary + Parakeet (Sommelier, 2026):** WER 6.26% to 3.92% vs. Whisper-large-v3 alone on LibriSpeech/TED-LIUM3, about 3x slower. — [arXiv 2603.25750](https://arxiv.org/pdf/2603.25750)
- **Multi-ASR fusion + SpeechLLM correction (Interspeech 2025)** for pseudo-labels: fuses Icefall, Parakeet and Whisper, then LLM correction. Ensemble WER is better than or comparable to the best single system. — [Prakash et al., ISCA](https://www.isca-archive.org/interspeech_2025/prakash25_interspeech.pdf)
- **ROVER over three hypotheses (2026):** lowers LibriSpeech WER, but gains are smaller on harder dev sets. — [arXiv 2609.23525](https://arxiv.org/pdf/2609.23525)
- **LLM fusion of multiple systems for long-form talks (KIT IWSLT 2025):** no gain over the individual systems. — [arXiv 2505.13036](https://arxiv.org/pdf/2505.13036)
- **Meeting-system fusion (historical):** DOVER-Lap fusion in the M2MeT 2022 challenge gave 2–15% relative DER reduction. — [arXiv 2202.03647](https://arxiv.org/pdf/2202.03647)
- **Confidence as the routing signal:** per-word entropy confidence (Tsallis) used as a prompt annotation or a word-level filter at a 90% threshold beats naive correction out-of-domain (see Q1). — [arXiv 2509.25048](https://arxiv.org/html/2509.25048)
- **Adjacent agent work:**
  - SpeechAnnotator is a multi-agent speech annotation framework. — [arXiv 2609.09947](https://arxiv.org/pdf/2609.09947)
  - A speculative-execution paper starts tool calls on streaming partial ASR with verification fallback. — [arXiv 2610.07641](https://arxiv.org/pdf/2610.07641)
  - Survey of non-intrusive ASR refinement (Aug 2025). — [arXiv 2508.07285](https://arxiv.org/pdf/2508.07285)
- **Practitioner pipelines** (WhisperX large-v3, optional ROVER reconciliation, LLM post-correction for meetings) exist as tool write-ups, not benchmarks. — [steeman.be](https://www.steeman.be/posts/local-whisper-transcription-with-speaker-diarization/)

### Inferences
- The best-supported agentic pattern is selective: (a) produce several hypotheses (beams, other models, biased/unbiased decodes); (b) localize uncertain or entity spans; (c) let the LLM pick or edit only there, with deterministic guardrails and a critic.
- "Re-decode uncertain segment with a different config", e.g. a biased re-run or a separation front-end on overlap regions, is a natural extension with no published measurements that I found.
- ROVER-type voting is cheap and robust when the systems are diverse. LLM fusion without alignment or confidence did not help in the KIT system.

### Gaps
- I found no published "ASR agent" that calls VAD, diarization, separation or re-decoding tools with reported WER/cpWER on AMI, NOTSOFAR or CHiME-8.
- RECOVER's dataset names, the LLMs used and latency were not in the abstract.
- I found no industry blog with measured agentic re-transcription gains.

## Q4. Long-form handling for lectures and forums: segmentation, context carry-over, term consistency

### Takeaway
Segmentation remains a primary failure source for speech-LLM long-form ASR. Fixed 30 s windows were most robust in a 2026 IWSLT system. Carrying previous transcript text into the prompt helps on contiguous lecture clips, but it propagates errors. Session-level terminology harmonization exists so far only as a data-construction step, not as a measured inference component.

### Cited Findings
- **FBK long-form SpeechLLMs (IWSLT 2026):** long-form performance is sensitive to segmentation and hallucination. Fixed 30 s segmentation was the most robust; discourse-aware segmentation is left as future work. — [arXiv 2606.26819](https://arxiv.org/pdf/2606.26819)
- **Generative Fusion Decoding (2024):** feeds transcripts of earlier contiguous lecture clips into both the ASR and LLM prompts, uses the LLM's long context, and beats Whisper's native long-form mode on contiguous short clips. — [arXiv 2405.14259](https://arxiv.org/pdf/2405.14259)
- **ParaASR (Jul 2026):** builds long-form data by having an LLM review a whole session to harmonize recurring terminology and entities across clips. — [arXiv 2607.29279](https://arxiv.org/pdf/2607.29279)
- **Erroneous context:** multimodal conversational context for LLM-ASR studies how a model handles wrong textual history, which is the error-propagation risk of carry-over. — [arXiv 2609.19765](https://arxiv.org/pdf/2609.19765)
- **NIM4-ASR (Apr 2026):** keeps context at phrase level for real-time use; sentence- or dialogue-level context is a latency trade-off. — [arXiv 2604.18105](https://arxiv.org/pdf/2604.18105)
- **Qwen3-ForcedAligner** provides word timestamps for re-segmentation and alignment. — [arXiv 2601.21337](https://arxiv.org/abs/2601.21337)

### Inferences
- A two-pass harness looks like the cheap win for lectures and panels. Pass 1 transcribes with VAD/fixed windows. Then extract recurring term candidates across the session (frequency + LLM), and do pass-2 biased re-decoding or constrained span correction to enforce cross-segment consistency.
- Carry over a term list rather than raw previous text, to limit hallucination and error propagation. This is an inference from the papers above, not a measured result.

### Gaps
- I found no measured cross-segment term-consistency metric (e.g. variance of an entity's spelling within a lecture) in published work.
- There is no direct comparison of VAD- vs. fixed- vs. semantic-segmentation for Qwen3-ASR on meetings.

## Q5. Automatic optimization of ASR harnesses (prompts, biasing lists, thresholds, structure vs. dev WER)

### Takeaway
Evidence is sparse but real. EvoPrompt-style search over GER prompts on CHiME-4 cut WER from 7.49% to about 4.88% for about $2–60 of API cost, but the evolved prompts overfit and badly hurt other domains. RL (GRPO) with WER-based rewards tunes hotword-prompt consumption at the training level. I found no published work that searches the whole pipeline structure (biasing list, thresholds, model routing) against dev WER.

### Cited Findings
- **Evolutionary prompt design for post-ASR correction (2024):**
  - Setup: Claude 3.5 Sonnet, text-only N-best, CHiME-4 subset of HyPoradise (GenSEC Task 1); 8,738 train / 1,320 test.
  - Result: default challenge prompt 7.49% WER vs. best evolved prompt plus one demonstration about 4.88% (about 35% relative).
  - Search: GA-EvoPrompt from 5 seed prompts, 3 iterations, 5 children per iteration via LLM crossover and mutation.
  - Cost: about $2.2 per CHiME-4 run, about $60 for the full study.
  - Overfitting: the top early prompt **hurt** other domains (Common Voice 13.56 vs. 11.06 baseline; WSJ 14.75 vs. 3.67). Another evolved prompt gave slight gains (CV 10.92, WSJ 2.79).

  — [arXiv 2407.16370](https://arxiv.org/html/2407.16370v1)
- **GRPO with reward = bias-word match + (1 − WER), trained with distractors**, for LLM-ASR hotword consumption. This is a training-time "optimize against WER" step. — [arXiv 2512.21828](https://arxiv.org/html/2512.21828)
- **Tunable knobs documented in the literature:**
  - biasing-list size and distractors — [arXiv 2506.06252](https://www.arxiv.org/pdf/2506.06252); [arXiv 2608.05759](https://arxiv.org/abs/2608.05759)
  - prompt word order — [arXiv 2608.05759](https://arxiv.org/abs/2608.05759)
  - confidence threshold (90% word-level filter best on one set) — [arXiv 2509.25048](https://arxiv.org/html/2509.25048)
  - hypothesis-selection strategy (LLM-Select vs. ROVER vs. adaptive routing) — [arXiv 2603.16411](https://arxiv.org/abs/2603.16411)
  - segmentation window — [arXiv 2606.26819](https://arxiv.org/pdf/2606.26819)

### Inferences
- Harness self-optimization for speech is a largely open research slot. The search space is well defined: term-retrieval k, list order, prompt template, confidence gate, edit-distance cap, ensemble members, segment length. The objective must be multi-metric (WER + B-WER + U-WER/false alarms) on a held-out split.
- The EvoPrompt result shows the main danger: overfitting to the dev domain. Cross-domain validation is mandatory.

### Gaps
- I found no paper that uses an LLM or agent optimizer (e.g. DSPy/GEPA/OPRO-style) over a full ASR pipeline against dev WER for meetings or lectures.
- No paper reported the compute cost of the search relative to ASR inference, except the EvoPrompt dollar figures.
