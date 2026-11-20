# Optimizing Multi-Module ("Compound AI") Pipelines End to End: Credit Assignment, Hybrid Search, Routing, Flywheels (2024 – Nov 2026)

Scope note: this file covers the multi-module and pipeline side only. General Meta-Harness, HarnessCompass, GEPA, MIPROv2, TextGrad, ADAS, AFlow and DGM background (single-harness headline numbers, overfitting gates, raw-trace ablation) is already in `../Harness RSI 语音识别准确率提升/harness_rsi_foundations.md` and is not repeated here.

Labels: [V] means verified in this session from the primary page (arXiv abstract or HTML). [R] means recalled background from well-known work, cited but not re-read this session. BG marks pre-2025 background.

Pipeline context: speech → VAD/diarization → LID → router (Qwen3-ASR or Audar-ASR) → post-processing.

---

## Q1. Compound-system optimizers: how each assigns credit to modules, and which support non-LLM knobs

### Takeaway
There are three credit-assignment families.
- **Global-feedback / textual backprop (TextGrad, Trace/OptoPrime, LLM-AutoDiff, ADOPT).** An LLM reads the execution trace and attributes blame to each node.
- **Search over whole-system scores (DSPy MIPROv2, GEPA, MASS, AFlow, ADAS, Meta-Harness).** Credit is implicit: one module is mutated at a time and the end metric is measured.
- **Learned per-module surrogate rewards (Optimas, LLMSelector).** Each module gets a local reward or estimator aligned with the global metric, so it can be tuned cheaply and in isolation.

Only Optimas (prompts, hyperparameters, model selection and weights) and Trace (numeric parameters as trainable nodes) natively cover heterogeneous, non-LLM knobs. LLMSelector covers per-module model choice. For a speech pipeline with thresholds plus model choice plus prompts plus code, Optimas-style local rewards are the closest fit.

### Cited Findings

**Optimas** (Wu, Sarthi, …, Zou, Leskovec; Stanford/Amazon; arXiv 2507.03041; ICLR 2026). [V] — [arXiv abs](https://arxiv.org/abs/2507.03041), [HTML](https://arxiv.org/html/2507.03041), project site optimas.stanford.edu
- **Credit assignment.** Each component k gets a Local Reward Function (LRF) r_k = h_k ∘ φ([x_k, y_k]), with a shared LLM backbone φ and one linear head per component.
  - LRFs are trained with a pairwise log-sigmoid ranking loss.
  - Preference pairs come from running the system up to component k, sampling two outputs for k, and labeling the better one by a Monte-Carlo estimate of the downstream global reward.
  - The required "local–global alignment property": raising the local reward raises global performance.
  - LRFs are trained offline first, then adapted online with small mini-batches plus a replay buffer whenever any component's configuration changes.
- **Per-type optimizers.**
  - Prompts: OPRO-style candidates ranked by mean local reward.
  - LLM weights: PPO with the LRF as reward.
  - Discrete or low-dimensional knobs (model selection, hyperparameters): sampling from a distribution weighted by local reward.
  - This is the only surveyed framework that explicitly mixes all four types.
- **Results (Table 2), Optimas vs. best baseline.** Average relative improvement 11.92%.

  | System (metric) | Optimas | Best baseline |
  |---|---|---|
  | Amazon (Acc) | 24.24 | REINFORCE 21.89 |
  | PubMedQA (Acc) | 69.13 | LLMSelector 67.93 |
  | STaRK-Prime (MRR) | 50.54 | DSPy 41.40 |
  | HotpotQA (F1) | 50.48 | DSPy 44.90 |
  | BigCodeBench (Pass) | 38.92 | TextGrad 35.71 |

- **Alignment quality.** Pairwise ranking accuracy is about 70–73% and is insensitive to backbone size (1B 72.58%, 3B 70.04%, 8B 71.20%).
  - 12.5% of the training data recovers 92.7% of full-data alignment.
  - The LRF beats an LLM judge with 20 in-context examples on pairwise accuracy: 77.96% vs. 49.52%.
- **Cost.** Equivalent full-system runs, averaged over tasks: Optimas 0.71k, TextGrad 0.80k, DSPy 0.79k. LLMSelector needs about 2.8k, and REINFORCE needs more than 3x Optimas's data.

**LLMSelector** (Chen, Davis, Hanin, Bailis, Zaharia, Zou, Stoica; arXiv 2502.14815, Feb 2025). [V] — [arXiv](https://arxiv.org/abs/2502.14815)
- Per-module model allocation. Iteratively pick one module and assign it the model with the best estimated module-wise performance (the estimate comes from an LLM); stop when nothing improves.
- **Key empirical premise:** end-to-end performance is "often monotonic" in each module's quality, holding the other modules fixed. That justifies coordinate-wise credit assignment.
- Accuracy gains of 5–70% over using one LLM for every module (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5; debate and self-refine systems).
- API calls scale linearly with the number of modules.
- The authors flag joint model-selection plus prompt optimization as open.

**Trace / OptoPrime** (Cheng, Nie, Swaminathan; Microsoft; arXiv 2406.16218; NeurIPS 2024) (BG). [V] — [arXiv](https://arxiv.org/html/2406.16218), [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/83ba7056bce2c3c3c27e17397cf3e1f0-Abstract.html), site microsoft.github.io/Trace
- Records the execution graph (a minimal subgraph) of a Python workflow, framed as Optimization with Trace Oracle (OPTO).
- OptoPrime prompts an LLM with the trace plus output feedback, and updates all trainable nodes jointly.
- The same optimizer, unchanged, does first-order numerical optimization, hyperparameter tuning, prompt optimization and robot-controller design, and is "competitive with specialized optimizers."
- Trainable nodes can be code, prompts or numeric parameters.
- The EMNLP 2025 survey notes Trace "updates all LLM prompts at once" from global NL feedback, avoiding TextGrad's per-node linear growth in LLM calls. — [Survey 2506.08234](https://arxiv.org/html/2506.08234v2)

**LLM-AutoDiff / AdalFlow** (Li Yin, Zhangyang Wang; arXiv 2501.16673, Jan 2025). [V, abstract-level] — [arXiv](https://arxiv.org/abs/2501.16673v2)
- Extends textual gradients to multi-component and cyclic graphs. Nodes are LLM components, functional (code) components, or loss components.
- A frozen "backward engine" LLM produces per-node feedback.
- Adds "time-sequential gradients" for nodes that are called repeatedly.
- Reported to beat textual-gradient baselines in both accuracy and training cost on classification, multi-hop RAG QA and agent pipelines. No numbers were retrieved.
- Code nodes pass gradients through but are not themselves optimized.

**ADOPT** (arXiv 2512.24933, Dec 2025). [V] — [HTML](https://arxiv.org/html/2512.24933v2)
- **Credit assignment.**
  - Builds a global textual gradient from final-task errors.
  - Decomposes it into step-level local gradients using an analysis of how each step's output depends on earlier steps ("dependency-guided estimation").
  - Allocates optimization budget across steps by Shapley value.
- **Test accuracy:**

  | Pipeline | MIPRO | GEPA | ADOPT |
  |---|---|---|---|
  | HotPotQA (2-step, loop) | 0.62 | 0.63 | 0.69 |
  | HoVer (5-step) | 0.62 | 0.65 | 0.71 |
  | Emergency (4-step, loop) | 0.78 | 0.83 | 0.89 |

  Amazon, a 3-step pipeline: TextGrad 0.90, Trace 0.92, ADOPT 0.94.
- TextGrad and Trace "don't natively support looping pipelines," so they have no scores on the looped tasks.
- **Ablations:**
  - Removing dependency-guided estimation: 0.69→0.64 on HotPotQA, 0.71→0.67 on HoVer. This is the largest drop.
  - Removing Shapley allocation barely changes accuracy (0.69→0.68 and 0.71→0.71) but costs more tokens: HotPotQA 5.52M→8.56M, HoVer 43.3M→62.3M.
- **Cost on HoVer:** ADOPT 43.3M tokens / 10.9 h, GEPA 31.1M / 25.7 h, MIPRO 75.5M / 15.3 h.

**MASS — Multi-Agent System Search** (Zhou et al., Google; arXiv 2502.02533; ICLR 2026). [V] — [arXiv HTML](https://arxiv.org/html/2502.02533v2)
- **Three stages:** (1) block-level, i.e. local per-agent, prompt optimization; (2) topology search in a pruned space of "influential" topologies; (3) workflow-level, i.e. global, prompt optimization conditioned on the chosen topology.
- **Stage ablation, Gemini 1.5 Pro average:**

  | Configuration | Score |
  |---|---|
  | Base | 63.54 |
  | + single-agent APO | 67.44 |
  | + Stage 1 (block-level) | 74.56 |
  | + Stage 2 (topology) | 77.55 |
  | + Stage 3 (workflow-level) | 78.40 |

- **Main table, 8-task average (Gemini 1.5 Pro):** MASS 78.79, Multi-Agent Debate 70.26, ADAS 69.72, Self-Consistency 68.18, CoT 65.28.
  - AFlow wins on 2WikiMQA, 76.51 vs. 73.34.
  - On Gemini Flash: MASS 74.30 vs. ADAS 64.75.
- Only a small fraction of topologies help. On HotpotQA only debate helped (about +3%); the others did not help or hurt.
- Training cost is about $5 for MASS, ADAS and AFlow alike.
- **Lesson stated by the authors:** "optimize agents locally before scaling their topology"; then tune prompts at the workflow level for interdependence.

**DSPy MIPROv2 / GEPA in multi-module programs.**
- MIPROv2 (2024, BG) uses a Bayesian surrogate (TPE) over per-module instruction × demo choices and credits modules only through the end metric. [R] — [arXiv 2406.11695](https://arxiv.org/abs/2406.11695)
- GEPA (ICLR 2026 Oral) mutates one module per step, with the target module chosen round-robin.
  - It reflects on that module's trace slice plus evaluator feedback.
  - It keeps a per-instance Pareto front rather than a single best candidate.
  - [V, via project page and a secondary blog] — [Berkeley Sky GEPA](https://sky.cs.berkeley.edu/project/gepa/), [arXiv 2507.19457](https://arxiv.org/abs/2507.19457)

**Taxonomy (EMNLP 2025 survey, Lee et al., arXiv 2506.08234).** [V] — [arXiv HTML](https://arxiv.org/html/2506.08234v2), list of papers at github.com/MiuLab/AISysOpt-Survey

| Method | Structure | Learning signal |
|---|---|---|
| TextGrad | fixed | NL feedback |
| Trace | fixed | NL feedback |
| DSPy, MIPRO | fixed | numeric metrics, "rejection-sampling–based" |
| ADAS, AFlow | flexible (Python code) | NL feedback |
| MASS | flexible (graph) | NL feedback |

- The survey separates textual parameters θ_T from numeric θ_N (weights, temperature).
- Named challenges: manual hyperparameter configuration, excessive computation burden (an API vs. GPU trade-off), limited experimental scope ("datasets widely used for single LLMs"), empirical NL feedback, safety, and inconsistent library support.
- The survey does not treat overfitting as a challenge. The earlier Lin et al. survey (arXiv 2410.16392, BG) has an explicit "Credit Assignment with Dynamic Program Analysis" section. — [arXiv 2410.16392](https://arxiv.org/html/2410.16392v1)

**Meta-Harness / HarnessCompass / AFlow / ADAS.** These assign credit implicitly: a coding-agent proposer reads raw traces from all modules and decides what to edit. Background is in the foundations file.
- Relevant here: Meta-Harness's ablation (scores only 34.6 median vs. full traces 50.0) is effectively evidence that per-module trace visibility is the credit signal. — [Meta-Harness](https://arxiv.org/html/2603.28052v1) [V in foundations file]

### Inferences
- Map onto our pipeline:
  - VAD and diarization thresholds (onset/offset, clustering threshold, minimum cluster size, chunking) and LID confidence thresholds are θ_N knobs. Optimas-style "sample from a local-reward-weighted distribution", or plain BO, fits them better than textual gradients.
  - The ASR choice per route and decoding configuration are model-selection or discrete knobs, where LLMSelector-style coordinate ascent is appropriate.
  - Post-processing LLM prompts and normalization code are θ_T or code, where GEPA, Trace, ADOPT or a Meta-Harness proposer fits.
- **Natural local rewards exist for speech modules**, unlike in generic LLM pipelines:
  - DER and JER for diarization;
  - LID accuracy, and LID accuracy weighted by downstream WER impact;
  - per-route WER/CER, and WER delta of post-processing.
  - Optimas's warning still applies: a local metric is useful only if it is aligned with the end metric (e.g., cpWER/tcpWER). That alignment should be measured, for example as rank correlation between the change in local metric and the change in global metric across candidate configurations.
- LLMSelector's monotonicity premise is plausible for ASR swaps but probably breaks for diarization, where under-segmentation vs. over-segmentation trade off differently against ASR context length.

### Gaps
- No numbers were retrieved for LLM-AutoDiff. Optimas's exact code repository URL was not confirmed; only the project site was.
- I found no published compound optimizer applied to a pipeline that contains a non-LLM neural model with continuous thresholds (e.g., a VAD) together with LLM prompts. Optimas's "hyperparameter" components are LLM-side settings.

---

## Q2. Joint vs. module-wise optimization; error propagation in cascades

### Takeaway
Evidence is mixed, and the useful pattern is **local first, then joint**.
- MASS gets most of its gain from per-block optimization (+7.1 points) and only +0.85 from the final joint pass.
- ADOPT shows that dependency-aware joint credit beats dependency-blind per-step updates by 4–5 points.
- In speech, cascades still beat unified models on multilingual speaker-attributed ASR (HINTT 2026: 15.28 vs. 18.50 tcpWER) even though they suffer error propagation.

So the practical answer is a modular cascade, with module-wise tuning against aligned local metrics plus a final end-to-end tuning pass, not replacing the cascade.

### Cited Findings
- **MASS stage ablation:** local block-level optimization 67.44→74.56; topology 77.55; global workflow-level prompts 78.40. Authors: optimize locally before composing. [V] — [MASS](https://arxiv.org/html/2502.02533v2)
- **ADOPT:** removing the cross-step dependency model (i.e., closer to independent step-wise credit) costs −5 points on HotPotQA and −4 on HoVer. "Joint" step optimization beat instruction-only (0.69 vs. 0.67; 0.71 vs. 0.69). [V] — [ADOPT](https://arxiv.org/html/2512.24933v2)
- **Optimas:** jointly aligned local rewards beat end-to-end-only optimizers (DSPy, TextGrad) and per-module RL without alignment (REINFORCE) on 5/5 systems, at equal or lower full-system-run budget. [V] — [Optimas](https://arxiv.org/html/2507.03041)
- **LLMSelector:** coordinate-wise (one-module-at-a-time) model allocation gets 5–70% gains, relying on monotonicity of end performance in module quality. [V] — [LLMSelector](https://arxiv.org/abs/2502.14815)
- **HINTT, 2nd MLC-SLM Challenge (arXiv 2610.08063, Oct 2026).** Directly relevant, because the cascade uses **Qwen3-ASR**. [V] — [HTML](https://arxiv.org/html/2610.08063)
  - Cascade: DiariZen-FT + SimAMResNet100 embedding → fine-tuned Qwen3-ASR → LLM generative error correction (GEC).
  - Dev tcpWER/CER averages:

    | System | tcpWER/CER |
    |---|---|
    | Official baseline | 79.15 |
    | Cascade | 16.00 |
    | + SpecAugment | 15.51 |
    | + GEC | 15.28 |
    | Unified VibeVoice-ASR-FT | 18.50 |
    | Whisper-FT in the cascade | 16.50 |

  - Eval-set progression: 16.70→16.39→16.27. GEC adds only −0.23 on dev and −0.12 on eval.
  - Diarization DER: DiariZen 17.04 → fine-tuned 3.25 → + better embedding 2.93. The unified model's DER is 5.12. Speaker confusion is the dominant residual error (2.27).
  - The unified model wins on es, th and ur but loses badly on en-IN (24.48 vs. 12.85), ja (25.47 vs. 16.03) and ru (21.73 vs. 13.99).
  - Settings: clustering threshold 0.7, minimum cluster size 16, 64 s chunks with a 48 s shift. **No threshold ablation and no oracle-diarization run**, so the propagation cost is not quantified.
- **Joint VAD + target-speaker ASR (Interspeech 2025).** Separate optimization causes "unstable training and error accumulation"; the joint model reaches 6.61% and 14.81% WER on its benchmarks. [V, snippet] — [Maeda et al. IS2025](https://www.isca-archive.org/interspeech_2025/maeda25_interspeech.pdf)
- **SpeakerLM (arXiv 2508.06372)** lists error propagation as the first limitation of cascaded SD+ASR. [V, snippet] — [arXiv](https://arxiv.org/pdf/2508.06372)
- **Joint ASR + role diarization (arXiv 2601.17640).** The joint model avoids boundary errors from external segmentation that "directly disrupt downstream transcription", and also beats an ASR-first baseline on WER. [V, snippet] — [arXiv](https://arxiv.org/abs/2601.17640)
- **LID-cascade fragility.**
  - LID-then-route multi-pass systems "rely on the accuracy of code-switch detection and language identification" and lose context at switch boundaries. [V, snippet; patent] — [USPTO 11417322](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11417322)
  - Older evidence (BG, 2018): adding language tags opened a persistent 2–2.5% absolute WER gap attributable to language-recognition errors (37.8%→40.3%). [V, snippet] — [arXiv 1808.00521](https://arxiv.org/pdf/1808.00521)
  - Code-switch errors concentrate at switch points and are masked in aggregate WER. One model improved overall WER while switch-point accuracy worsened. [vendor blog; treat as indicative] — [Deepgram](https://deepgram.com/learn/what-is-code-switching-asr-guide), [arXiv 2510.18724](https://arxiv.org/html/2510.18724v1)

### Inferences
- For our harness, compute **oracle-substitution deltas** per stage as the cheapest credit-assignment tool for a cascade:
  - global metric with reference segmentation vs. predicted;
  - reference LID vs. predicted;
  - best-of-two ASR (oracle router) vs. the actual router;
  - post-processing on vs. off.
  This decomposes end-to-end cpWER/tcpWER into per-module headroom. HINTT did not do this, which is exactly the gap that leaves propagation unquantified.
- HINTT suggests diarization fine-tuning (DER 17→3) and the ASR choice dominate. GEC is worth only about 0.1–0.2 absolute. Optimization budget should follow the measured headroom, which is also ADOPT's Shapley-allocation idea.
- For Arabic/English code-switching (common in Gulf meetings), a hard LID gate is a structural risk. Candidates for the search space: segment-level LID with a confidence margin; falling back to both ASRs plus selection; and evaluating switch-point error separately from aggregate WER.

### Gaps
- I found no paper that quantifies LID-router error propagation into WER for a modern speech-LLM ASR pair. Also no oracle-vs-predicted diarization ablation for Qwen3-ASR cascades.
- No published joint-vs-isolated tuning experiment on VAD and diarization thresholds against downstream cpWER was found in this session.

---

## Q3. Hybrid search (BO / Optuna for numeric knobs + LLM proposers for code and prompts); learned routers and cascades

### Takeaway
No retrieved method optimizes thresholds, model choice, prompts and code jointly in one loop with published numbers. The practical design is hierarchical:
- BO/TPE (or Optimas-style reward-weighted sampling) for low-dimensional numeric and categorical knobs;
- an LLM or coding-agent proposer for prompts and code;
- alternation between them (block-coordinate), as in MASS or LLMSelector.

LLM-in-the-loop BO (LLAMBO, an Optuna sampler) exists but is best used for warm-starting.

Routers are trained from logged or preference outcomes. RouteLLM uses pairwise preferences, with LLM-judge augmentation cutting strong-model calls from 26% to 14%. Confidence-threshold cascades are the speech-ASR precedent. Production routers are improved by distilling from logged failures (NVInfo: 70B router → fine-tuned 8B at 96% accuracy).

### Cited Findings
- **LLAMBO** (Liu et al., ICLR 2024, BG). An LLM serves as zero-shot warm-starter, surrogate (generative or discriminative, via in-context learning) and candidate sampler. The authors concede the much larger compute footprint and suggest using LLAMBO early in search or only as one component of a classical BO. [V] — [arXiv 2402.03921](https://arxiv.org/pdf/2402.03921)
- **Optuna hub LLAMBO sampler.** Numeric parameters go to LLAMBO and categorical ones to RandomSampler. Default model gpt-4o-mini; about $0.05–0.10 per 30-trial run. [V, snippet] — [Optuna hub](https://hub.optuna.org/samplers/llambo/)
- **Optuna TPE vs. Code Llama** for hyperparameter tuning (arXiv 2504.06006, 2025): a direct comparison exists; findings were not retrieved. — [arXiv](https://arxiv.org/pdf/2504.06006)
- **Prompt-space BO:**
  - HbBoPs (Hyperband + BO for black-box prompt selection, arXiv 2412.07820);
  - BO-LLM (an LLM-powered GP surrogate, with an LLM generating new prompts from top seeds, arXiv 2510.04384).
  - [V, snippet] — [HbBoPs](https://arxiv.org/pdf/2412.07820), [BO-LLM](https://arxiv.org/html/2510.04384v1)
- **HARBOR** (arXiv 2604.20938): cost-aware BO over harness configuration, positioned as complementary to Meta-Harness. Already noted in the foundations file. — [arXiv](https://arxiv.org/pdf/2604.20938)
- **Optimas:** discrete and hyperparameter components are optimized by sampling configurations from a local-reward-weighted distribution, inside the same loop as OPRO prompts and PPO weights. This is the closest published "hybrid." [V] — [Optimas](https://arxiv.org/html/2507.03041)
- **RouteLLM** (Ong et al., LMSYS/Berkeley; arXiv 2406.18665; ICLR 2025). [V] — [arXiv](https://arxiv.org/html/2406.18665v4), [LMSYS blog](https://lmsys.org/blog/2024-07-01-routellm/)
  - Routers (matrix factorization, BERT classifier, causal-LLM classifier, similarity-weighted ranking) are trained on Chatbot Arena pairwise preferences between a strong and a weak model.
  - Against GPT-4-only, cost falls by more than 85% on MT-Bench, 45% on MMLU and 35% on GSM8K, while keeping 95% of GPT-4 quality.
  - Matrix-factorization router on MT-Bench: 26% GPT-4 calls on Arena data alone, falling to 14% with LLM-judge-augmented labels (48%→75% cheaper than a random router).
  - About 1,500 in-domain augmented samples (under 2% of the data) materially improved MMLU routing. Small, targeted, logged in-domain labels are high-leverage.
- **FrugalGPT** (Chen, Zaharia, Zou, 2023, BG): an LLM cascade with a learned scorer deciding whether to accept an answer or escalate; reports up to 98% cost reduction at GPT-4 parity. [R] — [arXiv 2305.05176](https://arxiv.org/abs/2305.05176)
- **Unified routing + cascading** (Dekoninck et al., ICLR 2025, BG-ish): derives the optimal cascade-routing strategy from quality and cost estimators, and shows estimator quality is the bottleneck. [R] — [arXiv 2410.10347](https://arxiv.org/abs/2410.10347)
- **ASR routing precedent: confidence-based on-device → server ASR cascade.** At a 0.85 confidence threshold, overall WER on Maps (9.6%) beats either system alone, with only 53% of utterances escalated. [V, snippet] — [arXiv 2103.06716](https://arxiv.org/pdf/2103.06716)
- **e-WER** (ACL 2018, BG): per-utterance WER prediction from decoder features; estimated 25.3% vs. actual 28.5% WER on 3 h. This is a usable router and quality-estimation signal. [V, snippet] — [ACL P18-2004](https://aclanthology.org/P18-2004.pdf)
- **ROVER-style combination** beat the best single system in 5 of 9 cases (BG). [V, snippet] — [USC TASLP 2014](https://sail.usc.edu/publications/files/diverse_asr_ensemble_taslp2014.pdf)

### Inferences
- **Router for Qwen3-ASR vs. Audar-ASR.** Treat the LID→ASR routing as a RouteLLM-style learned router rather than a hard LID argmax.
  - Inputs: LID posteriors, an acoustic embedding, and the confidence of a cheap first pass.
  - Labels come from logged pairwise outcomes: run both ASRs on a sampled slice of traffic and use reference WER where available, else an LLM or Arabic-native judge or cross-model agreement.
  - The objective is expected WER at a compute budget, with an escalation option (run both, select or ROVER) for low-margin segments.
- **Proposed hierarchy:**
  - (a) Optuna TPE or multi-objective NSGA-II over VAD, diarization and LID thresholds, chunk sizes, and decoding parameters (beam, temperature, context-biasing strength), each trial scored on the held-out cpWER/tcpWER plus latency;
  - (b) a GEPA, Meta-Harness or ADOPT proposer for post-processing prompts, normalization rules and routing code;
  - (c) alternate, with the numeric layer re-run after each accepted code change, because changes shift the optimum. That shift is MASS's observation that later stages are conditioned on earlier ones.
  - An LLM may warm-start the Optuna priors (LLAMBO-style), but it should not replace TPE in the inner loop on cost grounds.

### Gaps
- I found no peer-reviewed head-to-head of an LLM-hybrid BO sampler vs. plain TPE on pipeline knobs. The findings of the Optuna-vs-Code-Llama paper were not retrieved.
- I found no 2025–26 paper on a learned per-utterance router between two modern speech-LLM ASR systems. The closest are confidence cascades and WER estimation.

---

## Q4. Production data flywheels and continual self-improvement loops

### Takeaway
The best-documented case is NVIDIA's NVInfo AI (2025; EACL 2026 industry track).
- A MAPE loop (monitor, analyze, plan, execute) mined 495 negative-feedback samples over 3 months and attributed them to pipeline stages: routing errors 5.25%, query-rephrasal errors 3.2%.
- Fixing the **router** via fine-tuning went from 70B to 8B at 96% accuracy, 10x smaller and 70% lower latency.

The generic recipe is: instrument every stage; classify errors and attribute them to a stage; grow the eval set from mined failures; fine-tune or edit only the responsible module; gate on a regression suite; roll out in stages.

### Cited Findings
- **NVInfo AI flywheel** (arXiv 2510.27051; EACL 2026 industry). [V] — [arXiv](https://arxiv.org/abs/2510.27051), [ACL Anthology](https://aclanthology.org/2026.eacl-industry.33.pdf)
  - A Mixture-of-Experts knowledge assistant serving more than 30,000 employees.
  - MAPE loop; 3 months of post-deployment feedback yielded 495 negative samples.
  - Failure modes: routing errors 5.25%, query-rephrasal errors 3.2%.
  - Router: Llama-3.1-70B replaced by a fine-tuned 8B, at 96% accuracy, 10x size reduction and 70% latency improvement.
  - Rephrasal module: +3.7% accuracy, −40% latency.
  - The abstract lists robustness under sparse feedback, privacy constraints and staged rollouts as key lessons (details not retrieved).
- **NVIDIA Data Flywheel Blueprint.** Curates datasets from production logs, then uses NeMo Customizer for fine-tuning and NeMo Evaluator for evaluation; runs continuously as traffic arrives. [V, snippet] — [build.nvidia.com](https://build.nvidia.com/nvidia/build-an-enterprise-data-flywheel)
- **Arize + NeMo:** online LLM-as-judge evaluators on production traces feed the flywheel. [V, snippet] — [Arize blog](https://arize.com/blog/building-the-data-flywheel-for-smarter-ai-systems-with-arize-ax-and-nvidia-nemo/)
- **Practitioner recipe** (LinkedIn; not authoritative): instrument all stages, use LLM classifiers for error type, and attribute errors to a stage. [V, snippet] — [LinkedIn](https://pk.linkedin.com/in/izazahmad)
- **RouteLLM:** a small in-domain augmentation set (about 1.5k labels) shifted router quality materially. Logged-outcome relabeling is high-leverage for routers. [V] — [LMSYS](https://lmsys.org/blog/2024-07-01-routellm/)
- **Optimas:** online LRF adaptation with a replay buffer of earlier preference data whenever a component changes. This is a mechanism for keeping per-module rewards valid as the pipeline evolves. [V] — [Optimas](https://arxiv.org/html/2507.03041)
- See the foundations file for the regression-gating and held-out-split practices of Meta-Harness and HarnessCompass, and for Voice Memory (NVIDIA, Jul 2026): a validation-gated memory for ASR error correction.

### Inferences
Speech-pipeline flywheel design:
1. Log per segment: VAD and diarization outputs with confidences; LID posteriors; route; both ASR hypotheses when shadow-run; post-processing diff; latency.
2. Mine errors using reference-free signals: low ASR confidence; cross-ASR disagreement; LID margin near the threshold; diarization speaker-count anomalies; user corrections. Use oracle substitution on the labeled subset for stage attribution.
3. Send top-uncertainty segments to human transcription, then fold them into a stratified, versioned eval set (by language, field, domain, overlap level), keeping a frozen test split that the proposer never sees.
4. Fix only the attributed module. Gate on no regression per stratum, not only on the aggregate, since aggregate WER hides switch-point errors (see Q2).
5. Roll out in stages via shadow mode, then canary.

### Gaps
- I found no published flywheel case study for a speech or ASR pipeline with per-stage attribution numbers. NVInfo is text-only.
- The details of NVInfo's stage-attribution method and rollout were not retrieved; only the abstract was read.

---

## Q5. Applications to speech and audio pipelines

### Takeaway
Very little exists. No retrieved paper applies DSPy, GEPA, Optimas, Trace or Meta-Harness to a diarization + LID + ASR cascade. The closest items are:
- challenge-system engineering reports such as HINTT, where hand-tuned cascades with Qwen3-ASR beat unified speech LLMs;
- joint-model papers that argue against cascade error propagation;
- latency-focused voice-agent engineering.

This is a genuine gap, and an opportunity for the user's harness.

### Cited Findings
- **HINTT (MLC-SLM 2026):** cascade with Qwen3-ASR plus LLM GEC beats unified VibeVoice-ASR (15.28 vs. 18.50 tcpWER); module-wise fine-tuning; GEC worth about 0.1–0.2 absolute. [V] — [arXiv 2610.08063](https://arxiv.org/html/2610.08063)
- **Voice agents (STT→LLM→TTS):** published optimization guidance is about latency, not accuracy.
  - Streaming the stages cuts 400–800 ms from P95 turn latency (vendor blog).
  - Speculative LLM prefill on eager end-of-turn trades extra calls for latency (Deepgram).
  - No DSPy-style automatic optimization of voice pipelines was found.
  - [vendor sources] — [FutureAGI](https://futureagi.com/blog/how-to-optimize-voice-agent-latency-2026/), [Deepgram](https://deepgram.com/blog/voice-agent-architecture-stt-llm-tts-pipeline-design)
- **Workflow-optimization survey 2026** ("From Static Templates to Dynamic Runtime Graphs", arXiv 2603.22386): a general survey of agent workflow optimization; no speech-specific section was identified. [snippet] — [arXiv](https://arxiv.org/pdf/2603.22386)
- The foundations file covers NVIDIA Voice Memory, which is the only harness-RSI loop on ASR (error correction only, not a full pipeline).

### Inferences
- A defensible novelty claim for the project is an Optimas- or ADOPT-style multi-module optimizer for a speech cascade. It would use:
  - native local metrics (DER, LID accuracy, per-route WER) with measured local–global alignment;
  - Optuna for thresholds;
  - a coding-agent proposer for routing and post-processing code;
  - oracle-substitution credit assignment;
  - stratified regression gating.

### Gaps
- Speech-translation cascade optimization (ASR→MT error propagation, 2025–26) was not searched in this session, because the tool-call budget was exhausted.
- No speech-specific end-to-end pipeline optimizer papers were found. Possibly some exist under terms not tried, such as "pipeline-aware ASR tuning" or "system combination search".
