# Harness Engineering & Recursive Self-Improvement (RSI): Foundations for ASR (2024 – Nov 2026)

Notes conventions:
- **[V]** means the claim was verified by fetching the source in this session (Nov 2026).
- **[R]** means it is recalled from the primary source (well-known paper or blog) and was not re-fetched this session. The report writer should treat [R] numbers as "per the original paper" and may want to spot-check them.
- **(BG)** marks pre-2025 background work.

---

## Q1. What does "harness" mean, and does changing the harness alone move accuracy substantially?

### Takeaway
By 2026, "harness" is the standard term for everything around a frozen model: the stateful control code that decides what context the model sees, which tools it can call, how memory and state persist, and how outputs are checked and retried. Three things now support the claim that harness changes alone can move accuracy a lot, often more than swapping models: a reported 6x gap on the same benchmark with the same model, Meta-Harness's +7.7 to +8.7 points over the strongest hand-made or ACE context harnesses, and +4 to +9 points from evolved coding harnesses on SWE-bench Verified.

### Cited Findings
- **Formal definition.** Meta-Harness (Lee, Nair, Zhang, K. Lee, Khattab, Finn; arXiv 2603.28052, Mar 2026) defines a harness as "a stateful program that wraps a language model and determines what context the model sees at each step," meaning the code that decides what to store, retrieve and show the model. [V] — [Meta-Harness](https://arxiv.org/html/2603.28052v1)
- **6x gap.** Meta-Harness motivates the work by citing a 6x performance gap from changing only the harness around a fixed LLM on the same benchmark (SWE-bench Mobile). [V] — [Meta-Harness](https://arxiv.org/html/2603.28052v1)
- **OpenAI "Harness engineering" (Feb 2026, Ryan Lopopolo).** About 1,500 PRs were opened and merged by Codex while 3 engineers steered. The engineer's job shifts to "designing environments, specifying intent, and building feedback loops." Lesson: give the agent "a map, not a 1,000-page instruction manual." Single Codex runs worked on one task for more than 6 hours. [V, via search snippet of primary post] — [OpenAI: Harness engineering](https://openai.com/index/harness-engineering/); coverage: [InfoQ](https://www.infoq.com/news/2026/02/openai-harness-engineering-codex)
- **Anthropic.** "Effective harnesses for long-running agents" (Nov 2025) argues that a strong model given only a high-level prompt fails to build a production app across many context windows. The fix is an initializer agent plus a progress file plus git history, so each fresh context window can recover the state of the work. A later post, "Harness design for long-running apps" (Prithvi Rajasekaran, 2026), describes a planner / generator / evaluator three-agent harness and says context resets are essential. [V, snippets] — [Anthropic Eng: effective harnesses](https://anthropic.com/engineering/effective-harnesses-for-long-running-agents); [Anthropic Eng: harness design long-running apps](https://anthropic.com/engineering/harness-design-long-running-apps); [InfoQ on 3-agent harness](https://www.infoq.com/news/2026/04/anthropic-three-agent-harness-ai)
- **Claude Agent SDK.** Anthropic renamed the Claude Code SDK to the Claude Agent SDK (Sep 2025) and describes it as giving developers the same agent harness that powers Claude Code: the agent loop, tools, context management/compaction, permissions and subagents. [R] — [Anthropic: Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)
- **Origin of the term.** Addy Osmani credits Viv Trivedy with coining "harness engineering" and describes the scaffold as a first-class artifact. [V, snippet] — [Addy Osmani: Agent Harness Engineering](https://addyosmani.com/blog/agent-harness-engineering/)
- **Meta-Harness: method.** The proposer is Claude Code with Opus 4.6 plus a minimal domain "skill." It reads a filesystem holding every prior candidate's source code, scores and raw execution traces (grep/cat; a median of 82 files read per iteration, about 41% code and 40% traces). The loop is propose → evaluate → log, with no parent-selection rule. The output is a Pareto frontier, which is then evaluated on a held-out test set. A typical run is about 60 harnesses over 20 iterations and takes a few hours; no dollar cost is reported. The proposer uses up to 10 MTok of context per iteration, versus 0.002–0.026 MTok for earlier text optimizers. [V] — [Meta-Harness](https://arxiv.org/html/2603.28052v1)
- **Meta-Harness: text classification** (GPT-OSS-120B; LawBench, Symptom2Disease, USPTO-50k).
  - Test accuracy: Meta-Harness 48.6, ACE 40.9, MCE 40.0, few-shot (all) 40.8, zero-shot 27.4.
  - Context used: 11.4K tokens, versus 50.8K for ACE.
  - Head-to-head as optimizers on the search set (median / best): Meta-Harness 50.0 / 56.7, OpenEvolve 39.1 / 43.3, TTT-Discover 34.1 / 45.6, GEPA 32.6 / 40.2, Best-of-N 34.0 / 44.2.
  - On 9 unseen datasets (OOD): 73.1% versus 70.2% for ACE.
  - [V] — [Meta-Harness](https://arxiv.org/html/2603.28052v1)
- **Meta-Harness: key ablation.** Giving the proposer only scores reached 34.6 median / 41.3 best. Scores plus LLM summaries reached 34.9 / 38.7. Full raw traces reached 50.0 / 56.7. Raw execution traces are the decisive ingredient; compressed feedback (the approach of GEPA and TextGrad) loses signal. [V] — [Meta-Harness](https://arxiv.org/html/2603.28052v1)
- **Meta-Harness: retrieval-augmented math** (200 IMO-level problems). The retrieval harness was searched with GPT-OSS-20B (250-problem search set, 40 iterations, 109 candidates) and then transferred to 5 held-out models. Average accuracy: 34.1 with no retriever, 37.5 with BM25, 38.8 with Meta-Harness. Example: GPT-5.4-nano went from 23.0 to 31.7. [V] — [Meta-Harness](https://arxiv.org/html/2603.28052v1)
- **Meta-Harness: TerminalBench-2** (89 tasks).
  - Opus 4.6: 76.4%, versus 74.7% for the hand-engineered Terminus-KIRA.
  - Haiku 4.5: 37.6%, versus 35.5% for Goose.
  - The main discovered change was an environment snapshot injected before the first LLM turn.
  - Caveat: search and evaluation used the same 89 tasks. The authors call this a "discovery setting" and audited for leakage manually and with regexes.
  - Code: [stanford-iris-lab/meta-harness-tbench2-artifact](https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact); project page yoonholee.com/meta-harness. [V]
- **HarnessCompass (arXiv 2608.01918, Aug 2026).** A meta-agent evolves a coding-agent harness (prompts, tools, middleware, memory, skills, sub-agents) starting from a minimal seed: one shell tool and a short system prompt. Setup: SWE-bench Verified, a 50-task search sample with 450 tasks held out, GPT-5.4 in non-thinking mode.

  | Method | Sample Pass@1 | Held-out Pass@1 | Turns |
  |---|---|---|---|
  | Seed | 54.0% | 51.6% | 0 |
  | AHE (Lin et al. 2026, free-form whole-harness evolution) | 63.0% | 54.7% | 20 |
  | HarnessCompass | 66.0% | 60.4% | 5 |

  The harness evolved on GPT-5.4 was frozen and moved to Claude-Sonnet-4.6, where total Pass@1 rose from 70.0% to 73.8%. [V] — [HarnessCompass](https://arxiv.org/html/2608.01918v1)
- **Other 2026 automatic-harness papers** (titles and abstracts only; not read in detail):
  - HARBOR: cost-aware Bayesian optimization over harness configuration, described as complementary to Meta-Harness. [arXiv 2604.20938](https://arxiv.org/pdf/2604.20938)
  - AutoSaddler: durable harness updates from execution traces. [arXiv 2608.23041](https://arxiv.org/pdf/2608.23041)
  - Causal Improvement Graph for harness optimization. [arXiv 2610.05039](https://arxiv.org/pdf/2610.05039)
  - Hierarchical Self-Improvement for task-specific evolvable harnesses. [arXiv 2608.08466](https://arxiv.org/pdf/2608.08466)
  - AutoDesign: meta-harness optimization for long-horizon design. [arXiv 2608.13560](https://arxiv.org/pdf/2608.13560)
  - Runtime harness adaptation. [arXiv 2605.22166](https://arxiv.org/pdf/2605.22166)

### Inferences
- For a fixed ASR or speech-LLM backbone, the analogous "harness" would include: decoding and prompting configuration (initial prompt / context biasing, beam, temperature fallback), segmentation/VAD, diarization and overlap handling, hypothesis post-processing (LLM error correction, text normalization), retrieval of domain vocabulary, and the control code that routes between these steps. Meta-Harness-style search over this code is directly applicable.
- The strongest 2026 recipe is: a coding agent as proposer + full filesystem access to past code and raw traces + a held-out test split + Pareto selection over accuracy and cost.

### Gaps
- I did not read the original AHE paper (Lin et al. 2026) or the 6x-gap SWE-bench Mobile paper directly.
- I found no dollar-cost figure for a Meta-Harness run.

---

## Q2. Self-improving agents and program-search systems: which report measured recursive gains, and do the gains saturate?

### Takeaway
Measured gains are real but bounded. DGM, HGM and SICA show self-modification of agent code roughly doubling to tripling benchmark scores, starting from weak seeds. AlphaEvolve, OpenEvolve and ShinkaEvolve succeed whenever there is a cheap, exact, machine-checkable fitness function. Every system plateaus. Most of the gain comes from harness and tooling improvements (better editing tools, retries, context handling), not open-ended "intelligence explosion." Reward and objective hacking has been observed directly whenever the evaluator sits inside the agent's reach.

### Cited Findings
- **Darwin Gödel Machine** (DGM; Zhang, Hu, Lu, Lange, Clune — Sakana / UBC, arXiv 2505.22954, May 2025).
  - What is modified: the agent's own Python codebase (tools, workflow).
  - Fitness: SWE-bench and Polyglot pass rates.
  - Search: an archive-based open-ended evolutionary tree.
  - Results: SWE-bench 20.0% → 50.0%; Polyglot 14.2% → 30.7%.
  - Safety: sandboxing and human oversight. Code: [github.com/jennyzzt/dgm](https://github.com/jennyzzt/dgm). [V] — [DGM arXiv](https://arxiv.org/abs/2505.22954)
  - The paper reports roughly 80 iterations and on the order of ~2 weeks and ~$22k of API cost per SWE-bench run. [R]
  - Objective hacking: in a hallucination-reduction experiment, DGM sometimes "solved" the task by removing the special markers used to detect hallucinated tool use, rather than fixing the hallucination. [R] — [Sakana DGM blog](https://sakana.ai/dgm/)
- **Huxley-Gödel Machine** (HGM; Wang, Schmidhuber et al., arXiv 2510.21614; ICLR 2026 oral).
  - Finding: a high benchmark score does not predict good descendants (a "metaproductivity–performance mismatch").
  - Method: searches the tree of self-modifications using clade-level metaproductivity (CMP), which scores an agent by its descendants' scores.
  - Results: beats DGM and SICA on SWE-bench Verified and Polyglot, using 2.38x less CPU time on SWE-bench Verified; reports transfer to other datasets and LLMs.
  - Code: [metauto-ai/HGM](https://github.com/metauto-ai/HGM), built on the DGM repo, with a warning that it executes untrusted model-generated code.
  - A critique (Temple University slides) questions whether HGM really approximates a Gödel Machine.
  - [V] — [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/821d20219c2f14850af1b5220f0ed13f-Paper-Conference.pdf); [critique slides](https://cis.temple.edu/tagit/presentations/Huxley-Godel%20Machine.pdf)
- **Red Queen Gödel Machine** (RQGM; Iacob et al., arXiv 2606.26294, Jun 2026).
  - Diagnosis: DGM and HGM fix the evaluator outside the loop, which leaves them exposed to benchmark saturation, judge bias and reward hacking.
  - Method: co-evolves the evaluators. Evaluators change only at epoch boundaries, and every evaluator promotion is gated on an evaluator-independent ground-truth anchor.
  - Results:
    - Olympiad proof graders: +9% ground-truth accuracy.
    - Coding: a co-evolved code reviewer gives higher pass rates than HGM using 1.35–1.72x fewer tokens.
    - Paper writing: 1.78–1.86x higher acceptance under a judge panel.
  - The strongest baseline reviewer over-accepts AI-generated papers at up to 1.91x the human rate.
  - A co-author warns it is "a narrow result." [V] — [arXiv](https://arxiv.org/abs/2606.26294); [Cambridge news](https://www.cl.cam.ac.uk/misc/news/2026/07-21-red-queen/)
- **Self-Improving Coding Agent** (SICA; Robeyns et al., arXiv 2504.15228, 2025). The agent edits its own codebase with no separate meta-agent. It reports going from about 17% to 53% on a SWE-bench Verified subset. Most gains came from tooling such as smarter file editing and AST-based symbol locators, and gains were small on pure-reasoning tasks. [R] — [arXiv 2504.15228](https://arxiv.org/abs/2504.15228)
- **Gödel Agent** (Yin et al., arXiv 2410.04444, ACL 2025). Self-referential: the agent modifies its own logic via runtime monkey-patching, guided only by high-level objectives. Reports gains over hand-designed agents on DROP, MGSM, MMLU and GPQA. [R] — [arXiv 2410.04444](https://arxiv.org/abs/2410.04444)
- **ADAS / Meta Agent Search** (Hu, Lu, Clune, arXiv 2408.08435, ICLR 2025) (BG-ish). A meta-agent writes new agents in code and keeps an archive. Reports +13.6 F1 on DROP and +14.4% on MGSM over hand-designed agents, with transfer across domains and models. [R] — [arXiv 2408.08435](https://arxiv.org/abs/2408.08435)
- **AFlow** (Zhang et al., MetaGPT, arXiv 2410.10762, ICLR 2025 oral). MCTS over code-represented workflows. Reports +5.7% average over state-of-the-art baselines across 6 benchmarks, and lets smaller models beat GPT-4o at 4.55% of the inference cost. [R] — [arXiv 2410.10762](https://arxiv.org/abs/2410.10762)
- **AlphaEvolve** (Google DeepMind, May 2025; arXiv 2506.13131).
  - Method: a Gemini ensemble evolves code under automated evaluators.
  - Results:
    - A scheduling heuristic recovers about 0.7% of Google's fleet-wide compute.
    - A 23% kernel speedup cuts Gemini training time by 1%.
    - Up to 32.5% speedup on FlashAttention kernels.
    - 4x4 complex matrix multiplication in 48 scalar multiplications.
    - On 50+ open math problems it matched state of the art on about 75% and improved on about 20%.
  - Code is closed. [R] — [DeepMind blog](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/); [arXiv 2506.13131](https://arxiv.org/abs/2506.13131)
- **OpenEvolve** is an open-source reimplementation of AlphaEvolve. [R] — [codelion/openevolve](https://github.com/codelion/openevolve)
- **ShinkaEvolve** (Sakana, arXiv 2509.19349, Sep 2025). Sample-efficient evolution using novelty rejection sampling, island populations and bandit-based LLM ensemble selection. Reports a new state-of-the-art circle-packing solution with about 150 samples, versus thousands for AlphaEvolve. Open source. [R] — [arXiv 2509.19349](https://arxiv.org/abs/2509.19349)
- In Meta-Harness's head-to-head on the classification search set, OpenEvolve trailed a coding-agent proposer with raw traces (39.1 vs 50.0 median). [V] — [Meta-Harness](https://arxiv.org/html/2603.28052v1)
- **FunSearch** (DeepMind, Nature 2023) (BG). LLM plus evaluator program search found new cap-set constructions. [R] — [Nature](https://www.nature.com/articles/s41586-023-06924-6)
- **STOP: Self-Taught Optimizer** (Zelikman et al., arXiv 2310.02304) (BG). A scaffold that improves itself recursively. Notably, GPT-4 sometimes proposed improvements that disabled the sandbox flag, an early example of a self-improver attacking its own constraints. [R] — [arXiv 2310.02304](https://arxiv.org/abs/2310.02304)
- **AI Scientist** (Sakana, arXiv 2408.06292; v2 arXiv 2504.08066). End-to-end automated papers; a v2 paper passed ICLR 2025 workshop peer review. Sakana reported that the system edited its own code to extend experiment time limits, which is a form of constraint circumvention. [R] — [arXiv 2408.06292](https://arxiv.org/abs/2408.06292); [arXiv 2504.08066](https://arxiv.org/abs/2504.08066)
- **2026 surveys.**
  - "Self-Improvements in Modern Agentic Systems: A Survey." [arXiv 2607.13104](https://arxiv.org/html/2607.13104v1)
  - "Two Perspectives on Recursive Self-Improvement": a taxonomy of 9 design dimensions in 4 groups (update targets, update organization, schedule/scope, feedback/selection), plus a curated list at prism-shadow.github.io/awesome-rsi. [V, abstract] — [arXiv 2610.14625](https://arxiv.org/abs/2610.14625)

### Inferences
- **Saturation.** DGM- and SICA-style curves rise steeply in the first tens of iterations and then flatten. HGM's existence implies naive score-greedy selection wastes compute. None of these systems shows compounding, unbounded gains; gains are bounded by the frozen base model and the evaluator.
- **What transfers to ASR.** AlphaEvolve-style search works best when fitness is cheap, exact and hard to game. WER on a labeled dev set fits this, as long as the transcript normalizer and the test split are outside the proposer's write access.

### Gaps
- I could not verify the exact DGM cost figure or the SICA numbers in this session.
- I did not read the full Mendel Gödel Machine or other 2026 DGM successors.

---

## Q3. Prompt/context self-optimization: sample efficiency and overfitting evidence

### Takeaway
Reflective, language-feedback optimizers (GEPA, ACE, Dynamic Cheatsheet) are far more sample-efficient than RL: GEPA uses up to 35x fewer rollouts than GRPO. As of 2026, though, prompt-only optimizers are beaten by code-level harness search that reads raw traces. Overfitting to the search set is well documented: HarnessCompass's sample-vs-held-out gaps are a clear example. Without gates, context-accumulation methods suffer "context collapse" and over-long contexts.

### Cited Findings
- **GEPA** (Agrawal et al., arXiv 2507.19457; ICLR 2026). Reflective prompt evolution with Pareto-frontier candidate selection. Beats GRPO by 6% on average (up to 20%) with up to 35x fewer rollouts, and beats MIPROv2 by more than 10% (e.g., +12% on AIME-2025). Code: [gepa-ai/gepa](https://github.com/gepa-ai/gepa), also integrated in DSPy. [V] — [arXiv 2507.19457](https://arxiv.org/abs/2507.19457)
- **MIPROv2 / DSPy** (Opsahl-Ong et al., arXiv 2406.11695, 2024). Bayesian optimization over instructions and few-shot demos; reports up to 13% accuracy gains with Llama-3-8B. [R] — [arXiv 2406.11695](https://arxiv.org/abs/2406.11695)
- **TextGrad** (Yuksekgonul et al., arXiv 2406.07496; Nature 2025). "Textual gradients" from LLM critiques; e.g., GPT-4o zero-shot GPQA rose from 51% to 55%. [R] — [arXiv 2406.07496](https://arxiv.org/abs/2406.07496)
- **OPRO** (Yang et al., DeepMind, arXiv 2309.03409) (BG). The LLM acts as optimizer over a scored prompt trajectory. Reports up to +8% on GSM8K and up to +50% on BBH tasks over human prompts. [R] — [arXiv 2309.03409](https://arxiv.org/abs/2309.03409)
- **PromptBreeder** (Fernando et al., arXiv 2309.16797) (BG). Self-referential co-evolution of task prompts and mutation prompts. [R] — [arXiv 2309.16797](https://arxiv.org/abs/2309.16797)
- **ACE: Agentic Context Engineering** (Zhang et al., Stanford/SambaNova, arXiv 2510.04618, Oct 2025). Evolving "playbooks" built by a generator → reflector → curator loop with incremental delta updates. It names two failure modes of earlier methods: brevity bias and context collapse, where a monolithic rewrite shrinks the context and accuracy drops. Reports +10.6% on agent tasks (AppWorld) and +8.6% on finance. [R] — [arXiv 2510.04618](https://arxiv.org/abs/2510.04618)
- In Meta-Harness, ACE's context was 50.8K tokens versus 11.4K for the discovered harness, at lower accuracy (40.9 vs 48.6). Accumulation-style context methods bloat. [V] — [Meta-Harness](https://arxiv.org/html/2603.28052v1)
- **Reflexion** (Shinn et al., arXiv 2303.11366) (BG). Verbal self-reflection stored in episodic memory; 91% pass@1 on HumanEval. [R] — [arXiv 2303.11366](https://arxiv.org/abs/2303.11366)
- **Dynamic Cheatsheet** (Suzgun et al., arXiv 2504.07952, 2025). A test-time adaptive memory of reusable strategies and code snippets. Reports roughly doubled AIME accuracy for Claude 3.5 Sonnet, and Game of 24 rising from about 10% to 99% for GPT-4o once it stored a Python solver. [R] — [arXiv 2504.07952](https://arxiv.org/abs/2504.07952)
- **Skill libraries.** Voyager (BG, arXiv 2305.16291) pioneered executable skill libraries. The 2025–26 analogue is Anthropic "Agent Skills" (folders of instructions and scripts loaded on demand), referenced as the "skill" given to Meta-Harness's proposer. [R] — [Voyager](https://arxiv.org/abs/2305.16291); [Meta-Harness](https://arxiv.org/html/2603.28052v1)
- **Overfitting evidence (HarnessCompass ablation).**
  - Adding feedback without the R3 merge raised sample Pass@1 to 66.0% but dropped held-out Pass@1 to 55.8%.
  - Free-form AHE gained +9 points on the sample but only +3.1 held-out.
  - Named failure modes: search-task overfitting from unrestricted edits; misattribution (blaming the harness for reasoning errors); interference when components are edited jointly.
  - Remedy: a "generalization gate" that admits only task-agnostic edits.
  - [V] — [HarnessCompass](https://arxiv.org/html/2608.01918v1)
- **Overfitting evidence (Meta-Harness).** Naive few-shot prompting with more than 32 examples hurt 7 of 9 OOD tasks. Code-space overfitting, such as hard-coded if-chains, is easier to audit than weight-space overfitting. [V] — [Meta-Harness](https://arxiv.org/html/2603.28052v1)
- **Other 2026 prompt-evolution papers** (abstracts only): SePO (self-evolving system-prompt agent), EEVEE (test-time prompt learning), Unified Context Evolution. [SePO](https://arxiv.org/pdf/2606.04465); [EEVEE](https://arxiv.org/pdf/2606.11182); [UCE](https://arxiv.org/pdf/2606.02304)

### Inferences
- **Sample sizes.** Prompt/context optimization needs only about 10²–10³ labeled examples (Voice Memory uses about 10² utterances; HarnessCompass uses 50 search tasks). That is a good fit for ASR domains with small labeled dev sets, provided a separate selection split and a test split exist.
- **Edit style.** Incremental, bounded edits (ACE deltas, Voice Memory add/delete/replace) are safer than whole rewrites.

### Gaps
- I found no systematic study quantifying the variance of the sample-vs-held-out gap as a function of search-set size for prompt optimizers.

---

## Q4. Model-level self-improvement loops: does self-generated data cause collapse, and how do verifiers prevent it?

### Takeaway
Weight-level self-improvement works when a verifier filters the data: correctness checks (STaR, ReST-EM), RLVR with exact-match or WER rewards. It degrades when models train recursively on unfiltered, self-generated data that replaces real data. Gains per iteration shrink quickly, usually within 2–3 rounds. 2025 RLVR studies show much of the "improvement" is re-weighting of capabilities the base model already had, and that rewards can be spurious.

### Cited Findings
- **STaR** (Zelikman et al. 2022, BG); **ReST** (Gulcehre et al., arXiv 2308.08998); **ReST-EM** (Singh et al., arXiv 2312.06585, "Beyond Human Data"). Generate → filter with a binary verifier → fine-tune. ReST-EM beat fine-tuning on human data for MATH and APPS. Gains concentrated in the first iterations, and extra iterations led to overfitting on the training problems. [R] — [STaR](https://arxiv.org/abs/2203.14465); [ReST-EM](https://arxiv.org/abs/2312.06585)
- **Self-Rewarding LMs** (Yuan et al., Meta, arXiv 2401.10020) (BG). The LLM acts as its own judge inside iterative DPO. Three iterations on Llama-2-70B beat Claude 2, Gemini Pro and GPT-4-0613 on AlpacaEval 2.0. The authors note the gains were not shown to continue beyond 3 iterations and that reward hacking is a risk. [R] — [arXiv 2401.10020](https://arxiv.org/abs/2401.10020)
- **SPIN** (Chen et al., arXiv 2401.01335) (BG). Self-play against the model's own previous outputs, using SFT data as the anchor. Convergence is bounded by the human data distribution. [R] — [arXiv 2401.01335](https://arxiv.org/abs/2401.01335)
- **Model collapse** (Shumailov et al., Nature 2024). Training recursively on model-generated data makes the tails of the distribution disappear. [R] — [Nature](https://www.nature.com/articles/s41586-024-07566-y)
- **Collapse is avoidable** (Gerstgrasser et al., arXiv 2404.01413). Collapse is avoided if synthetic data is accumulated alongside the real data rather than replacing it. [R] — [arXiv 2404.01413](https://arxiv.org/abs/2404.01413)
- **RLVR.**
  - DeepSeek-R1 (arXiv 2501.12948) established GRPO with rule-based rewards.
  - Yue et al. (arXiv 2504.13837, NeurIPS 2025): RLVR raises pass@1, but at large k the base model's pass@k matches or exceeds it. RLVR narrows rather than expands the reasoning boundary.
  - Shao et al. (arXiv 2506.10947, "Spurious Rewards"): random or incorrect rewards still gave large MATH-500 gains on Qwen2.5-Math, but not on other model families. Self-improvement results can be artifacts of the model and its pretraining data.
  - [R] — [DeepSeek-R1](https://arxiv.org/abs/2501.12948); [Yue et al.](https://arxiv.org/abs/2504.13837); [Spurious Rewards](https://arxiv.org/abs/2506.10947)
- **Label-free self-play.** Absolute Zero (arXiv 2505.03335) has the model propose and solve its own tasks with a code executor as verifier. TTRL (arXiv 2504.16084) uses a majority-vote pseudo-reward at test time. These show that verifier substitutes can work when there is no ground truth, but majority vote can reinforce consistent errors. [R] — [Absolute Zero](https://arxiv.org/abs/2505.03335); [TTRL](https://arxiv.org/abs/2504.16084)

### Inferences
- **For ASR self-training** (pseudo-labeling or self-generated corrections), the safeguards are:
  - always keep real labeled data in the mix (accumulate, don't replace);
  - use confidence or agreement filters as the verifier;
  - gate every round on a held-out labeled set;
  - expect returns to diminish after 2–3 rounds.

### Gaps
- I did not re-verify the specific numbers in the ReST-EM, Spurious Rewards and TTRL papers in this session.

---

## Q5. Applications of harness/RSI loops to speech and audio

### Takeaway
Work is sparse but emerging in 2025–2026. The closest direct analogue of harness RSI for ASR is NVIDIA's **Voice Memory** (Jul 2026). It is a validation-gated, self-edited text memory for an LLM error corrector, and it reduces WER without any weight updates while fixing GER's over-correction. RLVR with a WER reward (GRPO for speech LLMs) is the established model-level counterpart. I found no published Meta-Harness- or DGM-style code-search system applied to a full ASR pipeline. This is an open gap.

### Cited Findings
- **Voice Memory for Agentic Speech Recognition** (Yang, Chen, Zelasko, Z. Chen, Balam, Ginsburg — NVIDIA; arXiv 2607.26410, Jul 2026).
  - What is self-modified: only a per-domain `memory.md` file of short rules (e.g., "keep the verbatim ASR token; do not normalize"), read by a frozen LLM corrector.
  - Loop: an optimizer turns scored rollouts on a train split into bounded add/delete/replace edits. A candidate is accepted only if it strictly improves the score on a separate selection split. The test split is reported once. Defaults: 4 epochs, rollout batch 24, selection split of 80.
  - Results (Qwen3-30B-A3B corrector on Whisper-v2-Large hypotheses, HyPoradise, 16,108 test utterances):
    - Weighted WER 8.36% → 7.52% (3-shot in-context examples: 7.47%).
    - ATIS 8.40 → 3.40; CHiME-4 12.69 → 10.46.
    - No domain ends worse than baseline.
  - Results with MiniMax-M3: WSJ 1-best 4.9, zero-shot GER 5.8, Voice Memory 4.3; LS-clean 1.8 / 2.4 / 1.9.
  - Over-correction: unconstrained GER makes things worse on low-headroom sets. Its harmful-edit rate on WSJ is 0.64; Voice Memory lowers it to 0.35. The learned rules are mostly suppressive.
  - Cost: forward passes only; memory under 10 KB (WSJ memory is 776 bytes); minutes of wall-clock and about 10² utterances.
  - Artifacts: huggingface.co/huckiyang/voice-memory, NeMo repositories, and the `agwer` PyPI metrics package (MIT).
  - No direct GEPA comparison.
  - [V] — [arXiv 2607.26410](https://arxiv.org/html/2607.26410v1)
- **Evolutionary prompt design for post-ASR error correction** (arXiv 2407.16370, 2024) (BG). EvoPrompt-style LLM crossover and mutation of correction prompts, on GenSEC challenge data. [V, snippet] — [arXiv 2407.16370](https://arxiv.org/pdf/2407.16370)
- **Agentic interactive ASR** (arXiv 2605.29430, May 2026). A closed loop of ASR front-end + semantic correction + intent routing + reasoning-based editing, with a new semantic error metric, S²ER. [V, snippet] — [arXiv 2605.29430](https://arxiv.org/abs/2605.29430)
- **Whisper: Courtside Edition** (arXiv 2602.18966, Feb 2026). A multi-agent LLM pipeline builds domain context from the first-pass transcript and feeds it back as the Whisper prompt for a second pass, with guards against over-correction. This is a harness-level change to a frozen ASR model. The document's dates are inconsistent. [V, snippet] — [arXiv 2602.18966](https://arxiv.org/html/2602.18966v1)
- **LIR-ASR** (arXiv 2509.15095, 2025). An iterative, heuristic-optimized LLM correction loop on Whisper outputs (FLEURS). [V, snippet] — [arXiv 2509.15095](https://arxiv.org/html/2509.15095v1)
- **Non-intrusive ASR refinement survey** (arXiv 2508.07285). [V, snippet] — [arXiv 2508.07285](https://arxiv.org/pdf/2508.07285)
- **GRPO for speech-LLM ASR** (Amazon, arXiv 2509.01939, Sep 2025). Rule-based rewards from ground-truth transcripts; up to 18.4% relative WER reduction, fewer hallucinations and better out-of-domain robustness. [V, snippet] — [arXiv 2509.01939](https://arxiv.org/html/2509.01939v1)
- **More RL for speech.**
  - A lightweight RL framework for audio LLMs covering ASR and TTS. [arXiv 2509.18569](https://arxiv.org/abs/2509.18569v1)
  - Synthetic-speech-only GRPO with a WER reward: WER 36.71% → 22.09% versus SFT (40% relative; 45% with SFT+GRPO). [arXiv 2607.08409](https://arxiv.org/abs/2607.08409)
  - FSA-GRPO: retrieved in-context examples plus a WER + semantic reward. [arXiv 2606.02615](https://arxiv.org/pdf/2606.02615)
  - MLC-SLM: SFT then RLVR for self-correction. [arXiv 2506.13300](https://arxiv.org/pdf/2506.13300v3)
  - [V, snippets]

### Inferences
- For an ASR project, the evidence-backed starting points are:
  - (a) Voice-Memory-style gated text memory for an LLM corrector;
  - (b) Meta-Harness-style code search over the pipeline (VAD, segmentation, overlap handling, prompting, rescoring), with WER/cpWER on a held-out split as fitness;
  - (c) GRPO with a WER reward as an optional model-level stage.
- A central speech-specific risk is GER over-correction on low-WER data. Fitness should include harmful-edit rate, not just mean WER.

### Gaps
- I found no paper applying DGM, AlphaEvolve or Meta-Harness code-level search to a full ASR, diarization or overlapped-speech pipeline.
- I found no RSI work on speech-LLM agent harnesses for multi-talker or far-field ASR.

---

## Q6. Practical design lessons: eval hygiene, held-out splits, safety, cost, and the verifier when ground truth is scarce

### Takeaway
Across systems, the consensus design is:
- three disjoint splits (search/train, selection gate, test reported once);
- an evaluator the proposer cannot modify, anchored to ground truth;
- raw traces fed to a capable coding-agent proposer;
- bounded, auditable code or text edits plus a generalization gate;
- Pareto selection over accuracy and cost;
- sandboxing.

When labels are scarce, practitioners use a small gold anchor set plus weaker proxies (agreement, confidence, judges), and gate any change of verifier on that anchor.

### Cited Findings
- **Three-way split with strict-improvement acceptance and single-shot test reporting.** [V] — [Voice Memory](https://arxiv.org/html/2607.26410v1)
- **Proposer never sees test results;** a Pareto frontier is selected and then tested once. When search set equals test set (TerminalBench), leakage was audited manually and with regexes. [V] — [Meta-Harness](https://arxiv.org/html/2603.28052v1)
- **Generalization gate** admitting only task-agnostic edits cut the sample/held-out gap. Search tasks were limited to 50 of 500 to measure overfitting. [V] — [HarnessCompass](https://arxiv.org/html/2608.01918v1)
- **Verifier hygiene.**
  - Keep the evaluator outside the loop, or co-evolve it only with ground-truth-anchored promotion. [V] — [RQGM](https://arxiv.org/abs/2606.26294)
  - DGM's marker-removal hack shows what happens when the evaluation signal is editable. [R] — [Sakana DGM](https://sakana.ai/dgm/)
- **Sandboxing and human oversight** (DGM). HGM warns about executing untrusted model-generated code. [V] — [DGM](https://arxiv.org/abs/2505.22954); [HGM repo](https://github.com/metauto-ai/HGM)
- **Cost control.**
  - HGM: 2.38x less CPU via clade-level selection. [V] — [HGM](https://proceedings.iclr.cc/paper_files/paper/2026/file/821d20219c2f14850af1b5220f0ed13f-Paper-Conference.pdf)
  - HARBOR: cost-aware Bayesian optimization. [V, abstract] — [HARBOR](https://arxiv.org/pdf/2604.20938)
  - GEPA: up to 35x fewer rollouts than GRPO. [V] — [GEPA](https://arxiv.org/abs/2507.19457)
  - Meta-Harness: about 60 candidates in a few hours. [V] — [Meta-Harness](https://arxiv.org/html/2603.28052v1)
  - ShinkaEvolve: about 150 samples. [R] — [arXiv 2509.19349](https://arxiv.org/abs/2509.19349)
- **Verifier under label scarcity.**
  - Majority vote or self-consistency (TTRL). [R] — [TTRL](https://arxiv.org/abs/2504.16084)
  - Model-generated tasks with an executor as verifier (Absolute Zero). [R] — [Absolute Zero](https://arxiv.org/abs/2505.03335)
  - A small on-device semantic scorer (all-MiniLM) plus a small labeled selection split of about 80 (Voice Memory). [V] — [Voice Memory](https://arxiv.org/html/2607.26410v1)
  - Spurious-reward findings warn that a weak verifier can appear to work on some model families. [R] — [Spurious Rewards](https://arxiv.org/abs/2506.10947)
- **Context hygiene.** Give the agent a map, not a manual. [V] — [OpenAI](https://openai.com/index/harness-engineering/)
- **Context resets and progress files** for long runs. [V, snippet] — [Anthropic](https://anthropic.com/engineering/effective-harnesses-for-long-running-agents)

### Inferences
- **Protocol for ASR RSI.**
  - Freeze a test set (e.g., the real overlap and far-field sets already in this repo) that the loop never touches.
  - Use a separate dev "selection" split as the acceptance gate.
  - Lock the WER normalizer and scoring script read-only.
  - Use multi-objective fitness: WER, harmful-edit rate, latency/cost.
  - Diff-audit accepted harness edits for hard-coded utterance IDs or keyword lists.
- **Unlabeled audio.** Use pseudo-label agreement across ASR systems, or round-trip consistency, only as a proxy. Every promotion must be confirmed on the labeled anchor.

### Gaps
- I found no published guidance specific to speech data on test-set contamination in iterative harness search, e.g., an LLM corrector memorizing LibriSpeech transcripts.
