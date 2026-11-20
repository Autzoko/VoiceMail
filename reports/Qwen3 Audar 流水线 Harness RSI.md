# 让路由器学会"不该让谁开口"

**结论先行：** 截至 2026 年 11 月，笔记检索范围内**没有任何论文**在完整的"说话人分离 → LID → 多 ASR 路由 → 后处理"流水线外面加过 agentic harness 或自改进（RSI）闭环。已有的工作分散在三类彼此不相连的组件里：LLM 修正说话人标签（DiarizationLM、SEAL），单路 ASR 的自编辑纠错（Voice Memory、RECOVER），以及通用的多模块优化器（Optimas、MASS、ADOPT、Meta-Harness）。这些优化器从未用于含有连续阈值、非 LLM 神经模块的语音级联。对本项目来说，最有价值的 harness 不是更聪明的纠错 prompt，而是一个**风险不对称的路由控制器**，它要守住两道闸门。第一道是语种闸门：Audar 只能在 Qwen3-ASR 判定为阿拉伯语、且强制 `language Arabic` 时调用。原因是本项目实验显示，Audar 在自动语种模式下会把 63% 的 FLEURS-zh 和 82% 的 ASCEND 语句输出成阿拉伯文（多为翻译）。一个误路由的中英文片段几乎损失全部内容；而一个正确路由的阿拉伯语片段只省下约 5–15 个 WER 点。第二道是 SIR 闸门：远场重叠中占主导的 SWAP 错误在目标比干扰者响 5 dB 以上时基本消失，因此"检测低 SIR → 换输入或换条件 → 重新解码"必须是控制逻辑的一部分。RSI 应分四层：阈值/配置层（Optuna，基于缓存的双模型解码，几乎免费）、代码/路由层（Meta-Harness 式编码 agent，读取原始逐段轨迹）、上下文层（词表检索与规则记忆）、权重层（路由器蒸馏、分离伪标签、目标说话人微调）。四层共用一个提议者不可写的评测器，适应度是向量而非单一 WER：tcpWER/cpWER、Δcp、DER、按时长分桶的 LID 错误率、**输出文字不匹配率（script-mismatch rate）**、SWAP/100 词、删除率、B-WER、RTF。论文空白有四处：跨独立 ASR 模型的 LID 误路由传播从未被量化；"错语种/翻译式输出"没有标准指标；以 cpWER 为目标的自改进闭环不存在；"ASR 修正说话人标签 → 再训练分离器"的多轮闭环也没有人做过。下文严格区分文献证据和"我们的提议/估计"。

## 文献地图：每个环节都有零件，没有人把整条流水线放进闭环

先回答"是否已有论文做过"。按流水线环节梳理，结论很一致：**单环节方法成熟，跨环节控制与自改进几乎空白。** 与本项目最接近的整流水线工作是 MLC-SLM 2026 的 HINTT 系统。它的级联正是 DiariZen-FT → 微调 Qwen3-ASR → LLM GEC，dev 集 tcpWER 为 **15.28**，同时期的统一端到端模型 VibeVoice-ASR-FT 为 18.50。它的 DER 经微调从 17.04 降到 3.25，GEC 只贡献 −0.23。但它**没有做阈值消融，也没有做 oracle 分离替换**，所以误差传播的代价没有被量化 ([HINTT](https://arxiv.org/html/2610.08063))。另一个 MLC-SLM 系统把 Qwen3-ASR-1.7B 放在分离前端之后，依次做 SFT、合成语音 LoRA、GRPO（奖励为 WER/CER 加幻觉、重复、长度惩罚），dev tcpMER 为 23.70，比原版好 6.83 个点。最大收益来自 SFT，论文没有分析奖励作弊 ([2607.08208](https://arxiv.org/abs/2607.08208))。这两篇证明了"Qwen3-ASR 级联 + 模块微调"在会议场景可行，但都是人工调参的一次性工程，不是闭环。

| 环节 | 代表文献与做到的程度 | 明确**没有**做的（可发表空白） |
|---|---|---|
| 分离 / 说话人识别 | DiariZen（AMI 14.0、AliMeeting 12.5、NOTSOFAR 17.9 DER）([DiariZen](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md))；pyannote community-1（AMI-SDM 19.9）([pyannote](https://github.com/pyannote/pyannote-audio))；Audar-Diarization-V1 为 Sortformer v2.1 的 8 人扩展，宏平均 DER 22.03 为厂商数据且忽略重叠，**训练与评测都不含阿拉伯语** ([Audar-Diar card](https://huggingface.co/audarai/Audar-Diarization-V1))；唯一的阿拉伯语 DER 来自 12 个文件的厂商测试，pyannote 15.57 优于 Sortformer 离线 17.92 和流式 22.32 ([Soniqo](https://soniqo.audio/benchmarks/diarization)) | 没有公开的阿拉伯语会议分离基准；没有以下游 cpWER 为目标的分离调参研究；TS-SEP 只是指出 DER 最优与 WER 最优的分段设置方向相反 ([TS-SEP](https://arxiv.org/pdf/2303.03849)) |
| LID / 路由 | Qwen3-ASR-1.7B 的 LID 在朗读语音上为 98.7–99.9%，在对话 MLC-SLM 上为 **94.1%** ([Qwen3-ASR](https://arxiv.org/html/2601.21337))；1 秒以下 LID "远不能令人满意" ([2203.00328](https://arxiv.org/pdf/2203.00328))；模型内路由（LR-MoE、LoRA 语言专家）在 LID 错误时崩溃 ([LR-MoE](https://arxiv.org/pdf/2307.05956))；阿拉伯方言 ID 最好约 80%，域外约 56% ([NADI 2025](https://arxiv.org/pdf/2509.02038), [NADI 2026](https://arxiv.org/html/2609.27086v1)) | **没有论文量化 LID 误路由在多个独立 ASR 之间的传播**；没有"通用模型 vs 语言专家模型"基于置信度的选择研究；"错语种下的翻译式输出"只有 Whisper 论坛证据 ([OpenAI forum](https://community.openai.com/t/whisper-3-detects-incorrect-language-translates-instead-of-transcribes/828870))，没有同行评审研究，也没有指标 |
| ASR 与重叠 | CocktailASR-1 用参考语音条件化 Qwen3 解码器，AMI-SDM WER 21.81，同表 Qwen3-ASR-1.7B 为 38.18，但使用了约 100 万小时数据且未开源 ([CocktailASR](https://arxiv.org/html/2609.11274))；DiCoW/SE-DiCoW 用分离掩码条件化 Whisper，宏平均 tcpWER 相对降低 52.4% ([SE-DiCoW](https://arxiv.org/html/2601.19194)) | 没有"按 SIR 门控重解码"的系统；没有在 Qwen3-ASR 系模型上的开源目标说话人条件化 |
| 说话人标签后处理 | DiarizationLM 在 Fisher 上 WDER 相对降低 55.5% ([DiarizationLM](https://arxiv.org/abs/2401.03506))；SEAL 显示无声学条件的 LLM 会把 CHAE dev cpWER 从 15.52 **恶化**到 17.5，约束解码可防止改词 ([SEAL](https://arxiv.org/html/2501.08421v1))；纠错器对上游 ASR 有特异性 ([2406.04927](https://arxiv.org/abs/2406.04927)) | 没有中文或阿拉伯语结果，没有远场多方会议结果；**没有闭环用修正后的标签再训练分离器** |
| 单路 ASR 的 harness / 自改进 | Voice Memory：受严格改进门约束的规则记忆，加权 WER 8.36→7.52 ([Voice Memory](https://arxiv.org/html/2607.26410v1))；RECOVER 实体纠错代理 ([RECOVER](https://arxiv.org/abs/2603.16411))；Google 交互式会议说话人纠错，AMI 上 DER 相对 −31.99%，并用 LLM 模拟用户做规模化评测 ([2509.18377](https://arxiv.org/html/2509.18377v3)) | 都没有 LID，也没有多 ASR 路由；没有以 cpWER/tcpWER 为目标的 prompt/规则/配置进化 |
| 通用多模块优化 | Optimas 用局部奖励函数统一优化 prompt、超参、模型选择和权重，平均相对提升 11.92% ([Optimas](https://arxiv.org/html/2507.03041))；MASS 先局部后拓扑 ([MASS](https://arxiv.org/html/2502.02533v2))；ADOPT 依赖感知的信用分配 ([ADOPT](https://arxiv.org/html/2512.24933v2))；Meta-Harness 读原始轨迹的代码搜索 ([Meta-Harness](https://arxiv.org/html/2603.28052v1)) | **没有任何一个用于含 VAD/分离连续阈值的语音级联**；Optimas 的"超参"仍是 LLM 侧设置 |
| 生产数据飞轮 | NVInfo 的 MAPE 闭环把负反馈归因到具体阶段（路由错误 5.25%），再把 70B 路由器蒸馏成 8B，准确率 96% ([NVInfo](https://arxiv.org/abs/2510.27051))；RouteLLM 用约 1,500 条域内标注就明显改善路由 ([RouteLLM](https://arxiv.org/html/2406.18665v4)) | 没有语音流水线的飞轮案例，也没有逐阶段的归因数字 |

这张表的含义是：**我们的创新不在于发明某个新模块，而在于把已经存在的局部证据接成一个有因果依据的控制回路。** 每一条"没有做"都可以直接成为实验章节。其中最容易成文、且只有本项目有数据的两条是：通用/专家双模型之间的 LID 误路由传播，以及以 SIR 为路由信号的 SWAP 抑制。

## Audar 本质上是阿拉伯语专精的 Qwen3-ASR，所以它只能执行、不能裁决语种

文献与厂商资料给出了一个对 harness 设计至关重要的结构事实。Audar-ASR-V1-Turbo 的参数量（解码器 2.03B + 编码器 0.32B）与 Qwen3-ASR-1.7B 几乎完全一致。它随附 Qwen3-ASR 建模代码，推荐用 `qwen-asr` 推理，vLLM 也以 `Qwen3ASRForConditionalGeneration` 原生支持它。训练采用四阶段课程，最后一阶段是 KTO 偏好对齐，奖励维度包括阿拉伯语正字法和变音符 ([Audar Turbo card](https://huggingface.co/audarai/Audar-ASR-V1-Turbo))。由此推断（我们的推断，非厂商确认），Audar 是 Qwen3-ASR-1.7B 的全量或 PEFT 微调，`language X<asr_text>` 的输出协议和 tokenizer 都相同。厂商自己写明：强制语种"**是强制要求**"，`language None` 必须映射为空输出，否则音乐或静音会触发重复循环。模型卡没有公布任何英文或中文 ASR 数字，也没有说明上下文/热词功能的效果 ([Audar Turbo card](https://huggingface.co/audarai/Audar-ASR-V1-Turbo))。许可证是 AudarAI Community License，大规模或 MaaS 商用需要企业许可，这会限制权重层的 RSI。

本项目实验把这个推断变成了硬约束。在 11 个真实阿拉伯语测试集（约 26 小时，强制阿拉伯语，N1 归一化）上，Audar-Turbo 赢了 10 个：Casablanca-UAE **30.2 对 45.7**，Mixat 阿英语码转换 17.1 对 22.2，SADA 28.9 对 43.2，FLEURS-ar 6.5 对 9.7。马格里布方言对所有模型都在 60–130 WER 之间，自动 LID 模式会让 WER 升高 1–5 个点。但在中英文上，自动语种模式下 Audar 有 23% 的 FLEURS-en、63% 的 FLEURS-zh、82% 的 ASCEND 对话语句输出阿拉伯文。即使强制语种，它在对话上也比 Qwen3-ASR-1.7B 差约一倍（ASCEND zh MER 14.4 对 6.8，语码转换 19.5 对 10.0，FLEURS-en WER 6.7 对 4.8）；强制中文时仍有 20% 的语句含阿拉伯文字。相比之下，Qwen3-ASR-1.7B 的自动 LID 与强制语种几乎等价，在中英文上从不输出阿拉伯文。速度方面，在 A6000 上用 transformers 推理，Audar-Turbo 的 RTF 为 0.25，Qwen3-ASR-1.7B 为 0.19（本项目实验）。

这几组数字共同决定了 harness 的第一原则（我们的提议）：**Qwen3-ASR 负责裁决语种，Audar 只负责执行阿拉伯语转写。** 两个模型的语种后验不能直接比较，因为 Audar 的 LID 头明显偏向阿拉伯语；KTO 也改变了解码器置信度的校准，原始平均 log-prob 同样不能跨模型比较。如果需要"两个都跑、再选一个"，选择依据应是文字体系一致性、长度合理性，以及用对方模型做 teacher-forcing 的交叉打分。另有两个可直接检验的工程假设。其一，如果 Audar 的 `audio_tower.*` 权重与 Qwen3-ASR 相同，一次编码可以同时喂给两个解码器；但四阶段课程中包含预训练，编码器很可能已被更新，需要做张量差分验证。其二，两者同源意味着**错误相关**，用"Qwen3 与 Audar 一致"作为伪标签过滤器时，可靠性弱于 AssemblyAI 用两个异构模型、20% WER 一致性阈值的做法 ([Universal-1](https://arxiv.org/pdf/2404.07341v2))。因此需要第三个结构不同的弱验证器，例如 CTC 模型或强制对齐得分。最后一个工程约束：Qwen3-ForcedAligner **不支持阿拉伯语**，它只覆盖 11 种语言 ([Qwen3-ASR](https://arxiv.org/html/2601.21337))。阿拉伯语的词级时间戳和 SWAP 检测要另找对齐器（MMS 或 wav2vec2-Arabic 的 CTC 对齐），或者退回到分离片段级的时间戳。

## Harness 架构：七个模块、两道闸门、一条可回放的轨迹

以下全部是**我们的提议**。整体结构是：在一个确定性主干上挂可搜索的策略点；每个片段都留下完整轨迹，供 RSI 提议者读取。

**模块划分。**（M1）前端与分离：多通道时做 GSS 或波束形成；用 DiariZen、pyannote community-1、Audar-Diarization 三者输出 DOVER-Lap 融合的说话人段，并标出重叠区域。pyannote 的 `exclusive_speaker_diarization` 输出天然适合把重叠段显式标成"争议段" ([pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1))。（M2）语种裁决：以 Qwen3-ASR 首个 `language` token 的概率分布为主信号，以 VoxLingua107 ECAPA（约 2,000 万参数，开发集错误率 6.7%）为独立旁证 ([SpeechBrain](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa))，再用"说话人级语种先验"做平滑，即对同一说话人聚类的所有片段聚合语种后验。（M3）SIR/重叠估计：有近场通道时用近远场能量比；没有时用分离后各流的能量比或分离器的重叠后验。（M4）上下文构建：从议程、幻灯片 OCR、讲义和参会者名单建词表，按片段检索 top-k 后注入 `context`。（M5）解码执行：Qwen3-ASR（强制 zh/en/ar 或自动）、Audar（只允许强制 ar），可选 Qwen3-ASR-0.6B 作为 SWAP 探针。（M6）验证器：确定性检查与交叉模型检查。（M7）后处理：用保留原词的说话人标签迁移（TPST 或约束解码）做标签修正，只对实体片段做受约束的纠错。

**语种闸门的路由规则。** 路由阈值不应取 0.5，而应由不对称代价推出。设阿拉伯语片段送到 Audar 的收益为 G，即本项目实验中的 3–15 个 WER 点（FLEURS 3.2，Mixat 5.1，SADA 14.3，Casablanca-UAE 15.5），取约 10；设中英文片段被误送 Audar 的损失为 L，翻译式输出近乎整句错误，取约 100。则送 Audar 的盈亏平衡点是 P(ar) > L/(G+L) ≈ **0.9**（我们的估计）。P(ar) 处于中间带（例如 0.4–0.9）或片段短于约 2 秒时，同时跑两个模型，按文字体系和交叉打分选择。关于短段，文献显示 LID 在 1–2 秒以下急剧退化 ([2203.00328](https://arxiv.org/pdf/2203.00328))，Qwen3-ASR 的最小输入是 0.5 秒，更短的输入会被补零 ([qwen-asr utils](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/utils.py))。所以短段优先继承说话人级或邻近片段的语种，而不是逐段重新判断。阿拉伯语主导的阿英语码转换段送 Audar（Mixat 上 17.1 对 22.2），英语主导的送 Qwen3。Audar 输出之后还要过一道"翻译签名"检查：如果 M2 判为 zh/en，但 Audar 输出了阿拉伯文，或者 Audar 输出的阿拉伯文比例、长度与 Qwen3 假设严重不符，就丢弃 Audar 结果并回退。这条规则的先例只有语音识别里的置信度级联：端侧 → 服务端 ASR 在 0.85 阈值下，只升级 53% 的语句，就比任一单系统都好 ([2103.06716](https://arxiv.org/pdf/2103.06716))。

**SIR 闸门的重解码规则。** 本项目实验给出了因果剂量曲线：受控重混下，SWAP 增量在 −5 dB 时为 +21.1，0 dB 时 +8.8，5 dB 时 +2.1，10 dB 时 +0.6；远场重叠语句中有 28% 出现 SWAP，约 15.6 错误/100 词；1.7B 比 0.6B 更容易 SWAP；两个干扰者会使腐化和删除翻倍；近场受影响小得多；幻觉、空输出和 LID 错误合计 <0.3%。据此，控制逻辑应该是：对 M1 标出的重叠段，先由 M3 估计 SIR。SIR ≥ 5 dB 时直接解码。低于 5 dB 时，按代价依次尝试：切换近场通道；分离或目标说话人提取后重解码（文献警告前端会造成失配，USTC 在 GSS 数据上训练 ASR 后 tcpWER 从 16.57 降到 12.07 ([USTC-NERCSLIP](https://arxiv.org/html/2409.02041))，未经适配直接用会吃掉部分收益）；英文段可选 DiCoW 系目标条件化模型。同时活跃三人及以上的片段，标记为低可信度，而不是强行转写。还需要一个廉价的 SWAP 探针（我们的提议）：用 Qwen3-ForcedAligner（en/zh）给假设中的词定位，计算每个词跨度与目标说话人聚类和干扰者聚类的嵌入相似度，偏向干扰者的词标记为疑似 SWAP。1.7B 和 0.6B 在重叠段上的大幅分歧可作为第二个信号，因为两者的失败模式不同（前者 SWAP，后者腐化）。这个探针本身必须先在 Study C 重混数据上校准，那里每个词属于谁是已知的。重叠混合音频上的说话人嵌入噪声很大，探针的精度是一个未知数。

**词表注入与验证器。** Qwen3-ASR 的 SFT 包含上下文偏置数据，但论文**没有报告其效果的任何数字** ([Qwen3-ASR](https://arxiv.org/html/2601.21337))。外部对比发现，speech LLM（含 Qwen3-ASR-1.7B）在非朗读语音上泛化较差，并且对干扰词数量和**词在 prompt 中的顺序**敏感 ([Huber & Waibel](https://arxiv.org/abs/2608.05759))。因此 k、排序和模板都应作为搜索维度，大词表要先检索再注入 ([BR-ASR](https://arxiv.org/pdf/2505.19179))。Audar 的上下文效果完全未知，需要单独测量。验证器分三级。第一级是确定性检查：文字体系与路由语种一致；每秒字符数在合理范围；重复 n-gram；`language None` 映射为空。第二级是交叉模型一致性，只在中间带触发。第三级是受约束的 LLM 评审，只看实体片段，并且必须带逐词置信度。朴素纠错会把构音障碍语音的 WER 从 10.83% 推到 20.01%，加入置信度后回落到 10.58% ([Hernandez et al.](https://arxiv.org/html/2509.25048))。CHiME 综述也指出，LLM 后处理在真实远场会议上只有边际收益 ([CHiME DASR review](https://arxiv.org/html/2507.18161v2))，HINTT 里 GEC 只贡献约 0.1–0.2。

**轨迹格式。** Meta-Harness 的消融显示，只给分数时中位数 34.6，给完整原始轨迹时 50.0 ([Meta-Harness](https://arxiv.org/html/2603.28052v1))。所以轨迹必须是逐段、可回放、含所有中间决策的 JSONL。下面是我们建议的格式，每行一个片段：

```json
{"run_id":"h-0042","session":"mtg_017","seg_id":"mtg_017_s0193","t":[812.40,816.95],
 "diar":{"spk":"S3","conf":0.81,"overlap_ratio":0.42,"n_active":2,"systems":{"diarizen":"S3","pyannote":"S3","audar_diar":"S1"}},
 "lid":{"qwen_probs":{"ar":0.12,"en":0.85,"zh":0.03},"ecapa":"en","spk_prior":{"en":0.93},"final":"en","band":"confident"},
 "sir":{"est_db":1.8,"method":"css_energy","gate":"redecode"},
 "context":{"source":["agenda","slides_p12"],"terms":["NOTSOFAR","tcpWER"],"k":12,"order":"retrieval_score"},
 "decodes":[{"model":"qwen3-1.7b","lang":"en","input":"farfield","hyp":"...","lp":-0.31,"rtf":0.18},
            {"model":"qwen3-1.7b","lang":"en","input":"css_stream_1","hyp":"...","lp":-0.22,"rtf":0.19}],
 "verify":{"script_ok":true,"cps":13.1,"rep":false,"swap_probe":{"flag_words":3,"sim_margin":-0.07}},
 "select":{"chosen":1,"reason":"sir_redecode+swap_probe"},
 "post":{"edits":[],"tpst":true},
 "cost":{"gpu_s":1.7},
 "ref_metrics":{"wer":0.18,"swap_per100":0.0,"del":0.04}}
```

`ref_metrics` 字段只出现在搜索集的轨迹里。选择集和测试集的轨迹对提议者只返回聚合分数。所有模型解码都按 (片段, 模型, 语种, 输入流, context 哈希) 缓存，这是后文控制 RSI 成本的关键。

## RSI 设计：四个自改进层级共用一个锁死的评测器

文献给了三条可以直接落地的原则。其一，**先局部、后联合**：MASS 的分阶段消融为基线 63.54 → 块级优化 74.56 → 拓扑 77.55 → 工作流级 78.40，大部分收益来自局部 ([MASS](https://arxiv.org/html/2502.02533v2))。其二，**局部奖励必须与全局指标对齐**：Optimas 的局部奖励函数成对排序准确率约 70–73%，明显优于带 20 个上下文示例的 LLM 评审（77.96% 对 49.52%）([Optimas](https://arxiv.org/html/2507.03041))。其三，**依赖感知的信用分配**有实际价值：去掉 ADOPT 的依赖估计后，HotPotQA 从 0.69 跌到 0.64 ([ADOPT](https://arxiv.org/html/2512.24933v2))。语音流水线比一般 LLM 流水线多一个优势：**每个模块都有天然的局部指标**（DER、LID 准确率、分路 WER），只需测量它们与 cpWER/tcpWER 的对齐度。下表是我们提议的分层设计。

| 层级 | 被自修改的对象 | 优化器 | 局部适应度 | 晋升门 |
|---|---|---|---|---|
| L0 阈值 / 配置 | P(ar) 盈亏阈值与中间带、短段时长阈值、说话人先验权重、SIR 门限、分离的 clustering_threshold / min_duration_off、切分长度、context k | Optuna TPE / NSGA-II（多目标：tcpWER 对 GPU 时间）；可用 LLM 做初值（LLAMBO 式）但不进内循环 ([LLAMBO](https://arxiv.org/pdf/2402.03921)) | 分路 WER、SMR、LID 分桶错误率 | 选择集上各层不退化 |
| L1 代码 / 路由 | 路由程序、回退顺序、SWAP 探针逻辑、重解码策略、验证器组合 | Meta-Harness 式编码 agent，读文件系统中的历史候选源码与原始轨迹 | 全局向量适应度 + oracle 阶梯归因 | 严格改进 + 跨语种闸门 + diff 审计 |
| L2 上下文 / prompt | 词表检索、排序、模板；按领域的规则记忆 | GEPA 式反思进化；Voice Memory 式有界增删改 ([GEPA](https://arxiv.org/abs/2507.19457)) | B-WER、U-WER、有害编辑率 | 每个领域都不退化 |
| L3 权重 | 学习式路由器；分离器；Qwen3 目标说话人适配；阿拉伯语 LoRA | RouteLLM 式成对偏好训练；DOVER-Lap 加词汇修正的分离伪标签；GRPO（WER + SWAP/删除惩罚）；回放式 LoRA | 路由准确率、DER/核心错误、SWAP、领域 WER | 冻结回归集 + 回放 + 伪标签占比上限 |

**适应度向量与信用分配。** 主指标按语种分开：英文用 MeetEval 的 tcpWER/cpWER，中文用 CER 版本，阿拉伯语用 N1 归一化 WER 加 CER ([MeetEval](https://arxiv.org/pdf/2307.11394))。辅助指标包括：ORC-WER 与 Δcp（cpWER 减去不计说话人的 WER，用来单独衡量归属错误）([SEAL](https://arxiv.org/html/2501.08421v1))；DER 与说话人数误差；按 <1 s / 1–2 s / 2–5 s / >5 s 分桶的 LID 错误率；**输出文字不匹配率 SMR**（输出文字体系与参考语种不符的语句比例，文献中没有标准指标，需要我们定义）；SWAP/100 词、重叠词错误率、删除率；B-WER/U-WER；有害编辑率；RTF 和每小时音频的 GPU 秒数。阿拉伯语码转换段另报 PolyWER 这类允许音译的指标，因为 WER 会把语码转换系统之间的差距夸大约 3 倍 ([2605.19069](https://arxiv.org/abs/2605.19069), [PolyWER](https://aclanthology.org/2024.findings-emnlp.356.pdf))。信用分配的主工具是 **oracle 替换阶梯**：在带标注的开发集上，依次把 LID、分离、路由（两个 ASR 中取最优）、输入（近场或 oracle 分离）换成真值，相邻两级的差值就是该模块的余量。HINTT 恰恰缺这一步。Optimas 式的局部奖励只有在"局部指标变化与全局变化的秩相关"被测量并足够高时才使用；TS-SEP 已经证明 DER 和 WER 可能反向。预算分配按余量进行，即 ADOPT 的 Shapley 思路。

**划分与防过拟合、防作弊。** 搜索集、选择集、测试集按会话和说话人互斥。搜索集可以包含 Study C 风格的合成重混，选择集和测试集只用真实录音。阿拉伯语方面，11 个测试集中要固定留出整套数据集（例如 Casablanca-UAE 与 Mixat）作为域外测试，不能只按语句抽样。L1 修改还要通过**跨语种闸门**：在英文重叠数据上找到的 harness，必须在阿拉伯语和中文选择集上不退化，反之亦然。EvoPrompt 在 CHiME-4 上优化出的纠错 prompt 把 WSJ 从 3.67 恶化到 14.75 ([EvoPrompt for GER](https://arxiv.org/html/2407.16370v1))；HarnessCompass 的自由形式进化在样本集上涨 9 点，在留出集上只涨 3.1 点 ([HarnessCompass](https://arxiv.org/html/2608.01918v1))。归一化器、MeetEval、对齐器、SWAP 统计脚本和 SMR 计算放在只读目录，用校验和锁定。DGM 曾通过删除幻觉检测标记来"完成"任务 ([Sakana DGM](https://sakana.ai/dgm/))。评测器如果要演进（例如新的 SWAP 探针），每次晋升都必须在人工标注的锚点集上验证，即 RQGM 的规则 ([RQGM](https://arxiv.org/abs/2606.26294))。权重层要遵守三条：伪标签必须来自异构视角（委员会融合、跨模态、词汇修正），单教师伪标签在 pyannote 微调中没有带来增益 ([ShobdoSetu](https://arxiv.org/pdf/2603.19256))；每轮都回放原始数据，LoRA 加回放的组合减少了 55% 的遗忘 ([2512.16401](https://arxiv.org/html/2512.16401v4))；伪标签占比要设上限。还要监控输出分布漂移：假设长度、重复率、文字体系比例、说话人数分布。

**成本估算（我们的估计）。** 关键做法是"先缓存、后搜索"。对约 26 小时阿拉伯语数据，用 Qwen3 和 Audar 各做一次强制阿拉伯语解码，按本项目实验的 RTF 计算约需 26 × (0.25 + 0.19) ≈ **11.4 GPU 小时**（A6000，transformers）；换用 vLLM 会明显更少。缓存之后，L0 的阈值搜索只是在 CPU 上重放选择逻辑，数百次试验几乎没有成本。L1 每个候选只需对新引入的解码路径补跑解码。按 Meta-Harness 的规模（约 60 个候选、20 轮）粗估，在 JUBAIL 上用作业数组并行是可行的，量级为数百 GPU 小时。按项目约定，所有计算都通过 Slurm 提交，不在登录节点运行；提议者 agent 的主循环放在一个长时 CPU Slurm 作业里，只负责编辑候选代码、提交 GPU 评测作业和读取轨迹。NYUAIRLAB-127（A6000）适合做交互式调试和缓存预热。

## 路线图：先把误路由和 SWAP 变成可测量的变量，再交给搜索

下表中的"预期收益"全部是**我们的估计**，依据写在最后一列，并非测量结果。

| 优先级 | 内容与实验 | 预期收益（估计） | 依据 |
|---|---|---|---|
| P0 | 评测与轨迹底座：锁死的归一化器（en/zh/ar）、MeetEval、SMR、SWAP 统计、oracle 阶梯；双模型 × 多语种的解码缓存 | 无直接收益，但决定后续数字是否可信 | Meta-Harness 原始轨迹消融；DGM 作弊案例 |
| P1a | LID 安全路由器（规则版）：Qwen3 裁决、P(ar) 约 0.9 的不对称阈值、中间带双跑、说话人先验、翻译签名回退。实验 E1：在 11 个阿拉伯语集、FLEURS-en/zh、ASCEND，以及人工拼接的 ar/en/zh 多语会话上扫阈值 | SMR 从"Audar 自动语种"的 23–82% 降到 <1%；阿拉伯语段相对"全用 Qwen3"降低约 20–30% 相对 WER（oracle 路由的上限约 23–34%，LID 错误和中间带会吃掉一部分） | 本项目实验的双模型对比；Qwen3 对话 LID 94.1% |
| P1b | SIR 感知重解码 + SWAP 探针。实验 E2：先在 Study C 重混数据上校准探针（已知每个词的归属）；E3：在 LibriCSS 和 Study A 真实重叠数据上做 oracle 阶梯，测"若重解码路径完美"的上限 | 重叠语句的 SWAP 减少 30–60%；LibriCSS OV40 的 WER 从 19.8% 降到约 13–16% | 本项目实验：5 dB 时 SWAP 从 +8.8 降到 +2.1，SWAP 占 OV40 错误的 64%；GSS 失配消融 |
| P1c | 领域词表检索注入（Qwen3；Audar 先测 E4：context 对阿拉伯语术语是否生效） | 术语 B-WER 相对降低 20–40%；整体 WER 相对降低 1–5% | BR-ASR；Huber & Waibel 顺序敏感；幻灯片带来 34–35% ([Slides](https://arxiv.org/abs/2510.13979)) |
| P2a | L0 Optuna 联合搜索（含分离阈值，以 tcpWER 为目标而非 DER）。实验 E5：复现 TS-SEP 式的"DER 最优 ≠ WER 最优" | 在 P1 基础上相对再降 2–5% | TS-SEP 反向现象；HINTT 未做阈值消融，留有余量 |
| P2b | L1 Meta-Harness 式代码搜索 + 跨语种闸门 | 选择集再降 3–8%，测试集约为其一半 | Meta-Harness、HarnessCompass 的样本/留出差距 |
| P2c | 阿拉伯语分离 A/B：Audar-Diarization 对 pyannote community-1 对 DiariZen，以及 DOVER-Lap 融合（E6，需要少量标注的阿拉伯语多人会话） | 未知；先测 | 唯一的阿拉伯语 DER 只有 12 个文件；Audar-Diar 没有阿拉伯语数据 |
| P3a | 学习式路由器：从缓存的成对结果训练小分类器替代规则（RouteLLM 式），约 1–2k 条域内标签 | 中间带的选择准确率提升，双跑比例下降，成本降低 30–50% | RouteLLM 26%→14%；NVInfo 8B 路由器 96% |
| P3b | 分离器伪标签闭环：DOVER-Lap + DiarizationLM 式词汇修正 → 帧级 RTTM → 微调分割模型，至多 2 轮 | DER 相对降低 10–30%，但必须同时看核心错误 | 委员会伪标签 −37.4% DER ([Takashima](https://arxiv.org/pdf/2106.04764))；单教师无增益 |
| P3c | Qwen3 目标说话人微调（[参考音频][静音][混合] 格式，用 Study C 生成器在 −5 至 +10 dB 构造，加入目标缺席负样本），奖励含 SWAP 和删除惩罚 | 远场重叠 SWAP 大幅下降，但难以复现 CocktailASR 的约 16 个点 | CocktailASR 用约 100 万小时数据、8B 规模 |

先做 P1a 和 P1b 而不是直接上 RSI，理由来自数据本身。本项目实验已经给出了两条有因果依据的规则：误路由几乎造成整句损失，5 dB 是 SWAP 的阈值。不经搜索就能把它们写成代码并验证。RSI 的价值在于自动发现阈值、回退顺序，以及不同语种、不同场地之间可泛化的组合。另外，马格里布方言 60–130 WER 的问题，任何 harness 都解决不了：Talafha 等的上下文解码在方言上也只带来 9.15% 的相对降幅 ([Talafha et al.](https://arxiv.org/abs/2511.18774v2))。那属于权重层和数据问题，而且受 Audar 许可证约束，应单独立项。

## 风险：最可能的失败是"指标变好、转写没变好"

奖励作弊是首要风险，在这条流水线上有五种具体形态（我们的分析）。第一，**删除换 SWAP**：两个干扰者时模型本来就倾向删除，提议者可能学会对重叠段输出空串，SWAP 下降而信息丢失。删除率必须作为 Pareto 约束。第二，**路由到"不出错"的空输出**：把可疑片段判为 `language None` 也能降低 SMR，因此 SMR 必须与删除率、覆盖率一起看。第三，**正字法套利与 MSA 化**：Audar 的 KTO 目标本身就奖励特定正字法，Mixat 上的差距被厂商归因于拉丁/阿拉伯文字的书写选择 ([Audar Turbo card](https://huggingface.co/audarai/Audar-ASR-V1-Turbo))；可修改的归一化器或 LLM 纠错可以通过统一拼写制造虚假"收益"。第四，**词表泄漏**：词表只能来自议程、幻灯片等独立来源，并对 diff 做审计，拦截硬编码的语句 ID。第五，**相关验证器**：Qwen3 与 Audar 同源，它们的一致性会把共同错误认证为真值。跨模型共识优于自一致性，前提是模型之间去相关 ([LLMs as a Jury](https://arxiv.org/pdf/2607.10139))。

其余风险分属技术和外部条件。置信度在噪声下"具有欺骗性地高" ([2509.07195](https://arxiv.org/html/2509.07195v1))，所以远场条件下不能用 token 熵门控数据或驱动 TTA；文献中 SUTA 曾把 WER 从 83.61 推到 122.37 ([ASR-TRA](https://arxiv.org/html/2603.05231))。SWAP 探针依赖重叠混合音频上的说话人嵌入，精度未经验证；阿拉伯语缺少 Qwen3 对齐器，探针在阿拉伯语上会更弱。阿拉伯语多人远场会议没有公开语料，Study C 式重混只是近似，真实场景的混响、交叠模式和语码转换节奏都可能不同，所以至少需要少量自采标注会话作为锚点。Audar 的模型卡数字在不同版本之间漂移（平均 WER 24.78 与 23.17），Casablanca 的局限说明（约 63%）与表格（47.02）自相矛盾；它的 Community License 也可能禁止部分权重层改造或商用部署。最后是收益上限：所有自改进系统都会在几十轮内饱和，权重层的伪标签收益集中在前 2–3 轮 ([ReST-EM](https://arxiv.org/abs/2312.06585))。路线图里的数字应被理解为一次性台阶，而不是可以持续复利的曲线。

## 结论

这项研究的核心发现是：在"通用模型 + 专家模型"的语音流水线中，harness 的首要职能不是提升平均准确率，而是**管理不对称的灾难性失败**。Audar 的翻译式输出和远场 SWAP 有一个共同点：模型输出的是流畅、合法、高置信度的文本，任何纯文本层的验证器都看不出问题。只有能接触到声学证据（语种后验、SIR、说话人嵌入）的控制器才能拦截它们。这也解释了为什么现有的单路 ASR 自改进工作（Voice Memory、GER prompt 进化）在这里不够用，它们优化的是"已经说对了谁、说对了哪种语言"之后的文本。

由此产生一个可以投稿的研究框架："带声学闸门的自改进路由 harness"。它的贡献可以拆成三个独立可测的命题：一是 LID 误路由在通用/专家 ASR 对之间的传播代价，以及由此推出的不对称阈值；二是 SIR 作为路由变量的因果有效性，以及 SWAP 探针的准确度；三是在锁死的向量评测器下，阈值、代码、上下文三层搜索的样本/留出差距。三个命题都可以用本项目已有的 11 个阿拉伯语集、FLEURS/ASCEND 和三个重叠研究（A/B/C）直接起步，几乎不需要新数据。只有阿拉伯语多人会话这一块，必须靠一小批自采标注来补上。
