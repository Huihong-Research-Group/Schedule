# 前期所读论文有:
FinDKG: Dynamic Knowledge Graphs with Large Language Models for Detecting Global Trends in Financial Markets：新闻提取模型ickg

GOLLIE:ANNOTATION GUIDELINES IMPROVE ZERO-SHOT INFORMATION-EXTRACTION 介绍了gollie方法，提示词工程，给与AI一本完整的标注手册，先培养其为知识图谱构建专家。直接用开源模型并复制其提示词与方法效果并不好，因为它在使用手册训练时需要对模型本身进行微调

REBEL: Relation Extraction By End-to-end Language generation一种专门用于知识图谱构建的模型，但是年代较远2021，是基于2020年的小参数模型，直接使用不如现在的模型

FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation。FewRel数据集一种专门用来小样本进行关系分类的数据集，其Git库中有过去的知识图谱构建老方法protonet、pairwise等

ChatIE: Zero-Shot Information Extraction via Chatting with ChatGPT通过多步问答分解引导llm完成知识图谱构建问题

在FewRel数据集上展开的实验结果如下

<img width="582" height="785" alt="螢幕擷取畫面 2026-03-19 202934" src="https://github.com/user-attachments/assets/17005e37-f296-439a-b923-b0592c590970" />

https://docs.qq.com/sheet/DWkJjZ2Nwa09jaUFq?tab=000001

在开放域

2.1  CoNLL04（5 种关系类型）

模型	方法	Precision	Recall	F1

Qwen2.5-7B	Single	25.00%	11.72%	15.96%

Qwen2.5-7B	ChatIE	33.33%	36.55%	34.87%

Qwen2.5-7B	GoLLIE	50.00%	3.45%	6.45%

LLaMA-3.1-8B	Single	33.53%	39.31%	36.19%

LLaMA-3.1-8B	ChatIE	28.25%	52.41%	36.71%

Qwen2.5-14B-GPTQ	Single	22.22%	4.14%	6.98%

Qwen2.5-14B-GPTQ	ChatIE	46.77%	40.00%	43.12%

* Qwen2.5-14B-GPTQ + ChatIE 以 F1 43.12% 最优。GoLLIE 在 NYT 路径格式关系名上得分为 0。

2.2  NYT-multi（24 种关系类型）

模型	方法	Precision	Recall	F1

Qwen2.5-7B	Single	21.15%	6.55%	10.00%

Qwen2.5-7B	ChatIE	30.89%	22.62%	26.12%

Qwen2.5-7B	GoLLIE	0.00%	0.00%	0.00%

LLaMA-3.1-8B	Single	29.67%	16.07%	20.85%

LLaMA-3.1-8B	ChatIE	24.69%	23.81%	24.24%

Qwen2.5-14B-GPTQ	Single	18.18%	1.19%	2.23%

Qwen2.5-14B-GPTQ	ChatIE	41.30%	11.31%	17.76%

* NYT 关系名为路径格式（/people/person/nationality），难度显著高于 CoNLL04。LLaMA-3.1-8B + ChatIE 以 F1 24.24% 最优。

* 观察发现在一个句子中LLM会提取多个实体并从中判断出多对三元组，而给定评估的答案则没有这么多候选项，因此准确率大幅下降，之后可能要重新写脚本

# 关于模型增强

## 目前思路是进行一个三阶段 Pipeline：Schema Induction → Triple Extraction → Supervisor Refinement。
即：先给提示词让提取模型分析以下文本并定义一个提取模式（Schema）。 步骤： 1. 识别文本中的核心实体类型（如：人物、机构、技术、概念）。 2. 定义这些实体之间可能存在的逻辑关系。 3. 以 JSON 格式输出模式定义。再进行三元组提取。提取完后的结果让一个监督模型（可以与提取模型相同或不同）进行监督剔除逻辑矛盾的冗余三元组，目标是三元组不重复不遗漏，最后让提取模型综合原始输出与监督模型的意见完成最终输出

文献支持有：LLMs for Knowledge Graph Construction and Reasoning: Recent Capabilities and Future Opportunities，ChatIE: Zero-Shot Information Extraction via Chatting with ChatGPT证明了思维连引导对于模型的作用；
Leveraging LLMs Few-shot Learning to Improve
Instruction-driven Knowledge Graph Construction，SELF-REFINE:
Iterative Refinement with Self-Feedback论文证明了self-reflection，self-refinement，说明了这种监督审查时有效的

在看文献时知道许多模型输出因为格式谬误而导致输出准确率不高，因此schema设定中特意要求以 JSON 格式输出模式定义。
