# 2025/11/5
## 实验思路
实现一个 “双向协同机制”：
+ LLM → KG： 利用大语言模型的理解与抽取能力，从非结构化文本、报告、试验日志、运行记录等数据中自动构建知识图谱（结构化知识）；
+ KG → LLM： 将结构化知识反向注入LLM，在生成、推理、问答、决策等任务中提供知识约束与事实增强（减少幻觉）
## 论文推荐
- Knowledge Graph Prompting for Multi-Document Question Answering
- Knowledge Graph Enhanced Large Language Model Editing
- A Framework of Knowledge Graph-Enhanced Large Language Model Based on Question Decomposition and Atomic Retrieval
- Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph
- Construction of a knowledge graph for framework material enabled by large language models and its application

