"""
Reflexion-style Self-Supervised KG Construction Pipeline
参考：Shinn et al. "Reflexion: Language Agents with Verbal Reinforcement Learning" NeurIPS 2023
      arXiv:2303.11366

Reflexion 核心机制对应：
  Actor          → ChatIE 两阶段提取模型
  Evaluator      → 同一模型自我评估三元组质量（内部模拟反馈）
  Self-Reflector → 模型分析自身错误，生成语言反思
  Memory Buffer  → 跨句子积累反思经验，作为后续抽取的上下文

与原始 Reflexion 的区别：
  - 原版 Reflexion 依赖外部环境反馈（如代码执行结果）
  - 本方法使用"内部模拟反馈"（internally simulated feedback）
    即模型自己评估自己的输出质量，无需外部监督信号
  - 这符合 Reflexion 论文中提到的
    "sources: external or internally simulated"的设计

用法：
    python chatIE_reflexion.py --dataset conll04 --model qwen7b
    python chatIE_reflexion.py --dataset conll04 --model qwen14b
    python chatIE_reflexion.py --dataset nyt     --model qwen7b
    python chatIE_reflexion.py --dataset nyt     --model qwen14b
    python chatIE_reflexion.py --dataset conll04 --model qwen7b --max_trials 3
"""

import os
import json
import argparse
import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ───────────────────────────────────────────────
# 配置
# ───────────────────────────────────────────────
MODEL_PATHS = {
    'qwen7b':     '/root/protonet/models/Qwen/Qwen/Qwen2___5-7B-Instruct',
    'qwen14b':    '/root/protonet/models/Qwen/Qwen/Qwen2___5-14B-Instruct-GPTQ-Int4',
    'llama8b':    '/root/protonet/models/LLaMA/LLM-Research/Meta-Llama-3___1-8B-Instruct',
    'deepseek7b': '/root/protonet/models/DeepSeek/DeepSeek-R1-Distill-Qwen-7B',
}
DEEPSEEK_MODELS = {'deepseek7b'}

N_TEST         = 50
SEED           = 42
MAX_NEW_TOKENS = 512
MAX_TRIALS     = 2    # Reflexion 最大尝试轮数（含第一次）

CONLL04_RELATION_TYPES = {
    'Located_In':   'A location entity is situated within another location entity.',
    'Work_For':     'A person works for or is employed by an organization.',
    'OrgBased_In':  'An organization is based in or headquartered in a location.',
    'Live_In':      'A person lives in or resides in a location.',
    'Kill':         'A person kills another person.',
}

NYT_RELATION_TYPES = {
    '/location/location/contains':            'A location contains another location.',
    '/people/person/nationality':             'A person has a nationality.',
    '/people/person/place_lived':             'A person lives in a place.',
    '/business/person/company':              'A person works at a company.',
    '/location/country/capital':             'A country has a capital city.',
    '/people/person/place_of_birth':         'A person was born in a place.',
    '/people/person/children':               'A person has children.',
    '/location/us_state/capital':            'A US state has a capital.',
    '/business/company/founders':            'A company was founded by a person.',
    '/people/deceased_person/place_of_death':'A person died in a place.',
    '/sports/sports_team/location':          'A sports team is located in a place.',
    '/people/person/ethnicity':              'A person belongs to an ethnicity.',
    '/business/company/place_founded':       'A company was founded in a place.',
    '/location/neighborhood/neighborhood_of':'A neighborhood is part of a location.',
    '/location/country/administrative_divisions': 'A country has administrative divisions.',
    '/people/ethnicity/geographic_distribution':  'An ethnicity is distributed in a location.',
    '/sports/sports_team_location/teams':    'A location has sports teams.',
    '/people/person/religion':               'A person follows a religion.',
    '/business/company/industry':            'A company belongs to an industry.',
    '/film/film/featured_film_locations':    'A film was shot in a location.',
    '/location/administrative_division/country': 'An administrative division belongs to a country.',
    '/sports/sports_league/teams':           'A league has teams.',
    '/people/person/profession':             'A person has a profession.',
    '/business/company/advisors':            'A company has advisors.',
}

# ───────────────────────────────────────────────
# 数据加载
# ───────────────────────────────────────────────
def load_conll04(n=None):
    ds = load_dataset('DFKI-SLT/conll04')
    samples = []
    for item in ds['test']:
        tokens    = item['tokens']
        sentence  = ' '.join(tokens)
        ent_texts = [{'text': ' '.join(tokens[e['start']:e['end']]),
                      'type': e['type']} for e in item['entities']]
        golds = [{'head': ent_texts[r['head']]['text'].lower(),
                  'relation': r['type'],
                  'tail': ent_texts[r['tail']]['text'].lower()}
                 for r in item['relations']]
        samples.append({'sentence': sentence, 'entities': ent_texts,
                        'gold_triples': golds})
    return samples[:n] if n else samples

def load_nyt(n=None):
    ds = load_dataset('DFKI-SLT/nyt-multi')
    samples = []
    for item in ds['test']:
        tokens   = item['tokens']
        sentence = ' '.join(tokens)
        ent_set  = {}
        for r in item['relations']:
            for side in ('h', 't'):
                key = r[side]['text']
                if key not in ent_set:
                    ent_set[key] = {'text': key, 'type': r[side]['type']}
        ent_texts = list(ent_set.values())
        golds = [{'head': r['h']['text'].lower(), 'relation': r['type'],
                  'tail': r['t']['text'].lower()} for r in item['relations']]
        if golds:
            samples.append({'sentence': sentence, 'entities': ent_texts,
                            'gold_triples': golds})
    return samples[:n] if n else samples

# ───────────────────────────────────────────────
# 模型推理
# ───────────────────────────────────────────────
def load_model(name):
    path = MODEL_PATHS[name]
    print(f"  加载 {name}...")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16,
        device_map='auto', trust_remote_code=True)
    mdl.eval()
    print(f"  {name} 加载完成，显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    return mdl, tok

def chat(model, tokenizer, messages, model_name=''):
    is_deepseek = model_name in DEEPSEEK_MODELS
    if is_deepseek:
        merged = ' '.join(m['content'] for m in messages if m['role'] != 'assistant')
        messages_to_use = [{'role': 'user', 'content': merged}]
        gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS,
                          do_sample=True, temperature=0.6, top_p=0.95)
    else:
        messages_to_use = messages
        gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

    text = tokenizer.apply_chat_template(
        messages_to_use, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors='pt')
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen_kwargs['pad_token_id'] = (tokenizer.eos_token_id
                                  if tokenizer.eos_token_id else tokenizer.pad_token_id)
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    new = out[0][inputs['input_ids'].shape[1]:]
    result = tokenizer.decode(new, skip_special_tokens=True).strip()
    if '<think>' in result and '</think>' in result:
        result = result.split('</think>')[-1].strip()
    return result

# ───────────────────────────────────────────────
# 三元组解析
# ───────────────────────────────────────────────
def parse_triples(output, relation_types):
    triples, seen = [], set()
    for h, r, t in re.findall(r'\(([^,()]+),\s*([^,()]+),\s*([^,()]+)\)', output):
        h, r, t = h.strip().lower(), r.strip(), t.strip().lower()
        matched = fuzzy_match_relation(r, relation_types)
        if matched and h and t and h != 'none' and t != 'none':
            key = (h, matched, t)
            if key not in seen:
                seen.add(key)
                triples.append({'head': h, 'relation': matched, 'tail': t})
    if not triples:
        for h, r, t in re.findall(
                r'([^\|]+)\s*\|\s*([^\|]+)\s*\|\s*([^\|\n]+)', output):
            h, r, t = h.strip().lower(), r.strip(), t.strip().lower()
            matched = fuzzy_match_relation(r, relation_types)
            if matched and h and t:
                key = (h, matched, t)
                if key not in seen:
                    seen.add(key)
                    triples.append({'head': h, 'relation': matched, 'tail': t})
    return triples

def fuzzy_match_relation(text, relation_types):
    text_lower = text.lower().replace(' ', '_').replace('-', '_')
    for rel in relation_types:
        if rel.lower() == text_lower:
            return rel
    for rel in relation_types:
        rel_short = rel.lower().split('/')[-1]
        if rel_short in text_lower or text_lower in rel_short:
            return rel
    keywords = {
        'Located_In':  ['located', 'location', 'situated', 'in'],
        'Work_For':    ['work', 'employ', 'staff', 'member'],
        'OrgBased_In': ['based', 'headquarter', 'org'],
        'Live_In':     ['live', 'reside', 'home'],
        'Kill':        ['kill', 'murder', 'slay'],
    }
    for rel, kws in keywords.items():
        if rel in relation_types and any(kw in text_lower for kw in kws):
            return rel
    return None

# ───────────────────────────────────────────────
# Actor：ChatIE 两阶段提取
# 对应 Reflexion 中的 Actor 模块
# ───────────────────────────────────────────────
def actor_extract(model, tokenizer, sentence, entities, relation_types,
                  model_name='', memory_buffer=''):
    """
    Actor 模块：执行 ChatIE 两阶段提取
    当有 memory_buffer（反思记忆）时，将其作为额外上下文注入
    对应 Reflexion 的 ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION
    """
    rel_list    = '\n'.join([f"  - {r}: {d}" for r, d in relation_types.items()])
    entity_list = ', '.join([f"{e['text']} ({e['type']})" for e in entities])

    # 如果有反思记忆，在 system prompt 中注入
    # 对应 Reflexion 的 episodic memory buffer
    system_content = "You are a knowledge graph expert."
    if memory_buffer:
        system_content += (
            "\n\nLESSONS FROM PREVIOUS ATTEMPTS (Reflexion Memory):\n"
            + memory_buffer
            + "\nApply these lessons to improve your extraction."
        )

    # Stage 1: 识别关系类型
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content":
            f"Given this sentence, which of the following relation types exist "
            f"between the entities?\n\n"
            f"Sentence: {sentence}\n"
            f"Entities: {entity_list}\n\n"
            f"Relation types:\n{rel_list}\n\n"
            f"List only the relation types that exist (one per line). "
            f"If none exist, say 'None'."}
    ]
    stage1 = chat(model, tokenizer, messages, model_name)

    # Stage 2: 提取实体对
    messages.append({"role": "assistant", "content": stage1})
    messages.append({"role": "user", "content":
        f"Based on the relations you identified above, extract the specific entity pairs.\n"
        f"IMPORTANT: Only extract triples for the EXACT relation types you listed above.\n"
        f"Do NOT introduce new relation types not mentioned in your previous answer.\n\n"
        f"Use the SHORTEST form of entity names "
        f"(e.g., 'Jerusalem' not 'Arab east Jerusalem').\n\n"
        f"Output each triple in format: (head_entity, Relation_Type, tail_entity)\n"
        f"Use only entity names that appear in the sentence."
    })
    stage2 = chat(model, tokenizer, messages, model_name)

    triples = parse_triples(stage2, relation_types)
    # 纯代码去重和合法性检查
    approved = set(relation_types.keys())
    seen, result = set(), []
    for t in triples:
        key = (t['head'], t['relation'], t['tail'])
        if t['relation'] in approved and key not in seen:
            seen.add(key)
            result.append(t)

    return stage1, stage2, result

# ───────────────────────────────────────────────
# Evaluator：自我评估三元组质量
# 对应 Reflexion 中的 Evaluator 模块
# 使用"内部模拟反馈"（internally simulated feedback）
# ───────────────────────────────────────────────
def evaluator(model, tokenizer, sentence, triples, relation_types, model_name=''):
    """
    Evaluator 模块：模型自我评估当前提取结果的质量
    输出质量分数和问题描述，作为 Self-Reflector 的输入
    对应 Reflexion 的 Evaluator，使用内部模拟反馈而非外部信号
    """
    if not triples:
        return 0.0, {'summary': 'No triples were extracted. The sentence likely contains relations that were missed.', 'missing': ['All relations were missed'], 'wrong': [], 'entity_issues': []}

    triples_str = '\n'.join(
        [f"  {i+1}. ({t['head']}, {t['relation']}, {t['tail']})"
         for i, t in enumerate(triples)])
    rel_defs = '\n'.join([f"  - {r}: {d}" for r, d in relation_types.items()])

    eval_prompt = (
        "You are evaluating the quality of knowledge graph triples extracted from a sentence.\n\n"
        "Text: " + sentence + "\n\n"
        "Relation definitions:\n" + rel_defs + "\n\n"
        "Extracted triples:\n" + triples_str + "\n\n"
        "Evaluate the extraction quality by answering:\n\n"
        "Q1 - Completeness: Are there entity pairs in the text that have a valid relation "
        "but were NOT extracted? (missing triples)\n"
        "Q2 - Correctness: Are any extracted triples factually wrong based on the text? "
        "(wrong relation type or wrong entity)\n"
        "Q3 - Entity precision: Are entity names too long or contain unnecessary modifiers? "
        "(e.g., 'arab east jerusalem' should be 'jerusalem')\n\n"
        "Based on your evaluation, give an overall quality score from 0 to 10, "
        "where 10 means perfect extraction.\n\n"
        "Output ONLY in this JSON format:\n"
        '{"score": 7, "missing": ["description of missing triples"], '
        '"wrong": ["description of wrong triples"], '
        '"entity_issues": ["entity normalization suggestions"], '
        '"summary": "one-line summary of main issues"}'
    )

    messages = [
        {"role": "system", "content":
            "You are a strict knowledge graph quality evaluator. "
            "Be critical and identify specific issues."},
        {"role": "user", "content": eval_prompt}
    ]
    output = chat(model, tokenizer, messages, model_name)

    try:
        m = re.search(r'\{.*\}', output, re.DOTALL)
        if m:
            result = json.loads(m.group())
            score = float(result.get('score', 5)) / 10.0
            summary = result.get('summary', '')
            issues = {
                'missing':       result.get('missing', []),
                'wrong':         result.get('wrong', []),
                'entity_issues': result.get('entity_issues', []),
                'summary':       summary,
                'raw_score':     result.get('score', 5),
            }
            return score, issues
    except Exception:
        pass
    return 0.5, {'summary': 'evaluation parse failed', 'missing': [], 'wrong': [], 'entity_issues': []}

# ───────────────────────────────────────────────
# Self-Reflector：语言反思生成
# 对应 Reflexion 中的 Self-Reflection 模块
# ───────────────────────────────────────────────
def self_reflector(model, tokenizer, sentence, triples, eval_issues,
                   relation_types, model_name='', previous_reflection=''):
    """
    Self-Reflector 模块：基于 Evaluator 的反馈生成语言反思
    反思文本将存入 Memory Buffer，指导下一轮 Actor 的提取

    对应 Reflexion 原文：
    "self-reflective feedback acts as a semantic gradient signal by providing
    the agent with a concrete direction to improve upon"
    """
    triples_str = '\n'.join(
        [f"  ({t['head']}, {t['relation']}, {t['tail']})" for t in triples]) \
        or "  (none extracted)"

    issues_str = (
        "Missing triples: " + str(eval_issues.get('missing', [])) + "\n"
        "Wrong triples: "   + str(eval_issues.get('wrong',   [])) + "\n"
        "Entity issues: "   + str(eval_issues.get('entity_issues', [])) + "\n"
        "Summary: "         + str(eval_issues.get('summary', ''))
    )

    prev_ctx = ""
    if previous_reflection:
        prev_ctx = "\nPrevious reflection (build upon this):\n" + previous_reflection + "\n"

    reflect_prompt = (
        "You are reflecting on a failed knowledge graph extraction attempt "
        "to identify SPECIFIC, ACTIONABLE lessons for improvement.\n\n"
        "Text: " + sentence + "\n\n"
        "What you extracted:\n" + triples_str + "\n\n"
        "Quality evaluation:\n" + issues_str + "\n"
        + prev_ctx +
        "Write a SHORT reflection (3-5 sentences) that:\n"
        "1. Identifies EXACTLY what went wrong (specific entities or relations missed/wrong)\n"
        "2. States a CONCRETE rule to apply next time "
        "(e.g., 'When I see X pattern, I should extract Y relation')\n"
        "3. Notes any entity name normalization needed\n\n"
        "Be specific about THIS sentence, not generic advice.\n"
        "Write the reflection directly, no JSON, no headers:"
    )

    messages = [
        {"role": "system", "content":
            "You are reflecting on your own extraction mistakes to improve. "
            "Be specific and actionable."},
        {"role": "user", "content": reflect_prompt}
    ]
    reflection = chat(model, tokenizer, messages, model_name)
    return reflection.strip()

# ───────────────────────────────────────────────
# Reflexion 主循环
# Actor → Evaluate → Reflect → Actor（with memory）
# ───────────────────────────────────────────────
def reflexion_extract(model, tokenizer, sentence, entities, relation_types,
                      model_name='', max_trials=2, score_threshold=0.7):
    """
    完整的 Reflexion 循环：
    Trial 1: Actor 提取 → Evaluator 评分
    If score < threshold:
        Self-Reflector 生成反思 → 存入 Memory Buffer
        Trial 2: Actor 提取（携带反思记忆）→ Evaluator 评分
        ...
    返回最终三元组 + 完整的 Reflexion 轨迹
    """
    memory_buffer = ""       # episodic memory buffer
    best_triples  = []
    best_score    = 0.0
    trajectory    = []       # 记录完整轨迹

    for trial in range(max_trials):
        # ── Actor 执行 ──
        stage1, stage2, triples = actor_extract(
            model, tokenizer, sentence, entities, relation_types,
            model_name, memory_buffer)

        # ── Evaluator 评分 ──
        score, eval_issues = evaluator(
            model, tokenizer, sentence, triples, relation_types, model_name)

        trajectory.append({
            'trial':       trial + 1,
            'triples':     triples,
            'score':       round(score, 3),
            'eval_issues': eval_issues,
            'reflection':  None,
            'memory_used': memory_buffer[:150] if memory_buffer else None,
        })

        # 更新最佳结果
        if score > best_score or trial == 0:
            best_score   = score
            best_triples = triples

        # 达到阈值或最后一轮，停止
        if score >= score_threshold or trial == max_trials - 1:
            break

        # ── Self-Reflector：生成语言反思 ──
        reflection = self_reflector(
            model, tokenizer, sentence, triples, eval_issues,
            relation_types, model_name, memory_buffer)

        trajectory[-1]['reflection'] = reflection

        # ── Memory Buffer：累积反思（情景记忆）──
        memory_buffer = (memory_buffer + "\n" if memory_buffer else "") + \
                        f"[Trial {trial+1} reflection]: " + reflection

    return best_triples, best_score, trajectory

# ───────────────────────────────────────────────
# 评估
# ───────────────────────────────────────────────
def evaluate(pred_list, gold_list):
    tp = fp = fn = 0
    for preds, golds in zip(pred_list, gold_list):
        pred_set = {(t['head'], t['relation'], t['tail']) for t in preds}
        gold_set = {(t['head'], t['relation'], t['tail']) for t in golds}
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2*p*r / (p+r)  if (p+r)   > 0 else 0.0
    return {'precision': p, 'recall': r, 'f1': f1,
            'tp': tp, 'total_pred': tp+fp, 'total_gold': tp+fn}

# ───────────────────────────────────────────────
# 主函数
# ───────────────────────────────────────────────
def run(dataset_name, model_name, max_trials):
    np.random.seed(SEED)

    print(f"\n{'='*60}")
    print(f"Pipeline: Reflexion-style Self-Supervised KG Construction")
    print(f"参考: Shinn et al. NeurIPS 2023 arXiv:2303.11366")
    print(f"模型: {model_name}  数据集: {dataset_name.upper()}")
    print(f"最大尝试轮数: {max_trials}")
    print(f"{'='*60}\n")

    print("加载数据集...")
    if dataset_name == 'conll04':
        samples        = load_conll04(N_TEST)
        relation_types = CONLL04_RELATION_TYPES
    else:
        samples        = load_nyt(N_TEST)
        relation_types = NYT_RELATION_TYPES
    print(f"测试样本数: {len(samples)}\n")

    print("加载模型...")
    model, tokenizer = load_model(model_name)

    pred_list    = []
    gold_list    = []
    all_triples  = []
    examples     = []
    multi_trial_cnt = 0  # 触发反思的样本数

    for i, sample in enumerate(samples):
        sentence = sample['sentence']
        entities = sample['entities']
        golds    = sample['gold_triples']

        # Reflexion 循环
        final_triples, final_score, trajectory = reflexion_extract(
            model, tokenizer, sentence, entities, relation_types,
            model_name, max_trials=max_trials)

        if len(trajectory) > 1:
            multi_trial_cnt += 1

        pred_list.append(final_triples)
        gold_list.append(golds)
        for t in final_triples:
            all_triples.append({**t, 'source': sentence[:80]})

        if i < 5:
            examples.append({
                'sentence':   sentence,
                'trajectory': trajectory,
                'final':      final_triples,
                'gold':       golds,
            })

        if (i + 1) % 10 == 0:
            cur = evaluate(pred_list, gold_list)
            print(f"[{i+1}/{len(samples)}] "
                  f"P:{cur['precision']:.2%} "
                  f"R:{cur['recall']:.2%} "
                  f"F1:{cur['f1']:.2%} "
                  f"| 触发反思: {multi_trial_cnt}")

    metrics = evaluate(pred_list, gold_list)

    print(f"\n{'='*60}")
    print(f"📊 [Reflexion | {dataset_name.upper()} | {model_name}] 最终结果：")
    print(f"  Precision  : {metrics['precision']:.2%}")
    print(f"  Recall     : {metrics['recall']:.2%}")
    print(f"  F1         : {metrics['f1']:.2%}")
    print(f"  预测/答案/正确：{metrics['total_pred']} / {metrics['total_gold']} / {metrics['tp']}")
    print(f"  触发反思样本: {multi_trial_cnt} / {len(samples)} ({multi_trial_cnt/len(samples):.1%})")

    # 展示 Reflexion 轨迹示例
    if examples and len(examples[0]['trajectory']) > 1:
        ex = examples[0]
        print(f"\n【Reflexion 轨迹示例】")
        print(f"  文本: {ex['sentence'][:80]}...")
        for t in ex['trajectory']:
            print(f"  Trial {t['trial']}: score={t['score']:.2f} | triples={t['triples']}")
            if t['reflection']:
                print(f"  反思: {t['reflection'][:150]}...")

    fname = f"result_reflexion_{model_name}_t{max_trials}_{dataset_name}.json"
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump({
            'method':    'Reflexion-style Self-Supervised KG Construction',
            'reference': 'Shinn et al. NeurIPS 2023, arXiv:2303.11366',
            'model':     model_name,
            'dataset':   dataset_name,
            'max_trials':max_trials,
            'metrics':   {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in metrics.items()},
            'multi_trial_rate': round(multi_trial_cnt / len(samples), 4),
            'knowledge_graph':  {'total': len(all_triples), 'triples': all_triples},
            'examples':  examples,
        }, f, indent=2, ensure_ascii=False)
    print(f"✅ 结果保存到 {fname}")
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',    choices=['conll04', 'nyt'], default='conll04')
    parser.add_argument('--model',      choices=list(MODEL_PATHS.keys()), default='qwen7b')
    parser.add_argument('--max_trials', type=int, default=MAX_TRIALS,
                        help='Reflexion 最大尝试轮数（默认2）')
    args = parser.parse_args()
    run(args.dataset, args.model, args.max_trials)
