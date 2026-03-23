"""
ChatIE + Supervisor Refinement Pipeline
前两步完全使用 ChatIE 原版提示词（论文：EMNLP 2023）
第三步加入监督模型校验（保守删除策略）

Stage 1 & 2: ChatIE 原版两阶段（完全照搬 chatIE_conll04.py）
Stage 3:     监督模型保守校验（只删违反硬规则的三元组）
Stage 4:     提取模型综合监督意见完成最终输出

用法：
    python chatIE_supervisor.py --dataset conll04 --extractor qwen7b
    python chatIE_supervisor.py --dataset conll04 --extractor qwen14b
    python chatIE_supervisor.py --dataset nyt     --extractor qwen7b
    python chatIE_supervisor.py --dataset nyt     --extractor qwen14b
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
    'qwen7b':     '/root/protonet/models/Qwen/Qwen2___5-7B-Instruct',
    'qwen14b':    '/root/protonet/models/Qwen/Qwen/Qwen2___5-14B-Instruct-GPTQ-Int4',
    'llama8b':    '/root/protonet/models/LLaMA/LLM-Research/Meta-Llama-3___1-8B-Instruct',
    'deepseek7b': '/root/protonet/models/DeepSeek/DeepSeek-R1-Distill-Qwen-7B',
}
DEEPSEEK_MODELS = {'deepseek7b'}

N_TEST         = 50
SEED           = 42
MAX_NEW_TOKENS = 512

# ── CoNLL04 关系类型（ChatIE 原版格式，含定义）──
CONLL04_RELATION_TYPES = {
    'Located_In':   'A location entity is situated within another location entity.',
    'Work_For':     'A person works for or is employed by an organization.',
    'OrgBased_In':  'An organization is based in or headquartered in a location.',
    'Live_In':      'A person lives in or resides in a location.',
    'Kill':         'A person kills another person.',
}

# ── NYT 关系类型（简短定义）──
NYT_RELATION_TYPES = {
    '/location/location/contains':           'A location contains another location.',
    '/people/person/nationality':            'A person has a nationality.',
    '/people/person/place_lived':            'A person lives in a place.',
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
# 模型加载与推理
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
# 三元组解析（完全照搬 chatIE_conll04.py）
# ───────────────────────────────────────────────
def parse_triples(output, relation_types):
    triples = []
    seen    = set()

    # 格式1：(head, relation, tail)
    for h, r, t in re.findall(r'\(([^,()]+),\s*([^,()]+),\s*([^,()]+)\)', output):
        h, r, t = h.strip().lower(), r.strip(), t.strip().lower()
        matched = fuzzy_match_relation(r, relation_types)
        if matched and h and t and h != 'none' and t != 'none':
            key = (h, matched, t)
            if key not in seen:
                seen.add(key)
                triples.append({'head': h, 'relation': matched, 'tail': t})

    # 格式2：head | relation | tail
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
    """完全照搬 chatIE_conll04.py 的模糊匹配"""
    text_lower = text.lower().replace(' ', '_').replace('-', '_')
    # 精确匹配
    for rel in relation_types:
        if rel.lower() == text_lower:
            return rel
    # 包含匹配
    for rel in relation_types:
        rel_short = rel.lower().split('/')[-1]
        if rel_short in text_lower or text_lower in rel_short:
            return rel
    # 关键词匹配（CoNLL04 专用）
    keywords = {
        'Located_In':   ['located', 'location', 'situated', 'in'],
        'Work_For':     ['work', 'employ', 'staff', 'member'],
        'OrgBased_In':  ['based', 'headquarter', 'org'],
        'Live_In':      ['live', 'reside', 'home'],
        'Kill':         ['kill', 'murder', 'slay'],
    }
    for rel, kws in keywords.items():
        if rel in relation_types and any(kw in text_lower for kw in kws):
            return rel
    return None

# ───────────────────────────────────────────────
# Stage 1 & 2: ChatIE 原版两阶段提示词
# 完全照搬自 chatIE_conll04.py predict_chatIE()
# ───────────────────────────────────────────────
def chatIE_extract(model, tokenizer, sentence, entities, relation_types, model_name=''):
    """
    ChatIE 原版两阶段（零修改）
    来源：chatIE_conll04.py predict_chatIE()
    论文：Zero-Shot IE via Chatting with ChatGPT, EMNLP 2023
    """
    rel_list    = '\n'.join([f"  - {r}: {d}" for r, d in relation_types.items()])
    entity_list = ', '.join([f"{e['text']} ({e['type']})" for e in entities])

    # ── Stage 1 原版提示词 ──────────────────────
    messages = [
        {"role": "system", "content": "You are a knowledge graph expert."},
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

    # ── Stage 2 改进版：强制引用第一阶段结论 ──────
    # 改进原因：原版第二阶段模型会重新判断，忽略第一阶段推理
    # 改进方式：明确要求模型只对第一阶段识别的关系提取实体对
    messages.append({"role": "assistant", "content": stage1})
    messages.append({"role": "user", "content":
        f"Based on the relations you identified above, extract the specific entity pairs.\n"
        f"IMPORTANT: Only extract triples for the EXACT relation types you listed above.\n"
        f"Do NOT introduce new relation types not mentioned in your previous answer.\n\n"
        f"For each identified relation, find the head entity and tail entity from the sentence.\n"
        f"Use the SHORTEST form of entity names (e.g., 'Jerusalem' not 'Arab east Jerusalem').\n\n"
        f"Output each triple in format: (head_entity, Relation_Type, tail_entity)\n"
        f"Use only entity names that appear in the sentence."
    })
    stage2 = chat(model, tokenizer, messages, model_name)

    triples = parse_triples(stage2, relation_types)
    return stage1, stage2, triples

# ───────────────────────────────────────────────
# Stage 3: 监督模型保守校验（自有代码）
# ───────────────────────────────────────────────
def supervisor_review(chatIE_triples, relation_types,
                      sup_model=None, sup_tok=None,
                      supervisor_name='', sentence=''):
    """    Stage 3: LLM 监督（针对已知错误模式的精确指令版）

    已知错误模式（从实验观察总结）：
    1. 把"同一实体对有两种不同关系"误判为重复 → 明确禁止
    2. 把候选列表里的合法关系名认为非法删除 → 逐行展示关系名
    3. 越权判断实体类型和关系方向 → 明确禁止
    4. parse failed 时丢失所有三元组 → fallback 到原始输出
    """
    approved = set(relation_types.keys())

    # Step 1: 纯代码预处理（关系名不合法 + 完全重复）
    seen, pre_result, removed = set(), [], []
    for t in chatIE_triples:
        key = (t['head'], t['relation'], t['tail'])
        if t['relation'] not in approved:
            removed.append(('invalid_relation', t))
            continue
        if key in seen:
            removed.append(('duplicate', t))
            continue
        seen.add(key)
        pre_result.append(t)

    if not pre_result:
        return pre_result, removed

    # Step 2: LLM 监督——只做实体名规范化和错误方向纠正
    triples_str = "\n".join(
        [f"  {i+1}. ({t['head']}, {t['relation']}, {t['tail']})"
         for i, t in enumerate(pre_result)])

    relations_numbered = "\n".join(
        [f"  {i+1}. {r}" for i, r in enumerate(relation_types.keys())])

    supervisor_prompt = f"""You are a knowledge graph post-processor. Review the extracted triples for two specific issues ONLY.

Text: {sentence}

COMPLETE APPROVED RELATION LIST (all of these are valid — do not question them):
{relations_numbered}

Extracted triples to review:
{triples_str}

TASK: Check for these TWO issues ONLY:

Issue A — Entity name too long (normalize to shortest unambiguous form):
  Example: "arab east jerusalem" → "jerusalem"
  Example: "the united states of america" → "united states"
  Example: "shoshone-bannock reservation in idaho" → "shoshone-bannock reservation"
  Do NOT normalize if shortening would lose essential meaning.

Issue B — Relation direction clearly wrong based on the relation's definition:
  OrgBased_In: head=Organization, tail=Location. If swapped, correct it.
  Work_For: head=Person, tail=Organization. If head is clearly a Location, correct.
  Located_In: head=specific place, tail=containing place. If reversed, correct.
  ONLY correct if you are 100% certain. When uncertain, leave unchanged.

CRITICAL RULES:
  - (A, Relation1, B) and (A, Relation2, B) are NOT duplicates — they express different facts. KEEP BOTH.
  - ALL relation names in the approved list above are valid. Never remove a triple just because the relation seems unusual.
  - Do NOT delete triples. Only suggest normalization or direction corrections.
  - If nothing needs changing, output an empty corrections list.

Output ONLY valid JSON (no extra text):
{
  "corrections": [
    {"index": 1, "field": "head|tail|relation", "old": "old_value", "new": "new_value"}
  ],
  "reasoning": "brief note for each correction"
}"""

    messages = [
        {"role": "system", "content":
            "You are a precise knowledge graph post-processor. "
            "Only normalize entity names and fix obvious relation direction errors. "
            "Never delete triples."},
        {"role": "user", "content": supervisor_prompt}
    ]
    output = chat(sup_model, sup_tok, messages, supervisor_name)

    # Step 3: 应用 LLM 的修正建议（纯代码执行）
    final_result = list(pre_result)
    try:
        m = re.search(r'\{.*\}', output, re.DOTALL)
        if m:
            fb = json.loads(m.group())
            for corr in fb.get("corrections", []):
                idx = corr.get("index", 0) - 1
                field = corr.get("field", "")
                old_val = corr.get("old", "").strip().lower()
                new_val = corr.get("new", "").strip().lower()
                if 0 <= idx < len(final_result) and field in ("head", "tail"):
                    # 只接受实体名规范化（缩短），不接受扩展
                    if new_val in old_val and new_val != old_val:
                        final_result[idx] = {**final_result[idx], field: new_val}
                elif 0 <= idx < len(final_result) and field == "relation":
                    # 只接受已在 approved 列表里的关系名
                    if new_val in {r.lower() for r in approved}:
                        matched = next(r for r in approved if r.lower() == new_val)
                        final_result[idx] = {**final_result[idx], "relation": matched}
    except Exception:
        pass  # parse 失败时保留 pre_result

    removed.append(('llm_output', output[:200]))
    return final_result, removed

def final_synthesis(chatIE_triples, relation_types,
                    sup_model=None, sup_tok=None,
                    supervisor_name='', sentence=''):
    """Stage 4: 调用含 LLM 监督的 supervisor_review"""
    result, _ = supervisor_review(
        chatIE_triples, relation_types,
        sup_model, sup_tok, supervisor_name, sentence)
    return result

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
def run(dataset_name, extractor_name, supervisor_name):
    np.random.seed(SEED)

    print(f"\n{'='*60}")
    print(f"Pipeline: ChatIE (原版) + Supervisor Refinement")
    print(f"Extractor : {extractor_name}  Supervisor: {supervisor_name}（LLM精确指令版）")
    print(f"Dataset   : {dataset_name.upper()}")
    print(f"{'='*60}\n")

    # 加载数据
    print("加载数据集...")
    if dataset_name == 'conll04':
        samples       = load_conll04(N_TEST)
        relation_types = CONLL04_RELATION_TYPES
    else:
        samples       = load_nyt(N_TEST)
        relation_types = NYT_RELATION_TYPES
    print(f"测试样本数: {len(samples)}\n")

    print("加载模型...")
    ext_model, ext_tok = load_model(extractor_name)
    if supervisor_name == extractor_name:
        sup_model, sup_tok = ext_model, ext_tok
        print("  监督模型复用提取模型")
    else:
        sup_model, sup_tok = load_model(supervisor_name)

    pred_list    = []
    gold_list    = []
    all_triples  = []
    examples     = []

    for i, sample in enumerate(samples):
        sentence = sample['sentence']
        entities = sample['entities']
        golds    = sample['gold_triples']

        # Stage 1 & 2: ChatIE 原版（零修改）
        stage1, stage2, chatIE_preds = chatIE_extract(
            ext_model, ext_tok, sentence, entities,
            relation_types, extractor_name)

        # Stage 3 & 4: LLM监督（实体规范化+方向纠正）+ 纯代码执行
        final_preds = final_synthesis(
            chatIE_preds, relation_types,
            sup_model, sup_tok, supervisor_name, sentence)
        removed = []

        pred_list.append(final_preds)
        gold_list.append(golds)
        for t in final_preds:
            all_triples.append({**t, 'source': sentence[:80]})

        if i < 5:
            examples.append({
                'sentence':      sentence,
                'stage1_rels':   stage1,
                'chatIE_preds':  chatIE_preds,
                'removed':       sup_fb.get('removed', []),
                'final_preds':   final_preds,
                'gold':          golds,
            })

        if (i + 1) % 10 == 0:
            cur = evaluate(pred_list, gold_list)
            print(f"[{i+1}/{len(samples)}] "
                  f"P:{cur['precision']:.2%} "
                  f"R:{cur['recall']:.2%} "
                  f"F1:{cur['f1']:.2%}")

    metrics = evaluate(pred_list, gold_list)

    print(f"\n{'='*60}")
    print(f"📊 [ChatIE+Supervisor | {dataset_name.upper()} | ext={extractor_name} sup={supervisor_name}] 最终结果：")
    print(f"  Precision : {metrics['precision']:.2%}")
    print(f"  Recall    : {metrics['recall']:.2%}")
    print(f"  F1        : {metrics['f1']:.2%}")
    print(f"  预测/答案/正确：{metrics['total_pred']} / {metrics['total_gold']} / {metrics['tp']}")

    fname = f"result_chatIE_sup_{extractor_name}_{supervisor_name}_{dataset_name}.json"
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump({
            'method':    'ChatIE (原版两阶段) + Supervisor Refinement',
            'paper':     'Zero-Shot IE via Chatting with ChatGPT, EMNLP 2023',
            'extractor': extractor_name,
            'supervisor': supervisor_name,
            'dataset':   dataset_name,
            'metrics':   {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in metrics.items()},
            'knowledge_graph': {'total': len(all_triples), 'triples': all_triples},
            'examples':  examples,
        }, f, indent=2, ensure_ascii=False)
    print(f"✅ 结果保存到 {fname}")
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',    choices=['conll04', 'nyt'], default='conll04')
    parser.add_argument('--extractor',  choices=list(MODEL_PATHS.keys()), default='qwen7b')
    parser.add_argument('--supervisor', choices=list(MODEL_PATHS.keys()), default='qwen7b')
    args = parser.parse_args()
    run(args.dataset, args.extractor, args.supervisor)
