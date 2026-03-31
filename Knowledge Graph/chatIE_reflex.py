"""
Reflexion-style Self-Supervised KG Construction Pipeline
参考：Shinn et al. "Reflexion: Language Agents with Verbal Reinforcement Learning" NeurIPS 2023
      arXiv:2303.11366

用法：
    python chatIE_reflexion.py --dataset conll04 --model qwen7b
    python chatIE_reflexion.py --dataset webnlg  --model qwen14b
    python chatIE_reflexion.py --dataset duie    --model qwen7b
"""

import os
import json
import argparse
import re
import glob
import xml.etree.ElementTree as ET
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

# WebNLG 使用开放式抽取（改进一）：不限定关系列表
# WebNLG 有 400+ 种关系，预设少量关系会导致大量三元组因关系名不匹配而丢失
# 改为让模型自由抽取关系名，再与 gold 做模糊匹配
WEBNLG_RELATION_TYPES = None  # None 表示开放式，不限制关系类型

DUIE_RELATION_TYPES = {
    '配音':      'Voice Acting: A person provides voice acting for a character, movie, or show.',
    '导演':      'Director: A person directs a movie, show, or other audiovisual work.',
    '主演':      'Starring: A person stars as a main actor/actress in a movie or show.',
    '歌手':      'Singer: A person is the singer of a song or album.',
    '出生地':    'Birthplace: A person was born in a specific location.',
    '毕业院校':  'Alma Mater: A person graduated from a specific university or school.',
    '成立日期':  'Founded Date: An organization or company was founded on a specific date.',
    '作者':      'Author: A person is the author of a book or written work.',
    '妻子':      'Wife: A man is married to a woman (his wife).',
    '丈夫':      'Husband: A woman is married to a man (her husband).',
    '父母':      'Parents: A person is the parent of another person.',
    '朝代':      'Dynasty: A historical figure or event belongs to a specific Chinese dynasty.',
    '首都':      'Capital: A country or region has a specific capital city.',
    '总部地点':  'Headquarters: A company or organization is headquartered in a specific location.',
    '创始人':    'Founder: A person founded a company or organization.'
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

def load_webnlg_local(data_dir='/root/data/webnlg-dataset/release_v3.0/en/dev/**/*.xml', n=None):
    print(f"  从本地 XML 加载 WebNLG (Dev集)...")
    samples = []
    files = glob.glob(data_dir, recursive=True)
    if not files:
        print(f"  ⚠️ 警告: 未找到 WebNLG xml 文件！")
        return []
    for fpath in files:
        try:
            tree = ET.parse(fpath)
            root = tree.getroot()
            for entry in root.iter('entry'):
                lex = entry.find('.//lex')
                if lex is None or not lex.text:
                    continue
                sentence = lex.text.strip()
                golds = []
                for mtriple in entry.findall('.//mtriple'):
                    if mtriple.text:
                        parts = [p.strip() for p in mtriple.text.split('|')]
                        if len(parts) == 3:
                            head = parts[0].replace('_', ' ').lower()
                            relation = parts[1]
                            tail = parts[2].replace('_', ' ').lower()
                            golds.append({'head': head, 'relation': relation, 'tail': tail})
                if golds:
                    ent_set = set()
                    for t in golds:
                        ent_set.add(t['head'])
                        ent_set.add(t['tail'])
                    entities = [{'text': e, 'type': 'Entity'} for e in ent_set]
                    samples.append({'sentence': sentence, 'entities': entities, 'gold_triples': golds})
        except Exception:
            pass
    return samples[:n] if n else samples

def load_duie_local(data_dir='/root/data/duie', n=None):
    print(f"  从本地 Parquet 加载 DuIE...")
    samples = []
    try:
        data_files = {"validation": f"{data_dir}/data/*validation*.parquet"}
        ds = load_dataset("parquet", data_files=data_files)
        for item in ds['validation']:
            sentence = item['text']
            spo_list = item['spo_list']
            golds = []
            entities = []
            for spo in spo_list:
                head = spo['subject']
                relation = spo['predicate']
                tail = spo['object']
                golds.append({'head': head.lower(), 'relation': relation, 'tail': tail.lower()})
                entities.append({'text': head, 'type': spo.get('subject_type', 'Entity')})
                entities.append({'text': tail, 'type': spo.get('object_type', 'Entity')})
            if golds:
                samples.append({'sentence': sentence, 'entities': entities, 'gold_triples': golds})
    except Exception:
        pass
    return samples[:n] if n else samples

# ───────────────────────────────────────────────
# 模型推理 & 三元组解析
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

def parse_triples(output, relation_types):
    """
    改进二：支持 tail 包含逗号的情况（如 "harrietstown, new york"）
    策略：先按行解析，每行找第一个和第二个逗号作为分隔符
    """
    triples, seen = [], set()
    is_open = relation_types is None  # WebNLG 开放式

    # 格式1：(head, relation, tail) — tail 可能含逗号
    # 用贪婪匹配：第一个逗号前=head，最后一个逗号后=tail，中间=relation
    for match in re.finditer(r'\(([^()]+)\)', output):
        inner = match.group(1)
        parts = inner.split(',')
        if len(parts) >= 3:
            h   = parts[0].strip().lower()
            r   = parts[1].strip()
            t   = ','.join(parts[2:]).strip().lower()  # tail 可含逗号
            matched = r if is_open else fuzzy_match_relation(r, relation_types)
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
            matched = r if is_open else fuzzy_match_relation(r, relation_types)
            if matched and h and t:
                key = (h, matched, t)
                if key not in seen:
                    seen.add(key)
                    triples.append({'head': h, 'relation': matched, 'tail': t})
    return triples

def fuzzy_match_relation(text, relation_types):
    if relation_types is None:
        return text  # 开放式：直接返回原始关系名
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

def fuzzy_match_gold(pred_rel, gold_rel):
    """改进一：WebNLG 开放式评估时，用模糊匹配对齐 gold 关系名"""
    p = pred_rel.lower().replace('_', '').replace(' ', '').replace('-', '')
    g = gold_rel.lower().replace('_', '').replace(' ', '').replace('-', '')
    return p == g or p in g or g in p

# ───────────────────────────────────────────────
# Actor & Evaluator & Reflector
# ───────────────────────────────────────────────
def actor_extract(model, tokenizer, sentence, entities, relation_types,
                  model_name='', memory_buffer=''):
    
    if relation_types:
        rel_list = '\n'.join([f"  - {r}: {d}" for r, d in relation_types.items()])
        rel_prompt = f"Relation types:\n{rel_list}\n\nList only the relation types that exist (one per line). If none exist, say 'None'."
    else:
        rel_list = ""
        rel_prompt = "Identify ALL relations between entities in the sentence. List each relation type you find (one per line). Use concise relation names like 'birthPlace', 'leader', 'location'. If none exist, say 'None'."

    entity_list = ', '.join([f"{e['text']} ({e['type']})" for e in entities])

    system_content = "You are a knowledge graph expert."
    if memory_buffer:
        system_content += (
            "\n\nLESSONS FROM PREVIOUS ATTEMPTS (Reflexion Memory):\n"
            + memory_buffer
            + "\nApply these lessons to improve your extraction."
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content":
            f"Given this sentence, which of the following relation types exist "
            f"between the entities?\n\n"
            f"Sentence: {sentence}\n"
            f"Entities: {entity_list}\n\n"
            f"{rel_prompt}"}
    ]
    stage1 = chat(model, tokenizer, messages, model_name)

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
    
    if relation_types:
        approved = set(relation_types.keys())
    else:
        approved = None
        
    seen, result = set(), []
    for t in triples:
        key = (t['head'], t['relation'], t['tail'])
        if (approved is None or t['relation'] in approved) and key not in seen:
            seen.add(key)
            result.append(t)

    return stage1, stage2, result

def evaluator(model, tokenizer, sentence, triples, relation_types, model_name=''):
    if not triples:
        return 0.0, {'summary': 'No triples were extracted. The sentence likely contains relations that were missed.', 'missing': ['All relations were missed'], 'wrong': [], 'entity_issues': []}

    triples_str = '\n'.join(
        [f"  {i+1}. ({t['head']}, {t['relation']}, {t['tail']})"
         for i, t in enumerate(triples)])
         
    if relation_types:
        rel_defs = '\n'.join([f"  - {r}: {d}" for r, d in relation_types.items()])
    else:
        rel_defs = "Open extraction mode: No predefined relation types. Evaluate based on general knowledge graph validity."

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

def self_reflector(model, tokenizer, sentence, triples, eval_issues,
                   relation_types, model_name='', previous_reflection=''):
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

def reflexion_extract(model, tokenizer, sentence, entities, relation_types,
                      model_name='', max_trials=2, score_threshold=0.85):
    memory_buffer = ""
    best_triples  = []
    best_score    = 0.0
    trajectory    = []

    for trial in range(max_trials):
        stage1, stage2, triples = actor_extract(
            model, tokenizer, sentence, entities, relation_types,
            model_name, memory_buffer)

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

        if score > best_score or trial == 0:
            best_score   = score
            best_triples = triples

        if score >= score_threshold or trial == max_trials - 1:
            break

        reflection = self_reflector(
            model, tokenizer, sentence, triples, eval_issues,
            relation_types, model_name, memory_buffer)

        trajectory[-1]['reflection'] = reflection

        memory_buffer = (memory_buffer + "\n" if memory_buffer else "") + \
                        f"[Trial {trial+1} reflection]: " + reflection

    return best_triples, best_score, trajectory

# ───────────────────────────────────────────────
# 评估 & 主函数
# ───────────────────────────────────────────────
def evaluate(pred_list, gold_list, fuzzy_rel=False):
    """
    改进一：fuzzy_rel=True 时（WebNLG），关系名用模糊匹配
    实体名仍要求完全匹配（lower 后）
    """
    tp = fp = fn = 0
    for preds, golds in zip(pred_list, gold_list):
        if fuzzy_rel:
            # WebNLG 开放式：关系名模糊匹配
            matched_gold = set()
            for p in preds:
                hit = False
                for gi, g in enumerate(golds):
                    if gi in matched_gold:
                        continue
                    if (p['head'] == g['head'] and
                            p['tail'] == g['tail'] and
                            fuzzy_match_gold(p['relation'], g['relation'])):
                        tp += 1
                        matched_gold.add(gi)
                        hit = True
                        break
                if not hit:
                    fp += 1
            fn += len(golds) - len(matched_gold)
        else:
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
    elif dataset_name == 'nyt':
        samples        = load_nyt(N_TEST)
        relation_types = NYT_RELATION_TYPES
    elif dataset_name == 'webnlg':
        samples        = load_webnlg_local(n=N_TEST)
        relation_types = WEBNLG_RELATION_TYPES
    elif dataset_name == 'duie':
        samples        = load_duie_local(n=N_TEST)
        relation_types = DUIE_RELATION_TYPES
        
    print(f"测试样本数: {len(samples)}\n")

    print("加载模型...")
    model, tokenizer = load_model(model_name)

    pred_list    = []
    gold_list    = []
    all_triples  = []
    examples     = []
    multi_trial_cnt = 0

    for i, sample in enumerate(samples):
        sentence = sample['sentence']
        entities = sample['entities']
        golds    = sample['gold_triples']

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
            cur = evaluate(pred_list, gold_list, fuzzy_rel=(dataset_name=='webnlg'))
            print(f"[{i+1}/{len(samples)}] "
                  f"P:{cur['precision']:.2%} "
                  f"R:{cur['recall']:.2%} "
                  f"F1:{cur['f1']:.2%} "
                  f"| 触发反思: {multi_trial_cnt}")

    fuzzy_rel = (dataset_name == 'webnlg')
    metrics = evaluate(pred_list, gold_list, fuzzy_rel=fuzzy_rel)

    print(f"\n{'='*60}")
    print(f"📊 [Reflexion | {dataset_name.upper()} | {model_name}] 最终结果：")
    print(f"  Precision  : {metrics['precision']:.2%}")
    print(f"  Recall     : {metrics['recall']:.2%}")
    print(f"  F1         : {metrics['f1']:.2%}")
    print(f"  预测/答案/正确：{metrics['total_pred']} / {metrics['total_gold']} / {metrics['tp']}")
    print(f"  触发反思样本: {multi_trial_cnt} / {len(samples)} ({multi_trial_cnt/len(samples):.1%})")

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
    parser.add_argument('--dataset',    choices=['conll04', 'nyt', 'webnlg', 'duie'], default='conll04')
    parser.add_argument('--model',      choices=list(MODEL_PATHS.keys()), default='qwen7b')
    parser.add_argument('--max_trials', type=int, default=MAX_TRIALS,
                        help='Reflexion 最大尝试轮数（默认2）')
    args = parser.parse_args()
    run(args.dataset, args.model, args.max_trials)
