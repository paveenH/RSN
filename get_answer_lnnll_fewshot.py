#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute 4-option predictions using length-normalized NLL (LN-NLL) like lm-eval-harness.
Few-shot prefix: build_fewshot_prefix(task, k=5)
Template: "{context}\nAnswer: "
Scoring: compare candidates " A"/" B"/" C"/" D" by LN-NLL over the answer segment only.
"""

import json
from pathlib import Path
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm

from llms import VicundaModel
from detection.task_list import TASKS
from template import select_templates
from utils import load_json, build_fewshot_prefix


# ---------- LN-NLL helper ----------

def dump_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def rkey(role: str, suf: str):
    return f"{suf}_{role.replace(' ', '_')}"

def ln_nll_for_answer_segment(
    logits: torch.Tensor,      # (L, V), for the full_text (prefix+query+answer)
    full_ids: List[int],       # token ids of full_text (no special tokens)
    prefix_len: int            # number of tokens before the answer segment starts
) -> float:
    """
    logits[t] predicts full_ids[t+1]. We score tokens at positions [prefix_len .. len-1],
    using rows [prefix_len-1 .. len-2].
    Return mean NLL over answer tokens (length-normalized).
    """
    L = len(full_ids)
    if L - 1 < prefix_len:  # no answer tokens
        return float("inf")

    rows = logits[prefix_len - 1 : L - 1] if prefix_len > 0 else logits[: L - 1]
    tgt = torch.tensor(full_ids[prefix_len:], dtype=torch.long, device=logits.device)
    logprobs = F.log_softmax(rows, dim=-1)
    picked = logprobs.gather(-1, tgt.view(-1, 1)).squeeze(-1)  # (answer_len,)
    return float((-picked.mean()).item())


def main():
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    templates = select_templates(False)
    LABELS = templates["labels"]  # ["A","B","C","D"]
    template = templates["vanilla"]
    
    for task in TASKS:
        print(f"\n=== {task} ===")
        fewshot_prefix = build_fewshot_prefix(task=task, k=5)
        print(fewshot_prefix)
        print("------------------")
        print(template)

        data_path = MMLU_DIR / f"{task}.json"
        samples = load_json(data_path)

        roles = ["vanilla"]  
        role_stats = {r: {"correct": 0, "total": 0} for r in roles}
        role = "vanilla"

        with torch.no_grad():
            for sample in tqdm(samples, desc=task):
                ctx = sample["text"]
                gold_idx = sample["label"]
                true_label = LABELS[gold_idx]
            
                query_block = template.format(context=ctx)
                base_prompt = f"{fewshot_prefix}\n{query_block}"

                base_ids = vc.tokenizer(base_prompt, add_special_tokens=False).input_ids
                prefix_len = len(base_ids)

                lnnlls = []
                for cand in LABELS:
                    full_text = base_prompt + cand
                    toks = vc.tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(vc.model.device)
                    outputs = vc.model(**toks, return_dict=True)
                    logits = outputs.logits[0]                # (L, V)
                    full_ids = toks.input_ids[0].tolist()     # align with logits in length
                    nll = ln_nll_for_answer_segment(logits, full_ids, prefix_len=prefix_len)
                    lnnlls.append(nll)

                pred_idx = int(np.argmin(lnnlls))
                pred_label = LABELS[pred_idx]
                
                # attach answer+prob to sample
                sample[rkey(role, "answer")] = pred_label
                sample[rkey(role, "lnnlls")] = lnnlls

                # update stats
                rs = role_stats[role]
                rs["total"] += 1
                if pred_label == true_label:
                    rs["correct"] += 1

                # record
                
                rs = role_stats[role]
                rs["total"] += 1
                if pred_idx == gold_idx:
                    rs["correct"] += 1

                sample[f"answer_{role}"] = pred_label
                sample[f"lnnll_{task}_{role}"] = lnnlls  
            # accuracy summary
            accuracy = {}
            for role, s in role_stats.items():
                pct = s["correct"] / s["total"] * 100 if s["total"] else 0
                accuracy[role] = {**s, "accuracy_percentage": round(pct, 2)}
                print(f"{role:<25} acc={pct:5.2f}%  (correct {s['correct']}/{s['total']}), E={s['E_count']}")

            # save answers JSON
            ans_file = ANS_DIR / f"{task}_{args.size}_answers.json"
            dump_json({"data": samples, "accuracy": accuracy}, ans_file)
            print("[Saved answers]", ans_file)

        print("\n✅  All tasks finished.")

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MMLU LN-NLL (4-choice, few-shot prefix, no chat template)")
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--size", "-s", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--ans_file", required=True)
    args = parser.parse_args()

    MMLU_DIR = Path("/data2/paveen/RolePlaying/components/mmlu")
    ANS_DIR = Path(f"/data2/paveen/RolePlaying/components/{args.ans_file}/{args.model}")
    ANS_DIR.mkdir(parents=True, exist_ok=True)

    main()