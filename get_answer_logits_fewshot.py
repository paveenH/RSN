#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 11:55:17 2025

@author: paveenhuang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract highest-logit answer + probability **and** save last-token hidden states
for every role on every MMLU task.
"""

from pathlib import Path
import numpy as np
import torch
import argparse
from tqdm import tqdm
from llms import VicundaModel
from detection.task_list import TASKS
from template import select_templates
from utils import load_json, option_token_ids, build_fewshot_prefix, softmax_1d, dump_json, make_characters, construct_prompt



def rkey(role: str, suf: str):
    return f"{suf}_{role.replace(' ', '_')}"

def main():
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    
    templates = select_templates(args.use_E)
    LABELS = templates["labels"]

    for task in TASKS:
        print(f"\n=== {task} ===")
        opt_ids = option_token_ids(vc, LABELS)
        data_path = MMLU_DIR / f"{task}.json"
        samples = load_json(data_path)
        roles = make_characters(task, "non")
        role_stats = {r: {"correct": 0, "E_count": 0, "invalid": 0, "total": 0} for r in roles}
        
        fewshot_prefix = build_fewshot_prefix(task=task, k=5)
        
        print(fewshot_prefix)
        print("\n")
        for role in roles:
            print (f"{role} prompt")
            print("------------------")
            print(templates[role])
            print("----------------")

        with torch.no_grad():
            for sample in tqdm(samples, desc=task):
                ctx = sample["text"]
                true_idx = sample["label"]
                true_label = LABELS[true_idx]

                for role in roles:
                    # template = templates[role]
                    question_block = construct_prompt(vc, templates, ctx, role, use_chat=False)
                    prompt = f"{fewshot_prefix}\n{question_block}"
                    
                    logits = vc.get_logits([prompt], return_hidden=False)
                    logits = logits[0, -1].cpu().numpy()

                    # softmax over answer options
                    opt_logits = np.array([logits[i] for i in opt_ids])
                    probs = softmax_1d(opt_logits)
                    pred_idx = int(opt_logits.argmax())
                    pred_label = LABELS[pred_idx]
                    pred_prob = float(probs[pred_idx])

                    # attach answer+prob to sample
                    sample[rkey(role, "answer")] = pred_label
                    sample[rkey(role, "prob")] = pred_prob
                    sample[rkey(role, "softmax_" + task)] = probs.tolist()
                    sample[rkey(role, "logits_" + task)] = opt_logits.tolist()

                    # update stats
                    rs = role_stats[role]
                    rs["total"] += 1
                    if pred_label == true_label:
                        rs["correct"] += 1
                    elif pred_label == "E":
                        rs["E_count"] += 1
                    else:
                        rs["invalid"] += 1

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

    parser = argparse.ArgumentParser(description="Run MMLU role-based extraction with hidden-state saving")
    parser.add_argument("--model", "-m", required=True, help="Model name, used for folder naming")
    parser.add_argument("--size", "-s", required=True, help="Model size, e.g., `8B`")
    parser.add_argument("--model_dir", required=True, help="LLM checkpoint/model directory")
    parser.add_argument("--ans_file", required=True, help="LLM checkpoint/model directory")
    parser.add_argument("--use_E", action="store_true", help="Use five-choice template (A–E); otherwise use four-choice (A–D)")

    args = parser.parse_args()

    print("model: ", args.model)
    print("Loading model from:", args.model_dir)

    MMLU_DIR = Path("/data2/paveen/RolePlaying/components/mmlu")
    ANS_DIR = Path(f"/data2/paveen/RolePlaying/components/{args.ans_file}/{args.model}")
    ANS_DIR.mkdir(parents=True, exist_ok=True)
    main()
