#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 10:06:04 2025

@author: paveenhuang
"""


from pathlib import Path
import numpy as np
import torch
import argparse
from tqdm import tqdm
from llms import VicundaModel
from detection.task_list import TASKS
from template import select_templates
import utils

# ─────────────────────────── Main ───────────────────────────────

def main():
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    
    templates = select_templates(suite="action")
    LABELS = templates["labels"]    

    for task in TASKS:
        print(f"\n=== {task} ===")
        # ID
        opt_ids = utils.option_token_ids(vc, LABELS)
        # Load data
        data_path = MMLU_DIR / f"{task}.json"
        samples = utils.load_json(data_path)
        # role map
        roles = utils.make_characters(task, args.type)
        role_stats = {r: {str(i): 0 for i in range(10)} | {"total": 0} for r in roles}
        # record and print template
        tmp_record = utils.record_template(roles, templates)
        
        with torch.no_grad():
            for sample in tqdm(samples, desc=task):
                ctx = sample["text"]

                for role in roles:
                    prompt = utils.construct_prompt(vc, templates, ctx, role, False)
                    logits = vc.get_logits([prompt], return_hidden=False)
                    logits = logits[0, -1].cpu().numpy()

                    # softmax over score options (0–9)
                    opt_logits = np.array([logits[i] for i in opt_ids])
                    probs = utils.softmax_1d(opt_logits)
                    pred_idx = int(opt_logits.argmax())
                    pred_label = LABELS[pred_idx]   # "0"..."9"
                    pred_prob = float(probs[pred_idx])

                    # attach outputs to sample
                    sample[f"score_{role.replace(' ', '_')}"] = pred_label
                    sample[f"score_prob_{role.replace(' ', '_')}"] = pred_prob
                    sample[f"score_dist_{role.replace(' ', '_')}"] = probs.tolist()
                    sample[f"logits_{role.replace(' ', '_')}"] = opt_logits.tolist()

                    # update stats (count how many times each score is chosen)
                    rs = role_stats[role]
                    rs["total"] += 1
                    rs[pred_label] = rs.get(pred_label, 0) + 1
    


        # accuracy summary
        accuracy = {}
        for role, s in role_stats.items():
            pct = s["correct"] / s["total"] * 100 if s["total"] else 0
            accuracy[role] = {**s, "accuracy_percentage": round(pct, 2)}
            print(f"{role:<25} acc={pct:5.2f}%  (correct {s['correct']}/{s['total']}), E={s['E_count']}")

        # save answers JSON
        ans_file = ANS_DIR / f"{task}_{args.size}_answers.json"
        utils.dump_json({"data": samples, "accuracy": accuracy, "template": tmp_record}, ans_file)
        print("[Saved answers]", ans_file)


    print("\n✅  All tasks finished.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run MMLU role-based extraction with hidden-state saving")
    parser.add_argument("--model", "-m", required=True, help="Model name, used for folder naming")
    parser.add_argument("--size", "-s", required=True, help="Model size, e.g., `8B`")
    parser.add_argument("--type", required=True, help="Role type identifier, affects prompt and output directories")
    parser.add_argument("--model_dir", required=True, help="LLM checkpoint/model directory")
    parser.add_argument("--ans_file", required=True, help="LLM checkpoint/model directory")
    parser.add_argument("--save", action="store_true", help="Whether to save hidden states (default saves only logits/answers)")
    parser.add_argument("--use_chat", action="store_true", help="Use tokenizer.apply_chat_template for prompts")
    
    args = parser.parse_args()

    print("model: ", args.model)
    print("Loading model from:", args.model_dir)

    MMLU_DIR = Path("/data2/paveen/RolePlaying/components/mmlu")
    ANS_DIR = Path(f"/data2/paveen/RolePlaying/components/{args.ans_file}/{args.model}")
    HS_DIR = Path(f"/data2/paveen/RolePlaying/components/hidden_states_{args.type}/{args.model}")
    ANS_DIR.mkdir(parents=True, exist_ok=True)
    HS_DIR.mkdir(parents=True, exist_ok=True)
    main()
