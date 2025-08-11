#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract highest-logit answer + probability **and** save last-token hidden states
for every role on every MMLU task.
"""

import json
from pathlib import Path
from typing import Dict
import numpy as np
import torch
import argparse
from tqdm import tqdm
from llms import VicundaModel
from detection.task_list import TASKS
from template import select_templates
from utils import load_json, make_characters,option_token_ids


# ───────────────────── Helper functions ─────────────────────────


def softmax_1d(x: np.ndarray):
    e = np.exp(x - x.max())
    return e / e.sum()


def rkey(role: str, suf: str):
    return f"{suf}_{role.replace(' ', '_')}"


def dump_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ─────────────────────────── Main ───────────────────────────────


def main():
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    templates = select_templates(args.use_E)
    LABELS = templates["labels"]

    for task in TASKS:
        print(f"\n=== {task} ===")
        template = templates["default"]
        neutral_template = templates["neutral"]
        neg_template = templates["neg"]
        LABELS = templates["labels"]
        if args.use_E:
            print(template)
        else:
            print(neutral_template)
        
        opt_ids = option_token_ids(vc, LABELS)

        data_path = MMLU_DIR / f"{task}.json"
        if not data_path.exists():
            print("[Skip]", data_path, "not found")
            continue

        samples = load_json(data_path)
        roles = make_characters(task, args.type)
        role_stats = {r: {"correct": 0, "E_count": 0, "invalid": 0, "total": 0} for r in roles}
        # store hidden states per role
        hs_store: Dict[str, list[np.ndarray]] = {r: [] for r in roles}

        with torch.no_grad():
            for sample in tqdm(samples, desc=task):
                ctx = sample["text"]
                true_idx = sample["label"]
                if not 0 <= true_idx < len(LABELS):
                    continue
                true_label = LABELS[true_idx]

                for role in roles:
                    if role == "norole":
                        prompt = neutral_template.format(context=ctx)
                    elif "not" in role:
                        prompt = neg_template.format(character=role, context=ctx)
                    else:
                        prompt = template.format(character=role, context=ctx)
                        
                    if args.save:
                        logits, hidden = vc.get_logits([prompt], return_hidden=args.save)
                        last_hs = [lay[0, -1].cpu().numpy() for lay in hidden]  # list(len_layers, hidden_size)
                        # accumulate hidden states
                        hs_store[role].append(np.stack(last_hs, axis=0))  # (layers, hidden)
                    else:
                        logits = vc.get_logits([prompt], return_hidden=args.save)

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

        # save hidden states per role
        if args.save:
            for role, arr_list in hs_store.items():
                if not arr_list:
                    continue
                hs_np = np.stack(arr_list, axis=0)  # (n_samples, layers, hidden)
                safe_role = role.replace(" ", "_").replace("-", "_")
                hs_file = HS_DIR / f"{safe_role}_{task}_{args.size}.npy"
                np.save(hs_file, hs_np)
                print("[Saved HS]", hs_file)

    print("\n✅  All tasks finished.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run MMLU role-based extraction with hidden-state saving")
    parser.add_argument("--model", "-m", required=True, help="Model name, used for folder naming")
    parser.add_argument("--size", "-s", required=True, help="Model size, e.g., `8B`")
    parser.add_argument("--type", required=True, help="Role type identifier, affects prompt and output directories")
    parser.add_argument("--model_dir", required=True, help="LLM checkpoint/model directory")
    parser.add_argument("--ans_file", required=True, help="LLM checkpoint/model directory")
    parser.add_argument("--use_E", action="store_true", help="Use five-choice template (A–E); otherwise use four-choice (A–D)")
    parser.add_argument("--save", action="store_true", help="Whether to save hidden states (default saves only logits/answers)")
    
    args = parser.parse_args()

    print("model: ", args.model)
    print("Loading model from:", args.model_dir)

    MMLU_DIR = Path("/data2/paveen/RolePlaying/components/mmlu")
    ANS_DIR = Path(f"/data2/paveen/RolePlaying/components/{args.ans_file}/{args.model}")
    HS_DIR = Path(f"/data2/paveen/RolePlaying/components/hidden_states_{args.type}/{args.model}")
    ANS_DIR.mkdir(parents=True, exist_ok=True)
    HS_DIR.mkdir(parents=True, exist_ok=True)
    main()
