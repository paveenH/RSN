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
from tqdm import tqdm

import get_answer_alltasks as ga
from llms import VicundaModel

# ─────────────────────── Configuration ──────────────────────────
TASKS       = ga.TASKS    
MODEL = "mistral_base"      # list of MMLU tasks
SIZE        = "7B"
TYPE        = "non"
LABELS      = ["A", "B", "C", "D", "E"]
SAVE = False
 
MODEL_DIR   = "mistralai/Mistral-7B-v0.3"
# MODEL_DIR = "mistralai/Mistral-7B-Instruct-v0.3"
print("Loading model from:", MODEL_DIR)

MMLU_DIR    = Path("/data2/paveen/RolePlaying/components/mmlu")
ANS_DIR     = Path(f"/data2/paveen/RolePlaying/components/answer_{TYPE}_logits/{MODEL}")
HS_DIR      = Path(f"/data2/paveen/RolePlaying/components/hidden_states_{TYPE}/{MODEL}")
ANS_DIR.mkdir(parents=True, exist_ok=True)
HS_DIR.mkdir(parents=True,  exist_ok=True)

# ───────────────────── Helper functions ─────────────────────────

def option_token_ids(vc: VicundaModel):
    ids = []
    for opt in LABELS:
        tok = vc.tokenizer(opt, add_special_tokens=False).input_ids
        if len(tok) != 1:
            raise ValueError(f"Option {opt} maps to {tok}, expected single token")
        ids.append(tok[0])
    return ids


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
    vc = VicundaModel(model_path=MODEL_DIR)
    vc.model.eval()
    opt_ids = option_token_ids(vc)

    for task in TASKS:
        print(f"\n=== {task} ===")
        data_path = MMLU_DIR / f"{task}.json"
        if not data_path.exists():
            print("[Skip]", data_path, "not found")
            continue

        samples = ga.load_json(data_path)
        roles   = ga.make_characters(task, TYPE)
        role_stats = {r: {"correct":0,"E_count":0,"invalid":0,"total":0} for r in roles}
        # store hidden states per role
        hs_store: Dict[str, list[np.ndarray]] = {r: [] for r in roles}

        with torch.no_grad():
            for sample in tqdm(samples, desc=task):
                ctx       = sample["text"]
                true_idx  = sample["label"]
                if not 0 <= true_idx < len(LABELS):
                    continue
                true_label = LABELS[true_idx]

                for role in roles:
                    prompt = vc.template.format(character=role, context=ctx)
                    if SAVE:
                        logits, hidden = vc.get_logits([prompt], return_hidden=SAVE)
                        last_hs = [lay[0, -1].cpu().numpy() for lay in hidden]  # list(len_layers, hidden_size)
                        # accumulate hidden states
                        hs_store[role].append(np.stack(last_hs, axis=0))  # (layers, hidden)
                    else:
                        logits = vc.get_logits([prompt], return_hidden=SAVE)
                    
                    logits = logits[0, -1].cpu().numpy()
    
                    # softmax over answer options
                    opt_logits = np.array([logits[i] for i in opt_ids])
                    probs      = softmax_1d(opt_logits)
                    pred_idx   = int(opt_logits.argmax())
                    pred_label = LABELS[pred_idx]
                    pred_prob  = float(probs[pred_idx])

                    # attach answer+prob to sample
                    sample[rkey(role, "answer")] = pred_label
                    sample[rkey(role, "prob")]   = pred_prob

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
        ans_file = ANS_DIR / f"{task}_{SIZE}_answer.json"
        dump_json({"data": samples, "accuracy": accuracy}, ans_file)
        print("[Saved answers]", ans_file)

        # save hidden states per role
        if SAVE:
            for role, arr_list in hs_store.items():
                if not arr_list:
                    continue
                hs_np = np.stack(arr_list, axis=0)  # (n_samples, layers, hidden)
                safe_role = role.replace(" ", "_").replace("-", "_")
                hs_file = HS_DIR / f"{safe_role}_{task}_{SIZE}.npy"
                np.save(hs_file, hs_np)
                print("    [Saved HS]", hs_file)
        
    print("\n✅  All tasks finished.")

if __name__ == "__main__":
    main()