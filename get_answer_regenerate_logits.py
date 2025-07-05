#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for VicundaModel with neuron editing -> logits-based answer selection.
Loads the model once, applies diff matrices to hidden states, and for each prompt
directly reads out the last‐token logits to pick A/B/C/D/E.
"""

import os
import json
import numpy as np
from tqdm import tqdm
import get_answer as ga
import get_answer_logits as gal
from llms import VicundaModel
import torch

# ─────────────────────── Configuration ──────────────────────────

TASKS = ga.TASKS
LABELS = ["A", "B", "C", "D", "E"]

# ───────────────────── Helper Functions ─────────────────────────

def build_char_diff(alpha: int, start: int, end: int):
    diff_char = np.load(f"{HS_MEAN}/diff_mean_{SIZE}.npy")
    diff_none = np.load(f"{HS_MEAN}/none_diff_mean_{SIZE}.npy")
    diff = (diff_char - diff_none).squeeze(0).squeeze(0)  # (layers, hidden)

    for layer in range(diff.shape[0]):
        if start <= layer < end:
            layer_vec = diff[layer]
            idxs = np.argsort(np.abs(layer_vec))[-TOP:]
            mask = np.zeros_like(layer_vec, dtype=bool)
            mask[idxs] = True
            diff[layer] = layer_vec * mask
        else:
            diff[layer] = 0

    return diff[1:] * alpha

def run_task(
    vc: VicundaModel,
    template: str,
    task: str,
    diff_mtx: np.ndarray,
    opt_ids: list[int],
):
    """Run one task with a fixed diff_mtx, returning updated data + accuracy."""
    # load data
    data_path = os.path.join(MMLU_DIR, f"{task}.json")
    data = ga.load_json(data_path)
    roles = ga.make_characters(task, TYPE)

    # stats accumulator
    stats = { r: {"correct":0, "E_count":0, "invalid":0, "total":0} for r in roles }

    for sample in tqdm(data, desc=task):
        ctx = sample.get("text", "")
        true_idx = sample.get("label", -1)
        if not (0 <= true_idx < len(LABELS)):
            continue
        true_lab = LABELS[true_idx]

        for role in roles:
            prompt = template.format(character=role, context=ctx)
            # get raw logits after hooking in diff
            raw_logits = vc.regenerate_logits([prompt], diff_mtx)[0]
            # pick among options A–E
            opt_logits = np.array([raw_logits[i] for i in opt_ids])
            
            exp = np.exp(opt_logits - opt_logits.max())
            soft = exp / exp.sum()
            
            pred_idx = int(opt_logits.argmax())
            pred_lab = LABELS[pred_idx]
            pred_prb = float(soft[pred_idx])

            # write back answer
            key_ans  = f"answer_{role.replace(' ', '_')}"
            key_prob = f"prob_{role.replace(' ', '_')}"
            sample[key_ans]  = pred_lab
            sample[key_prob] = pred_prb

            # update stats
            st = stats[role]
            st["total"] += 1
            if pred_lab == true_lab:
                st["correct"] += 1
            elif pred_lab == "E":
                st["E_count"] += 1
            else:
                st["invalid"] += 1

    # accuracy summary
    accuracy = {}
    for role, s in stats.items():
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0
        accuracy[role] = {**s, "accuracy_percentage": round(pct, 2)}
        print(f"{role:<25} acc={pct:5.2f}%  (correct {s['correct']}/{s['total']}), E={s['E_count']}")

    return data, accuracy

# ─────────────────────────── Main ───────────────────────────────

def main():
    vc = VicundaModel(model_path=MODEL_DIR)
    vc.model.eval()
    opt_ids = gal.option_token_ids(vc)
    template = vc.template

    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        diff_mtx = build_char_diff(alpha, st, en)
        for task in TASKS:
            print(f"\n=== {task} | α={alpha} | layers={st}-{en}| TOP={TOP} ===")
            with torch.no_grad():  
                updated_data, accuracy = run_task(vc, template, task, diff_mtx, opt_ids)

            # save JSON
            out_dir = os.path.join(SAVE_ROOT, f"{MODEL}_{alpha}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{task}_{SIZE}_answers_{TOP}_{st}_{en}.json")
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump({"data": updated_data, "accuracy": accuracy}, fw, ensure_ascii=False, indent=2)
            print("Saved →", out_path)

    print("\n✅  All tasks finished.")


if __name__ == "__main__":
    
    ###################################################
    MODEL = "llama3_base"
    HS = "llama3_base"
    SIZE = "3B"
    TYPE = "non"
    MODEL_DIR = "meta-llama/Llama-3.2-3B"
    
    TOP = 15
    ALPHAS_START_END_PAIRS = [[4, (7, 17)],[1, (1, 29)]]
    
    print("Model: ", MODEL)
    print("HS: ", HS)
    print ("Import model from ", MODEL_DIR)
    
    ANS = f"answer_modified_logits_{TYPE}_val"
    MMLU_DIR = "/data2/paveen/RolePlaying/components/mmlu"
    SAVE_ROOT = f"/data2/paveen/RolePlaying/components/{ANS}"
    os.makedirs(SAVE_ROOT, exist_ok=True)
    if "logits" in ANS:
        HS_MEAN = f"/data2/paveen/RolePlaying/components/hidden_states_mean/{HS}_{TYPE}_logits"
    else:
        HS_MEAN = f"/data2/paveen/RolePlaying/components/hidden_states_mean/{HS}_{TYPE}"
    main()
    
    ###################################################
    MODEL = "llama3"
    HS = "llama3"
    SIZE = "3B"
    TYPE = "non"
    MODEL_DIR = "meta-llama/Llama-3.2-3B-Instruct"
    
    TOP = 15
    ALPHAS_START_END_PAIRS = [[4, (7, 17)],[1, (1, 29)]]
    
    print("Model: ", MODEL)
    print("HS: ", HS)
    print ("Import model from ", MODEL_DIR)
    
    ANS = f"answer_modified_logits_{TYPE}_val"
    MMLU_DIR = "/data2/paveen/RolePlaying/components/mmlu"
    SAVE_ROOT = f"/data2/paveen/RolePlaying/components/{ANS}"
    os.makedirs(SAVE_ROOT, exist_ok=True)
    if "logits" in ANS:
        HS_MEAN = f"/data2/paveen/RolePlaying/components/hidden_states_mean/{HS}_{TYPE}_logits"
    else:
        HS_MEAN = f"/data2/paveen/RolePlaying/components/hidden_states_mean/{HS}_{TYPE}"
    main()
    
    ###################################################
    MODEL = "hermes"
    HS = "hermes"
    SIZE = "3B"
    TYPE = "non"
    MODEL_DIR = "NousResearch/Hermes-3-Llama-3.2-3B"
    
    TOP = 15
    ALPHAS_START_END_PAIRS = [[4, (7, 17)],[1, (1, 29)]]
    
    print("Model: ", MODEL)
    print("HS: ", HS)
    print ("Import model from ", MODEL_DIR)
    
    ANS = f"answer_modified_logits_{TYPE}_val"
    MMLU_DIR = "/data2/paveen/RolePlaying/components/mmlu"
    SAVE_ROOT = f"/data2/paveen/RolePlaying/components/{ANS}"
    os.makedirs(SAVE_ROOT, exist_ok=True)
    if "logits" in ANS:
        HS_MEAN = f"/data2/paveen/RolePlaying/components/hidden_states_mean/{HS}_{TYPE}_logits"
    else:
        HS_MEAN = f"/data2/paveen/RolePlaying/components/hidden_states_mean/{HS}_{TYPE}"
    main()
