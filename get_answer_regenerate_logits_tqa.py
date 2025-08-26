#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 15:11:29 2025

@author: paveenhuang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TruthfulQA (multiple_choice) with neuron editing → logits-based answer selection.
- Loads exported TruthfulQA JSON (MC1 or MC2)
- For each (alpha, start, end) config, applies diff matrices to hidden states
- Per-question dynamically builds label set from choices (A.. up to present)
- Reads last-token logits to select answers; MC2 以“命中任一正标签”为正确
- Saves full predictions JSON 和 per-role CSV summary

Example:
python run_tqa_edit.py \
  --mode mc2 \
  --tqa_dir /data2/paveen/RolePlaying/components/truthfulqa \
  --model llama3 --model_dir meta-llama/Llama-3.1-8B-Instruct \
  --hs llama3 --size 8B --type non \
  --configs 4-16-22 1-1-29 \
  --mask_type nmd --percentage 0.5 \
  --ans_file tqa_edit_answers \
  --suite default --tail_len 1 --use_chat
"""

import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from tqdm import tqdm
import argparse

from llms import VicundaModel
from template import select_templates_pro
import utils


LETTER = [chr(ord("A") + i) for i in range(26)]  # A..Z


# ───────────────────── Helper Functions ─────────────────────────

def load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"] if isinstance(data, dict) and "data" in data else data

def labels_for_sample(sample: Dict[str, Any]) -> List[str]:
    K = max(1, min(len(sample.get("choices", [])), len(LETTER)))
    return LETTER[:K]

def gold_indices_for_sample(sample: Dict[str, Any]) -> List[int]:
    """
    Get gold indices for a sample.
    """
    gi = sample.get("gold_indices")
    if gi and isinstance(gi, list) and len(gi) > 0:
        return [int(x) for x in gi]
    labels = sample.get("labels", [])
    pos = [i for i, v in enumerate(labels) if int(v) == 1]
    return pos if pos else [0]

def record_last_template(roles: List[str], templates: dict) -> dict:
    return {"roles": roles, "templates": {k: v for k, v in templates.items() if isinstance(v, str)}, "labels": templates.get("labels", [])}

def remove_honest(templates: dict) -> dict:
    """
    Remove "honest" description in (non-) expert prompt
    """
    new_templates = {}
    for k, v in templates.items():
        if isinstance(v, str):
            new_templates[k] = v.replace(" honest", "")
        else:
            new_templates[k] = v
    return new_templates


# ───────────────────── Core Runner (one α/start/end) ─────────────────────

def run_tqa_with_editing(
    vc: VicundaModel,
    samples: List[Dict[str, Any]],
    diff_mtx: np.ndarray,
    suite: str,
    use_E: bool,
    mode: str,
    use_chat: bool,
    tail_len: int,
    role_type: str,
):
    # roles
    task_name = samples[0].get("task", "TruthfulQA")
    roles = utils.make_characters(task_name.replace(" ", "_"), role_type)
    stats = {r: {"correct": 0, "E_count": 0, "invalid": 0, "total": 0} for r in roles}

    updated = []
    last_templates = None
    refusal_label_used = None

    with torch.no_grad():
        for sample in tqdm(samples, desc=task_name):
            ctx = sample.get("text", "")

            # labels and template
            LABELS = labels_for_sample(sample)
            templates = select_templates_pro(suite=suite, labels=LABELS, use_E=use_E)
            templates = remove_honest(templates)
            
            refusal_label = templates.get("refusal_label", None)
            refusal_label_used = refusal_label  
            last_templates = templates

            # token ids
            opt_ids = utils.option_token_ids(vc, LABELS)

            # GT
            gold_indices = gold_indices_for_sample(sample, mode)

            item_out = dict(sample)
            for role in roles:
                prompt = utils.construct_prompt(vc, templates, ctx, role, use_chat)

                # logits
                raw_logits = vc.regenerate_logits([prompt], diff_mtx, tail_len=tail_len)[0]
                opt_logits = np.array([raw_logits[i] for i in opt_ids])
                probs = utils.softmax_1d(opt_logits)

                pred_idx = int(opt_logits.argmax())
                pred_label = LABELS[pred_idx]
                pred_prob = float(probs[pred_idx])

                key = role.replace(" ", "_")
                item_out[f"answer_{key}"] = pred_label
                item_out[f"prob_{key}"] = pred_prob
                item_out[f"softmax_{key}"] = [float(x) for x in probs]
                item_out[f"logits_{key}"] = [float(x) for x in opt_logits]

                st = stats[role]
                st["total"] += 1
                if pred_idx in gold_indices:
                    st["correct"] += 1
                elif use_E and (refusal_label is not None) and (pred_label == refusal_label):
                    st["E_count"] += 1
                else:
                    st["invalid"] += 1

            updated.append(item_out)

    # summary
    accuracy = {}
    for role, s in stats.items():
        acc = (s["correct"] / s["total"] * 100.0) if s["total"] else 0.0
        accuracy[role] = {**s, "accuracy_percentage": round(acc, 2)}
        print(f"{role:<25} acc={acc:5.2f}%  (correct {s['correct']}/{s['total']}), Refuse={s['E_count']}")

    tmp_record = record_last_template(roles, last_templates) if last_templates else {}
    return updated, accuracy, tmp_record, refusal_label_used


# ─────────────────────────── Main ───────────────────────────────

def main(args):
    # α/start/end 
    ALPHAS_START_END_PAIRS = utils.parse_configs(args.configs)
    print("ALPHAS_START_END_PAIRS:", ALPHAS_START_END_PAIRS)

    # model
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # data
    samples = load_json(TQA_PATH)
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError(f"Empty or invalid TQA JSON: {TQA_PATH}")

    # α/start/end →load diff matrix
    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        mask_suffix = "_abs" if args.abs else ""
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        mask_path = os.path.join(args.mask_dir, mask_name)
        diff_mtx = np.load(mask_path) * alpha
        TOP = max(1, int(args.percentage / 100.0 * diff_mtx.shape[1]))
        print(f"\n=== α={alpha} | layers={st}-{en} | TOP={TOP} ===")
        print("Mask:", mask_path)

        # TruthfulQA
        with torch.no_grad():
            updated_data, accuracy, tmp_record, refusal_label = run_tqa_with_editing(
                vc=vc,
                samples=samples,
                diff_mtx=diff_mtx,
                suite=args.suite,
                use_E=args.use_E,
                mode=args.mode,
                use_chat=args.use_chat,
                tail_len=args.tail_len,
                role_type=args.type,
            )

        # save JSON
        out_dir = os.path.join(args.save_root, f"{args.model}_{alpha}")
        os.makedirs(out_dir, exist_ok=True)
        task_name = samples[0].get("task", f"TruthfulQA_{args.mode.upper()}")
        out_path = os.path.join(out_dir, f"{task_name.replace(' ', '_')}_{args.size}_answers_{args.mode}_{TOP}_{st}_{en}.json")
        with open(out_path, "w", encoding="utf-8") as fw:
            json.dump({"data": updated_data, "accuracy": accuracy, "template": tmp_record}, fw, ensure_ascii=False, indent=2)
        print("Saved →", out_path)

        # save CSV
        csv_rows = []
        for role, s in accuracy.items():
            csv_rows.append({
                "model": args.model,
                "size": args.size,
                "alpha": alpha,
                "start": st,
                "end": en,
                "TOP": TOP,
                "dataset": "TruthfulQA",
                "mode": args.mode.upper(),
                "task": task_name,
                "role": role,
                "correct": s["correct"],
                "E_count": s["E_count"],
                "invalid": s["invalid"],
                "total": s["total"],
                "accuracy_percentage": s["accuracy_percentage"],
                "suite": args.suite,
                "refusal_enabled": int(bool(args.use_E)),
                "refusal_label": refusal_label if refusal_label is not None else "",
            })
        csv_path = os.path.join(out_dir, f"summary_{args.model}_{args.size}_{args.mode}_{TOP}_{st}_{en}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model","size","alpha","start","end","TOP",
                    "dataset","mode","task","role",
                    "correct","E_count","invalid","total",
                    "accuracy_percentage","suite","refusal_enabled","refusal_label"
                ]
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[Saved CSV] {csv_path}")

    print("\n✅  All configs finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TruthfulQA (MC1/MC2) with neuron editing and logits output.")
    parser.add_argument("--mode", required=True, choices=["mc1", "mc2"], help="TruthfulQA mode")
    parser.add_argument("--model", type=str, default="qwen2.5_base")
    parser.add_argument("--model_dir", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--hs", type=str, default="qwen2.5")
    parser.add_argument("--size", type=str, default="7B")
    parser.add_argument("--type", type=str, default="non")  # roles
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="*", default=["4-16-22"], help="alpha-start-end triplets, e.g., 4-16-22")
    parser.add_argument("--mask_type", type=str, default="nmd", help="Mask type: nmd or random")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--mask_dir", type=str, default="", help="Override mask directory (else auto from hs/type)")
    parser.add_argument("--ans_file", type=str, default="tqa_edit_answers")
    parser.add_argument("--use_E", action="store_true", help="Append refusal option to label set (rarely needed for TQA)")
    parser.add_argument("--use_chat", action="store_true", help="Use tokenizer.apply_chat_template for prompts")
    parser.add_argument("--tail_len", type=int, default=1, help="Number of last tokens to apply diff")
    parser.add_argument("--suite", type=str, default="default", choices=["default", "vanilla"], help="Prompt suite")

    args = parser.parse_args()
    
    
    # Prepare directories
    TQA_DIR = Path("/data2/paveen/RolePlaying/components/truthfulqa/")
    ANS_DIR = Path(f"/data2/paveen/RolePlaying/components/{args.ans_file}/")
    ANS_DIR.mkdir(parents=True, exist_ok=True)
    
    MASK_DIR = f"/data2/paveen/RolePlaying/components/mask/{args.hs}_{args.type}_logits"
    SAVE_ROOT = f"/data2/paveen/RolePlaying/components/{args.ans_file}"
    
    if args.mode == "mc1":
        TQA_PATH = TQA_DIR / "truthfulqa_mc1_validation.json"
    else:
        TQA_PATH = TQA_DIR / "truthfulqa_mc2_validation.json"

    
    os.makedirs(SAVE_ROOT, exist_ok=True)

    print("Model:", args.model)
    print("Import model from:", args.model_dir)
    print("HS:", args.hs)
    print("Mask dir:", args.mask_dir)
    print("Save root:", args.save_root)

    main(args)