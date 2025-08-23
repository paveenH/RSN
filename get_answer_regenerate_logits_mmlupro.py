#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for VicundaModel on MMLU-Pro with neuron editing → logits-based answer selection.
- Loads a single combined MMLU-Pro JSON, groups by `task`
- For each task, dynamically builds the label set (A.. up to max present) and (optionally) appends refusal
- Applies diff matrices to hidden states and reads last-token logits to select answers
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm
import argparse
from pathlib import Path

from llms import VicundaModel
from template import select_templates_pro
import utils


# ───────────────────── Helper Functions ─────────────────────────

def letters_from_max_label(samples):
    """Infer label letters from the max label index in samples (0-based)."""
    max_label = max(int(s["label"]) for s in samples)
    K = max(1, max_label + 1)
    return [chr(ord("A") + i) for i in range(K)]  # ["A", "B", ...]


def run_task_pro(
    vc: VicundaModel,
    task: str,
    samples: list,
    diff_mtx: np.ndarray,
    suite: str,
    use_E: bool,
):
    """
    Run one MMLU-Pro task (all its samples) with a fixed diff_mtx.
    Returns updated samples, accuracy dictionary, and recorded templates.
    """
    # Dynamic labels (from data, optionally append refusal label)
    base_labels = letters_from_max_label(samples)
    templates = select_templates_pro(suite=suite, labels=base_labels, use_E=use_E)
    LABELS = templates["labels"]
    refusal_label = templates.get("refusal_label", None)

    # Candidate token ids
    opt_ids = utils.option_token_ids(vc, LABELS)

    # Roles and Stats accumulator
    roles = utils.make_characters(task.replace(" ", "_"), args.type)
    stats = {r: {"correct": 0, "E_count": 0, "invalid": 0, "total": 0} for r in roles}
    tmp_record = utils.record_template(roles, templates)

    # Iterate over samples
    for sample in tqdm(samples, desc=task):
        ctx = sample.get("text", "")
        true_idx = int(sample.get("label", -1))
        if not 0 <= true_idx < len(letters_from_max_label(samples)):
            raise ValueError(
                f"[Error] Task {task} has invalid label index {true_idx} "
                f"(valid range: 0–{len(letters_from_max_label(samples))-1}). Sample: {sample}"
            )
        true_lab = LABELS[true_idx]

        for role in roles:
            prompt = utils.construct_prompt(vc, templates, ctx, role, args.use_chat)

            # Logits with editing applied
            raw_logits = vc.regenerate_logits([prompt], diff_mtx, tail_len=args.tail_len)[0]
            opt_logits = np.array([raw_logits[i] for i in opt_ids])

            exp = np.exp(opt_logits - opt_logits.max())
            soft = exp / exp.sum()

            pred_idx = int(opt_logits.argmax())
            pred_lab = LABELS[pred_idx]
            pred_prb = float(soft[pred_idx])

            role_key = role.replace(" ", "_")
            sample[f"answer_{role_key}"] = pred_lab
            sample[f"prob_{role_key}"] = pred_prb
            sample[f"softmax_{role_key}"] = [float(p) for p in soft]
            sample[f"logits_{role_key}"] = [float(l) for l in opt_logits]

            st = stats[role]
            st["total"] += 1
            if pred_lab == true_lab:
                st["correct"] += 1
            elif use_E and (pred_lab == (refusal_label if refusal_label is not None else LABELS[-1])):
                # If refusal_label is defined, use it; otherwise assume last label is refusal
                st["E_count"] += 1
            else:
                st["invalid"] += 1

    # 7) Summarize accuracy
    accuracy = {}
    for role, s in stats.items():
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0
        accuracy[role] = {**s, "accuracy_percentage": round(pct, 2)}
        print(f"{role:<25} acc={pct:5.2f}%  (correct {s['correct']}/{s['total']}), Refuse={s['E_count']}")

    return samples, accuracy, tmp_record


# ─────────────────────────── Main ───────────────────────────────

def main():

    ALPHAS_START_END_PAIRS = utils.parse_configs(args.configs)
    print("ALPHAS_START_END_PAIRS:", ALPHAS_START_END_PAIRS)

    # Load model
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # Load combined MMLU-Pro JSON
    all_samples = utils.load_json(Path(args.mmlupro_json))
    tasks = sorted({s["task"] for s in all_samples})
    print(f"Found {len(tasks)} tasks in MMLU-Pro JSON.")

    # Outer loop: each alpha / start-end pair (load corresponding mask)
    for alpha, (st, en) in ALPHAS_START_END_PAIRS:
        mask_suffix = "_abs" if args.abs else ""
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        mask_path = os.path.join(MASK_DIR, f"{mask_name}")
        diff_mtx = np.load(mask_path) * alpha  # shape typically (L-1, H) or (L, H)
        TOP = max(1, int(args.percentage / 100 * diff_mtx.shape[1]))
        print(f"\n=== α={alpha} | layers={st}-{en} | TOP={TOP} ===")
        
        # Inner loop: iterate over tasks
        for task in tasks:
            task_samples = [s for s in all_samples if s["task"] == task]
            print(f"\n--- Task: {task} ---")
            with torch.no_grad():
                updated_data, accuracy, tmp_record = run_task_pro(
                    vc=vc,
                    task=task,
                    samples=task_samples,
                    diff_mtx=diff_mtx,
                    suite=args.suite,       # "default" or "vanilla"
                    use_E=args.use_E,
                )

            # Save JSON (aligned with regenerate.py naming)
            out_dir = os.path.join(SAVE_ROOT, f"{args.model}_{alpha}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{task.replace(' ', '_')}_{args.size}_answers_{TOP}_{st}_{en}.json")
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump(
                    {"data": updated_data, "accuracy": accuracy, "template": tmp_record},
                    fw, ensure_ascii=False, indent=2
                )
            print("Saved →", out_path)

    print("\n✅  All tasks finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Vicunda model (MMLU-Pro) with neuron editing and logits output.")
    # Same arguments as original version + new args for mmlupro_json and suite
    parser.add_argument("--model", type=str, default="qwen2.5_base")
    parser.add_argument("--model_dir", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--hs", type=str, default="qwen2.5")
    parser.add_argument("--size", type=str, default="7B")
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="*", default=["4-16-22", "1-1-29"],
                        help="List of alpha-start-end triplets, e.g. 4-16-22")
    parser.add_argument("--mask_type", type=str, default="nmd", help="Mask type: nmd or random")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--ans_file", type=str, default="answer_mdf")
    parser.add_argument("--use_E", action="store_true", help="Append a refusal option to the label set")
    parser.add_argument("--use_chat", action="store_true", help="Use tokenizer.apply_chat_template for prompts")
    parser.add_argument("--tail_len", type=int, default=1, help="Number of last tokens to apply diff (default: 1)")
    parser.add_argument("--suite", type=str, default="default", choices=["default", "vanilla"], help="Prompt suite for MMLU-Pro")

    args = parser.parse_args()

    print("Model: ", args.model)
    print("Import model from ", args.model_dir)
    print("HS: ", args.hs)
    print("Mask Type:", args.mask_type)

    # Directory organization same as before
    MMLU_PRO_DIR = Path("/data2/paveen/RolePlaying/components/mmlupro/mmlupro_test.json")
    MASK_DIR = f"/data2/paveen/RolePlaying/components/mask/{args.hs}_{args.type}_logits"
    SAVE_ROOT = f"/data2/paveen/RolePlaying/components/{args.ans_file}"
    if args.abs:
        SAVE_ROOT += "_abs"
    os.makedirs(SAVE_ROOT, exist_ok=True)

    main()