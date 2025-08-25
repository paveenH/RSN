#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TruthfulQA runner (MC1 / MC2) with VicundaModel + neuron editing → logits-based option selection.

- Reuses utils.py: load_json, make_characters, construct_prompt, option_token_ids,
  parse_configs, softmax_1d, dump_json, record_template
- Input JSON (your exported truthfulqa_mc{1,2}_validation.json) contains:
  {
    "task": "TruthfulQA MC1" | "TruthfulQA MC2",
    "text": "question + enumerated options 'A) ...'",
    "choices": [...],            # K <= 10
    "labels": [0/1,...],         # MC1: single 1; MC2: multiple 1s
    "gold_indices": [idx,...]    # indices where labels == 1
  }

- Matches the MMLU-Pro style:
  * Supports role sets (utils.make_characters / --type)
  * For each sample, extracts logits of A..J from the last token
  * Predicts argmax; correctness = pred_idx ∈ gold_indices
  * Outputs per-role stats and per-sample details

Usage:
python run_truthfulqa.py \
  --mode mc2 \
  --json ./components/truthfulqa/truthfulqa_mc2_validation.json \
  --model qwen2.5_base \
  --model_dir Qwen/Qwen2.5-7B \
  --hs qwen2.5 \
  --size 7B \
  --type non \
  --mask_dir /data2/paveen/RolePlaying/components/mask/qwen2.5_non_logits \
  --configs 4-16-22 1-1-29 \
  --percentage 0.5 \
  --ans_root /data2/paveen/RolePlaying/components/answer_tqa
"""

import os
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from tqdm import tqdm

from llms import VicundaModel
from template import select_templates_pro
import utils  

LETTER10 = [chr(ord("A") + i) for i in range(10)]  # A..J

# ───────────────────── Core ─────────────────────


def run_truthfulqa_role(
    vc: VicundaModel,
    role: str,
    templates: Dict[str, str],
    samples: List[Dict[str, Any]],
    diff_mtx: np.ndarray,
    use_chat: bool,
    tail_len: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Run TruthfulQA (MC1/MC2) for one role.
    Returns: updated samples (with answer_* fields) and role statistics.
    """
    # Candidate token ids: letters A..J
    opt_ids_all = utils.option_token_ids(vc, templates["labels"])

    stats = {"correct": 0, "invalid": 0, "total": 0}
    role_key = role.replace(" ", "_")
    
    

    for i, sample in enumerate(tqdm(samples, desc=f"Role={role}")):
        text = sample["text"]
        gold_indices = set(int(g) for g in sample.get("gold_indices", []))
        K = min(len(sample.get("choices", [])), len(LETTER10))
        if K == 0:
            continue

        opt_ids = opt_ids_all[:K]
        prompt = utils.construct_prompt(vc, templates, text, role, use_chat)

        with torch.no_grad():
            raw_logits = vc.regenerate_logits([prompt], diff_mtx, tail_len=tail_len)[0]  # vocab logits for last token

        opt_logits = np.array([raw_logits[i] for i in opt_ids], dtype=np.float32)
        soft = utils.softmax_1d(opt_logits)

        pred_local = int(opt_logits.argmax())  # 0..K-1
        pred_letter = templates["labels"][pred_local]
        pred_prob = float(soft[pred_local])

        # Correct if predicted index is among gold indices (MC1/MC2 unified rule)
        is_correct = pred_local in gold_indices

        # Write back into sample
        sample[f"answer_{role_key}"]   = pred_letter
        sample[f"prob_{role_key}"]     = pred_prob
        sample[f"softmax_{role_key}"]  = [float(x) for x in soft]
        sample[f"logits_{role_key}"]   = [float(x) for x in opt_logits]
        sample[f"correct_{role_key}"]  = bool(is_correct)

        stats["total"]   += 1
        stats["correct"] += int(is_correct)
        if not is_correct:
            stats["invalid"] += 1

    acc = (stats["correct"] / stats["total"] * 100) if stats["total"] else 0.0
    stats["accuracy_percentage"] = round(acc, 2)
    return samples, stats

def run_truthfulqa(
    vc: VicundaModel,
    samples: List[Dict[str, Any]],
    diff_mtx: np.ndarray,
    tqa_mode: str,
    use_chat: bool,
    tail_len: int,
    out_dir: str,
    model_tag: str,
    size_tag: str,
    cfg_tag: str,
    type_tag: str,
):
    # Dynamic labels: A..J truncated to K (per question); prebuild full A..J token ids
    templates = select_templates_pro(suite=suite, labels=base_labels, use_E=use_E)
    templates = utils.record_template(roles, templates)
    # Roles
    task_name = (samples[0].get("task") or f"TruthfulQA {tqa_mode.upper()}").replace(" ", "_")
    roles = utils.make_characters(task_name, type_tag)
    if not roles:
        roles = ["neutral"]  # fallback

    # Record templates (compatibility with original script)
    tmp_record = utils.record_template(roles, templates)

    # Run each role
    all_stats = {}
    for role in roles:
        samples, stats = run_truthfulqa_role(
            vc=vc,
            role=role,
            templates=templates,
            samples=samples,
            diff_mtx=diff_mtx,
            use_chat=use_chat,
            tail_len=tail_len,
        )
        all_stats[role] = stats
        print(f"{role:<25} acc={stats['accuracy_percentage']:5.2f}%  "
              f"(correct {stats['correct']}/{stats['total']})")

    # Save JSON
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"tqa_{tqa_mode}_{model_tag}_{size_tag}_{cfg_tag}.json")
    utils.dump_json({"data": samples, "accuracy": all_stats, "template": tmp_record}, Path(json_path))
    print("Saved →", json_path)

    # Save CSV summary
    csv_rows = []
    for role, s in all_stats.items():
        csv_rows.append({
            "mode": tqa_mode,
            "model": model_tag,
            "size": size_tag,
            "cfg": cfg_tag,
            "role": role,
            "correct": s["correct"],
            "invalid": s["invalid"],
            "total": s["total"],
            "accuracy_percentage": s["accuracy_percentage"],
        })
    csv_path = os.path.join(out_dir, f"summary_tqa_{tqa_mode}_{model_tag}_{size_tag}_{cfg_tag}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mode","model","size","cfg","role","correct","invalid","total","accuracy_percentage"]
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[Saved CSV] {csv_path}")

# ───────────────────── Main ─────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Vicunda model (TruthfulQA MC1/MC2) with neuron editing and logits output.")
    parser.add_argument("--mode", type=str, choices=["mc1","mc2"], required=True, help="TruthfulQA mode")
    parser.add_argument("--json", type=str, required=True, help="Path to exported TruthfulQA JSON")
    parser.add_argument("--model", type=str, default="qwen2.5_base")
    parser.add_argument("--model_dir", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--hs", type=str, default="qwen2.5")
    parser.add_argument("--size", type=str, default="7B")
    parser.add_argument("--type", type=str, default="non", help="role type for utils.make_characters")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="*", default=["4-16-22"], help="alpha-start-end triplets, e.g. 4-16-22 or neg1-11-20")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing mask .npy files")
    parser.add_argument("--mask_type", type=str, default="nmd")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--ans_root", type=str, default="./answer_tqa")
    parser.add_argument("--tail_len", type=int, default=1)
    parser.add_argument("--use_chat", action="store_true")
    args = parser.parse_args()

    print("Model:", args.model, "from", args.model_dir)
    print("HS:", args.hs, "| mask_type:", args.mask_type)

    # Load model
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # Load samples
    samples = load_tqa_json(args.json)
    assert isinstance(samples, list) and len(samples) > 0, "Empty or invalid TruthfulQA JSON."

    # Loop configs (alpha, start-end)
    cfgs = utils.parse_configs(args.configs)
    for alpha, (st, en) in cfgs:
        suffix = "_abs" if args.abs else ""
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{suffix}.npy"
        mask_path = os.path.join(args.mask_dir, mask_name)
        diff_mtx = np.load(mask_path) * alpha

        TOP = max(1, int(args.percentage / 100.0 * diff_mtx.shape[1]))
        cfg_tag = f"TOP{TOP}_{st}_{en}_a{alpha}"

        print(f"\n=== TQA {args.mode.upper()} | α={alpha} | layers={st}-{en} | TOP={TOP} ===")
        out_dir = os.path.join(args.ans_root, f"{args.model}_{alpha}")
        os.makedirs(out_dir, exist_ok=True)

        run_truthfulqa(
            vc=vc,
            samples=[dict(s) for s in samples],  # copy per config to avoid overwriting
            diff_mtx=diff_mtx,
            tqa_mode=args.mode,
            use_chat=args.use_chat,
            tail_len=args.tail_len,
            out_dir=out_dir,
            model_tag=args.model,
            size_tag=args.size,
            cfg_tag=cfg_tag,
            type_tag=args.type,
        )

    print("\n✅  All runs finished.")

if __name__ == "__main__":
    main()