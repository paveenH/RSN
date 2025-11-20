#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for VicundaModel with RSN lesion -> logits-based answer selection.
Loads the model once, zeros out RSN neurons per layer, and for each prompt
reads out last-token logits to pick A/B/C/D/E.
"""

import os
import json
import numpy as np
from tqdm import tqdm
import argparse

from llms import VicundaModel
from detection.task_list import TASKS
from template import select_templates
import utils


# ───────────────────── Helper Function ─────────────────────────


def run_task_lesion(
    vc: VicundaModel,
    task: str,
    rsn_indices_per_layer: list[list[int]],
):
    """Run one task with RSN-lesion, returning updated data + accuracy."""

    # template
    templates = select_templates(args.suite, args.E)
    LABELS = templates["labels"]
    opt_ids = utils.option_token_ids(vc, LABELS)

    # load data
    data_path = os.path.join(MMLU_DIR, f"{task}.json")
    data = utils.load_json(data_path)
    roles = utils.make_characters(task, args.type)

    tmp_record = utils.record_template(roles, templates)

    # stats
    stats = {r: {"correct": 0, "E_count": 0, "invalid": 0, "total": 0} for r in roles}

    for sample in tqdm(data, desc=f"{task}"):
        ctx = sample.get("text", "")
        true_idx = sample.get("label", -1)
        true_lab = LABELS[true_idx]

        for role in roles:

            prompt = utils.construct_prompt(vc, templates, ctx, role, args.use_chat)

            # RSN lesion: knock-out neurons
            raw_logits = vc.regenerate_rsn_lesion(
                [prompt],
                rsn_indices_per_layer=rsn_indices_per_layer,
            )[0]
    

            # restrict to options A–E
            opt_logits = np.array([raw_logits[i] for i in opt_ids])

            exp = np.exp(opt_logits - opt_logits.max())
            soft = exp / exp.sum()

            pred_idx = int(opt_logits.argmax())
            pred_lab = LABELS[pred_idx]
            pred_prb = float(soft[pred_idx])

            role_key = role.replace(" ", "_")
            sample[f"answer_{role_key}"] = pred_lab
            sample[f"prob_{role_key}"] = pred_prb
            sample[f"softmax_{role_key}"] = [float(x) for x in soft]
            sample[f"logits_{role_key}"] = [float(x) for x in opt_logits]

            # update stats
            s = stats[role]
            s["total"] += 1
            if pred_lab == true_lab:
                s["correct"] += 1
            elif pred_lab == "E":
                s["E_count"] += 1
            else:
                s["invalid"] += 1

    # summary
    accuracy = {}
    for role, s in stats.items():
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0
        accuracy[role] = {**s, "accuracy_percentage": round(pct, 2)}
        print(f"{role:<25} acc={pct:5.2f}%  (correct {s['correct']}/{s['total']}), E={s['E_count']}")

    return data, accuracy, tmp_record


# ─────────────────────────── Main ───────────────────────────────


def main():

    CONFIGS = utils.parse_configs(args.configs)
    LAYER_RANGES = [cfg[1] for cfg in CONFIGS]   # extract (start,end)
    print("LAYER_RANGES:", LAYER_RANGES)

    # Load model
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    for _, (st, en) in CONFIGS:
        # ====== Load mask ======
        mask_suffix = "_abs" if args.abs else ""
        mask_name = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        mask_path = os.path.join(MASK_DIR, mask_name)

        print("\nLoading mask:", mask_path)
        diff_mtx = np.load(mask_path)  # shape = (L-1, H), already removed embedding

        num_layers, H = diff_mtx.shape

        # ====== Compute TOP-k ======
        TOP = max(1, int(args.percentage / 100 * H))
        print(f"Using TOP={TOP} neurons per layer")

        # ====== Compute lesion indices ONLY in [st,en), empty for others ======
        rsn_indices_per_layer = []
        for layer in range(num_layers):

            row = diff_mtx[layer]
            if np.all(row == 0):
                rsn_indices_per_layer.append([])
            else:
                idx = np.argsort(-np.abs(row))[:TOP]
                rsn_indices_per_layer.append(list(map(int, idx)))

        print(f"Prepared RSN indices for {len(rsn_indices_per_layer)} layers")
        print(f"Lesion only layers: {st}–{en-1}")

        # ====== Run tasks ======
        print(f"\n=== RSN-Lesion | layers={st}-{en} | TOP={TOP} ===")

        for task in TASKS:

            updated_data, accuracy, tmp_record = run_task_lesion(
                vc=vc,
                task=task,
                rsn_indices_per_layer=rsn_indices_per_layer,
            )

            # save
            out_dir = os.path.join(SAVE_ROOT, f"lesion_{TOP}_{st}_{en}")
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(out_dir, f"{task}_{args.size}_answers_{TOP}_{st}_{en}.json")
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump(
                    {"data": updated_data, "accuracy": accuracy, "template": tmp_record},
                    fw,
                    ensure_ascii=False,
                    indent=2,
                )

            print("Saved →", out_path)

    print("\n All RSN-lesion tasks finished.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Vicunda model with RSN-lesion and logits output.")

    # model
    parser.add_argument("--model", type=str, default="qwen2.5_base")
    parser.add_argument("--model_dir", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--hs", type=str, default="qwen2.5")
    parser.add_argument("--size", type=str, default="7B")
    parser.add_argument("--type", type=str, default="non")
    # key arguments for RSN lesion 
    parser.add_argument("--percentage", type=float, default=0.5, help="Top-k percentage used to select RSNs per layer from mask")
    parser.add_argument("--configs", nargs="*", default=["1-10-21"], help="Same format as diff: alpha-start-end, but alpha is ignored. e.g. 1-16-22")
    parser.add_argument("--mask_type", type=str, default="nmd", help="Mask type (same as diff version)")
    parser.add_argument("--abs", action="store_true", help="Use _abs mask suffix (same as diff version)")
    # saving + template
    parser.add_argument("--ans_file", type=str, default="answer_rsn_lesion")
    parser.add_argument("--E", action="store_true")
    parser.add_argument("--use_chat", action="store_true")
    parser.add_argument("--suite", type=str, default="default", choices=["default", "vanilla"])
    # data source in server
    parser.add_argument("--data", type=str, default="data1", choices=["data1", "data2"])

    args = parser.parse_args()

    print("Model:", args.model)
    print("Import model from:", args.model_dir)
    print("HS:", args.hs)

    # paths (same structure as diff)
    MASK_DIR = f"/{args.data}/paveen/RolePlaying/components/mask/{args.hs}_{args.type}_logits"
    MMLU_DIR = f"/{args.data}/paveen/RolePlaying/components/mmlu"
    SAVE_ROOT = f"/{args.data}/paveen/RolePlaying/components/{args.model}/{args.ans_file}"

    os.makedirs(SAVE_ROOT, exist_ok=True)

    main()
