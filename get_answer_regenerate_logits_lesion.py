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
    st: int,
    en: int,
):
    """Run one task with RSN-lesion, returning updated data + accuracy."""

    # template
    templates = select_templates(args.suite, args.use_E)
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
                start=st,
                end=en,
            )[
                0
            ]  # (V,)

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

    LAYER_RANGES = [pair[1] for pair in utils.parse_configs(args.layers)]
    print("LAYER_RANGES:", LAYER_RANGES)

    # Load model
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # Load RSN indices (per-layer)
    rsn_path = os.path.join(RSN_DIR, args.rsn_file)
    rsn_indices_per_layer = utils.load_json_or_npy(rsn_path)
    # ensure list[list[int]]
    rsn_indices_per_layer = [list(map(int, x)) for x in rsn_indices_per_layer]

    for st, en in LAYER_RANGES:
        print(f"\n=== RSN-Lesion | layers={st}-{en} ===")

        for task in TASKS:

            updated_data, accuracy, tmp_record = run_task_lesion(
                vc=vc,
                task=task,
                rsn_indices_per_layer=rsn_indices_per_layer,
                st=st,
                en=en,
            )

            # save json
            out_dir = os.path.join(SAVE_ROOT, f"lesion_{st}_{en}")
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(out_dir, f"{task}_{args.size}_lesion_{st}_{en}.json")
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump(
                    {
                        "data": updated_data,
                        "accuracy": accuracy,
                        "template": tmp_record,
                    },
                    fw,
                    ensure_ascii=False,
                    indent=2,
                )

            print("Saved →", out_path)

    print("\n All RSN-lesion tasks finished.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Vicunda model with RSN-lesion and logits output.")

    parser.add_argument("--model", type=str, default="qwen2.5_base")
    parser.add_argument("--model_dir", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--hs", type=str, default="qwen2.5")
    parser.add_argument("--size", type=str, default="7B")
    parser.add_argument("--type", type=str, default="non")
    parser.add_argument("--layers", nargs="*", default=["0-32"], help="List of start-end layer ranges, e.g. 16-22, 0-32")
    parser.add_argument("--rsn_file", type=str, default="rsn_indices.npy", help="File containing per-layer RSN indices")
    parser.add_argument("--ans_file", type=str, default="answer_rsn_lesion")
    parser.add_argument("--use_E", action="store_true")
    parser.add_argument("--use_chat", action="store_true")
    parser.add_argument("--suite", type=str, default="default", choices=["default", "vanilla"])
    parser.add_argument("--data", type=str, default="data1", choices=["data1", "data2"])

    args = parser.parse_args()

    print("Model:", args.model)
    print("Import model from:", args.model_dir)
    print("HS:", args.hs)

    # paths
    RSN_DIR = f"/{args.data}/paveen/RolePlaying/components/rsn/{args.hs}_{args.type}_logits"
    MMLU_DIR = f"/{args.data}/paveen/RolePlaying/components/mmlu"
    SAVE_ROOT = f"/{args.data}/paveen/RolePlaying/components/{args.model}/{args.ans_file}"

    os.makedirs(SAVE_ROOT, exist_ok=True)
    main()
