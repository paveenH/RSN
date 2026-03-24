#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Aug 22 17:27:15 2025

Extract highest-logit answer + probability **and** save last-token hidden states
for every role on every task — switched to MMLU-Pro combined JSON.

@author: paveenhuang

"""

from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
import argparse
from tqdm import tqdm
import csv
import h5py

from llms import VicundaModel
from template import select_templates_pro
import utils


def main():
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # Load mmlupro json file
    all_samples: List[dict] = utils.load_json(DATA_DIR)

    # group by "task"
    tasks = sorted({s["task"] for s in all_samples})
    print(f"Found {len(tasks)} tasks in MMLU-Pro JSON.")

    rows = []  # collect stats for CSV
    for task in tasks:
        print(f"\n=== {task} ===")
        samples = [s for s in all_samples if s["task"] == task]
        if not samples:
            raise ValueError(f"empty task: {task}")

        # role list
        custom_roles = None
        if args.roles:
            custom_roles = [r.strip() for r in args.roles.split(",")]
        roles = utils.make_characters(task.replace(" ", "_"), custom_roles)
        role_stats = {r: {"correct": 0, "E_count": 0, "invalid": 0, "total": 0} for r in roles}

        # Initialize HDF5 files for streaming save if needed
        h5_files: Dict[str, h5py.File] = {}
        h5_datasets: Dict[str, h5py.Dataset] = {}
        if args.save:
            for role in roles:
                safe_role = role.replace(" ", "_").replace("-", "_")
                hs_file = HS_DIR / f"{safe_role}_{task.replace(' ', '_')}_{args.size}.h5"
                h5_files[role] = h5py.File(hs_file, 'w')
            sample_count = len(samples)

        with torch.no_grad():
            for sample_idx, sample in enumerate(tqdm(samples, desc=task)):
                # labels & template
                K = int(sample.get("num_options"))
                labels = [chr(ord("A") + i) for i in range(K)]
                templates = select_templates_pro(suite=args.suite, labels=labels, use_E=args.use_E, cot = args.cot)
                LABELS = templates["labels"]
                refusal_label = templates.get("refusal_label")

                if not args.use_E:
                    templates = utils.remove_honest(templates)

                # get ids of options
                opt_ids = utils.option_token_ids(vc, LABELS)

                ctx = sample["text"]
                true_idx = int(sample["label"])
                true_label = LABELS[true_idx]

                for role in roles:
                    prompt = utils.construct_prompt(vc, templates, ctx, role, False)
                    logits = vc.get_logits([prompt], return_hidden=args.save)

                    # Extract hidden states if saving
                    if args.save and isinstance(logits, tuple):
                        logits, hidden = logits
                        last_hs = [lay[0, -1].half().cpu().numpy() for lay in hidden]  # list(len_layers, hidden_size), FP16
                        # Stream save to HDF5
                        hs_array = np.stack(last_hs, axis=0)  # (layers, hidden)
                        if role not in h5_datasets:
                            # Create dataset on first write
                            hs_shape = hs_array.shape
                            h5_datasets[role] = h5_files[role].create_dataset(
                                'hidden_states',
                                shape=(sample_count,) + hs_shape,
                                dtype='float16',
                                chunks=(1,) + hs_shape
                            )
                        h5_datasets[role][sample_idx] = hs_array

                    logits = logits[0, -1].float().cpu().numpy()

                    # Only in k options in the task
                    opt_logits = np.array([logits[i] for i in opt_ids])
                    probs = utils.softmax_1d(opt_logits)
                    pred_idx = int(opt_logits.argmax())
                    pred_label = LABELS[pred_idx]
                    pred_prob = float(probs[pred_idx])

                    # attach answer+prob to sample
                    rk = role.replace(' ', '_')
                    sample[f"answer_{rk}"] = pred_label
                    sample[f"prob_{rk}"] = pred_prob
                    sample[f"softmax_{rk}"] = probs.tolist()
                    sample[f"logits_{rk}"] = opt_logits.tolist()

                    # statistics
                    rs = role_stats[role]
                    rs["total"] += 1
                    if pred_label == true_label:
                        rs["correct"] += 1
                    elif args.use_E and refusal_label is not None and pred_label == refusal_label:
                        rs["E_count"] += 1
                    else:
                        rs["invalid"] += 1
        
        print(labels)
        print(LABELS)
        print("refuse label ", refusal_label)
        tmp_record = utils.record_template(roles, templates)

        # summary + collect rows
        for role, s in role_stats.items():
            pct = s["correct"] / s["total"] * 100 if s["total"] else 0.0
            print(f"{role:<25} acc={pct:5.2f}%  (correct {s['correct']}/{s['total']}), Refuse={s['E_count']}")
            rows.append({
                "model": args.model,
                "size": args.size,
                "suite": args.suite,
                "refusal_enabled": int(bool(args.use_E)),
                "refusal_label": refusal_label if refusal_label is not None else "",
                "task": task,
                "role": role,
                "correct": s["correct"],
                "E_count": s["E_count"],
                "invalid": s["invalid"],
                "total": s["total"],
                "accuracy_percentage": round(pct, 2),
            })
        
        # save per-task detailed JSON
        task_dir = ANS_DIR / "orig"
        task_dir.mkdir(parents=True, exist_ok=True)
        ans_file = task_dir / f"{task.replace(' ', '_')}_{args.size}_answers.json"
        utils.dump_json({"data": samples, "template": tmp_record}, ans_file)
        print("[Saved answers]", ans_file)

        # close HDF5 files
        if args.save:
            for role in roles:
                if role in h5_files:
                    h5_files[role].close()
                    safe_role = role.replace(" ", "_").replace("-", "_")
                    hs_file = HS_DIR / f"{safe_role}_{task.replace(' ', '_')}_{args.size}.h5"
                    print("[Saved HS]", hs_file)

    # save task performance CSV
    csv_file = ANS_DIR / f"summary_{args.model}_{args.size}.csv"
    fieldnames = [
        "model","size","suite","refusal_enabled","refusal_label",
        "task","role","correct","E_count","invalid","total","accuracy_percentage"
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ Saved summary CSV to {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MMLU-Pro role-based extraction")
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--size", "-s", required=True)
    parser.add_argument("--type", required=True)
    parser.add_argument("--task_name", type=str, default=None,
                        help="Task name for hidden_states subdirectory (e.g., mmlupro, factor). "
                             "If not set, no subdirectory is created.")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--ans_file", required=True)
    parser.add_argument("--use_E", action="store_true")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--suite", type=str, default="default", choices=["default","vanilla", "action"])
    parser.add_argument("--data", type=str, default="default", choices=["data1", "data2"])
    parser.add_argument("--save", action="store_true", help="Whether to save hidden states (default saves only logits/answers)")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Base directory for data/output (e.g., /work/<user>/RolePlaying/components). "
                             "If not set, falls back to /{data}/paveen/RolePlaying/components")
    parser.add_argument("--roles", type=str, default=None,
                        help="Comma-separated list of roles. Use {task} as placeholder for task name. "
                             "E.g., 'neutral,{task} expert,non {task} expert'")
    parser.add_argument("--hs_dir", type=str, default=None,
                        help="Base directory for hidden states output (e.g., /data1/paveen/ConfSteer/HiddenStates). "
                             "If not set, falls back to {base}/hidden_states_{type}/{model}/{task}/")

    args = parser.parse_args()

    print("model: ", args.model)
    print("Loading model from:", args.model_dir)

    # Path setup
    if args.base_dir:
        BASE = Path(args.base_dir)
    else:
        BASE = Path(f"/{args.data}/paveen/RolePlaying/components")

    DATA_DIR = BASE / args.test_file
    ANS_DIR = BASE / args.model / args.ans_file
    if args.hs_dir:
        HS_DIR = Path(args.hs_dir) / args.model / args.task_name
    else:
        hs_base = BASE / f"hidden_states_{args.type}" / args.model
        HS_DIR = hs_base / args.task_name if args.task_name else hs_base
    ANS_DIR.mkdir(parents=True, exist_ok=True)
    HS_DIR.mkdir(parents=True, exist_ok=True)

    main()