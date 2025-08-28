#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run LM Evaluation Harness twice:
- Baseline (original, no editing)
- Edited (with RSN hooks via HFLMWithRSN)

Saves two JSON results for对比。
"""

import argparse
import json
from pathlib import Path

import numpy as np
from lm_eval import evaluator

from hf_rsn import HFLMWithRSN
from lm_eval.models.huggingface import HFLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", required=True, help="HF model id or local path")
    parser.add_argument("--tasks", nargs="+", default=["mmlu"], help="Harness tasks, e.g., mmlu truthfulqa_mc2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per task (for quick sanity)")
    parser.add_argument("--output_dir", type=Path, default=Path("./harness_results"))

    # RSN args
    parser.add_argument("--diff_path", type=str, help="Path to .npy diff (shape [n_layers,H] or [H])")
    parser.add_argument("--alpha", type=float, default=1.0, help="Scale factor for diff")
    parser.add_argument("--tail_len", type=int, default=1)
    parser.add_argument("--layer_indices", type=str, default=None,
                        help="Comma-separated layer indices to edit, e.g., 11,12,13. If omitted, edit all.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1) Baseline (original) ----------
    print("\n=== Running BASELINE (original) ===")
    base = HFLM(
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        # other HFLM kwargs if needed, e.g. use_accelerate=True
    )
    res_base = evaluator.simple_evaluate(
        model=base,
        tasks=args.tasks,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    base_path = args.output_dir / "results_original.json"
    with base_path.open("w", encoding="utf-8") as f:
        json.dump(res_base, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {base_path}")

    # ---------- 2) Edited (RSN) ----------
    if args.diff_path is None:
        print("\n[Edited run skipped] --diff_path not provided.")
        return

    print("\n=== Running EDITED (RSN enabled) ===")
    diff = np.load(args.diff_path)
    layer_indices = None
    if args.layer_indices:
        layer_indices = [int(x) for x in args.layer_indices.split(",") if x.strip() != ""]

    rsn_cfg = {
        "diff_matrices": diff,
        "alpha": args.alpha,
        "tail_len": args.tail_len,
        "layer_indices": layer_indices,
    }
    edited = HFLMWithRSN(
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        rsn_cfg=rsn_cfg,
    )
    res_edit = evaluator.simple_evaluate(
        model=edited,
        tasks=args.tasks,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    edit_path = args.output_dir / "results_edited.json"
    with edit_path.open("w", encoding="utf-8") as f:
        json.dump(res_edit, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {edit_path}")

    print("\n✅ Done. Compare results_original.json vs results_edited.json")


if __name__ == "__main__":
    main()