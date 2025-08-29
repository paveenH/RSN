#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run LM Evaluation Harness twice per config:
- Baseline (original, no editing)
- Edited (with RSN hooks via HFLMWithRSN)

"""

import json
from pathlib import Path
import argparse
import numpy as np

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from hf_rsn import HFLMWithRSN
import utils


def _to_py(o):
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(v) for v in o]
    return o


def mask_filename(mask_type: str, percentage: float, start: int, end: int, size: str, use_abs: bool):
    suf = "_abs" if use_abs else ""
    return f"{mask_type}_{percentage}_{start}_{end}_{size}{suf}.npy"


def run_one_eval(pretrained, tasks, batch_size, limit, rsn_cfg, out_path: Path):
    if rsn_cfg is None:
        model = HFLM(pretrained=pretrained, batch_size=batch_size)
    else:
        model = HFLMWithRSN(pretrained=pretrained, batch_size=batch_size, rsn_cfg=rsn_cfg)

    res = evaluator.simple_evaluate(
        model=model,
        tasks=tasks,
        batch_size=batch_size,
        limit=limit,
    )

    metrics_only = _to_py(res.get("results", {}))
    print (metrics_only)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_only, f, ensure_ascii=False, indent=2)
    print(f"[Saved metrics] {out_path}")
    
    del model
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return res

def main(args):
    
    # Configuration
    if args.configs:
        cfgs = utils.parse_configs(args.configs)
        print("ALPHAS_START_END_PAIRS:", cfgs)
    else:
        cfgs = []
    
    if not cfgs:
        base_out = SAVE_DIR / f"{args.tasks}_original_{args.model}_{args.size}.json"
        print("\n=== Running BASELINE (original only, no configs) ===")
        run_one_eval(
            pretrained=args.model_dir,
            tasks=args.tasks,
            batch_size="auto",
            limit=args.limit,
            rsn_cfg=None,
            out_path=base_out,
        )
        print("\n✅  Done (baseline only).")
        return
    
    for alpha, (st, en) in cfgs:
        mask_name = mask_filename(args.mask_type, args.percentage, st, en, args.size, args.abs)
        mask_path = Path(MASK_DIR) / mask_name

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        print(f"\n=== α={alpha} | layers={st}-{en} | mask={mask_path.name} ===")
        diff = np.load(str(mask_path))
        TOP = int(max(1, args.percentage/100.0 * (diff.shape[1] if diff.ndim==2 else diff.shape[0])))

        rsn_cfg = {
            "diff_matrices": diff,
            "alpha": alpha,
            "tail_len": args.tail_len,
            "layer_indices": None,
        }
        edit_out = SAVE_DIR / f"{args.tasks}_{args.model}_{args.size}_{TOP}_{st}_{en}_tail{args.tail_len}.json"
        print("\n=== Running EDITED (RSN enabled) ===")
        run_one_eval(
            pretrained=args.model_dir,
            tasks=args.tasks,
            batch_size="auto",
            limit=args.limit,
            rsn_cfg=rsn_cfg,
            out_path=edit_out,
        )

    print("\n✅  Done (edited runs).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LM Evaluation Harness (original vs edited) with RSN hooks, using your original-style CLI.")
    parser.add_argument("--model", type=str, default="qwen2.5_base")
    parser.add_argument("--model_dir", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--hs", type=str, default="qwen2.5")
    parser.add_argument("--size", type=str, default="7B")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="*", default=["4-16-22"], help="alpha-start-end triplets, e.g., 4-16-22")
    parser.add_argument("--mask_type", type=str, default="nmd", help="Mask type: nmd or random")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--ans_file", type=str, default="tqa_edit_answers")
    parser.add_argument("--tail_len", type=int, default=1, help="Number of last tokens to apply diff")
    parser.add_argument("--tasks", nargs="+", default=["mmlu"], help="Harness task list, e.g., mmlu truthfulqa_mc2")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per task (for quick sanity)")

    args = parser.parse_args()

    SAVE_DIR = Path(f"/data2/paveen/RolePlaying/components/{args.ans_file}/")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR = f"/data2/paveen/RolePlaying/components/mask/{args.hs}_non_logits"
    
    print("Model:", args.model)
    print("Import model from:", args.model_dir)
    print("HS:", args.hs)
    print("Mask dir:", MASK_DIR)
    print("Save root:", SAVE_DIR)
    
    main(args)