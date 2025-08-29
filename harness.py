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


def run_one_eval(pretrained, tasks, batch_size, limit, rsn_cfg, out_path: Path, fewshot: int | None):
    if rsn_cfg is None:
        model = HFLM(pretrained=pretrained, batch_size=batch_size, device_map="auto")
    else:
        model = HFLMWithRSN(pretrained=pretrained, batch_size=batch_size, rsn_cfg=rsn_cfg, device_map="auto")

    res = evaluator.simple_evaluate(model=model, tasks=tasks, batch_size=batch_size, limit=limit, num_fewshot=fewshot)

    metrics_only = _to_py(res.get("results", {}))
    print(metrics_only)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_only, f, ensure_ascii=False, indent=2)
    print(f"[Saved metrics] {out_path}")

    del model
    import gc, torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return res


def make_save_path(
    save_dir: Path,
    task: str,
    model: str,
    size: str,
    *,
    fewshot: bool = False,
    alpha: float | None = None,
    top: int | None = None,
    st: int | None = None,
    en: int | None = None,
    tail_len: int | None = None,
    is_baseline: bool = False,
) -> Path:
    """
    - baseline: {task}[_fewshot]_original_{model}_{size}.json
    - edited:   {task}[_fewshot]_{model}_{size}_[neg]alpha_{TOP}_{st}_{en}_tail{tail}.json
    """
    few = "_fewshot" if fewshot else ""
    if is_baseline:
        return save_dir / f"{task}{few}_original_{model}_{size}.json"

    assert alpha is not None and top is not None and st is not None and en is not None and tail_len is not None
    if alpha >= 0:
        alpha_tag = f"{alpha}"
    else:
        alpha_tag = f"neg{abs(alpha)}"

    return save_dir / f"{task}{few}_{model}_{size}_{alpha_tag}_{top}_{st}_{en}_tail{tail_len}.json"


def main(args):

    task0 = args.tasks[0]

    # Configuration
    if args.configs:
        cfgs = utils.parse_configs(args.configs)
        print("ALPHAS_START_END_PAIRS:", cfgs)
    else:
        cfgs = []

    if not cfgs:
        base_out = make_save_path(SAVE_DIR, task0, args.model, args.size, fewshot=bool(args.fewshot), is_baseline=True)

        print("\n=== Running BASELINE (original only, no configs) ===")

        run_one_eval(
            pretrained=args.model_dir,
            tasks=args.tasks,
            batch_size="auto",
            limit=args.limit,
            rsn_cfg=None,
            out_path=base_out,
            fewshot=args.fewshot,
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
        top = int(max(1, args.percentage / 100.0 * (diff.shape[1] if diff.ndim == 2 else diff.shape[0])))

        rsn_cfg = {
            "diff_matrices": diff,
            "alpha": alpha,
            "tail_len": args.tail_len,
            "layer_indices": None,
        }

        edit_out = make_save_path(
            SAVE_DIR,
            task0,
            args.model,
            args.size,
            fewshot=bool(args.fewshot),
            alpha=alpha,
            top=top,
            st=st,
            en=en,
            tail_len=args.tail_len,
            is_baseline=False,
        )

        print("\n=== Running EDITED (RSN enabled) ===")
        run_one_eval(
            pretrained=args.model_dir,
            tasks=args.tasks,
            batch_size="auto",
            limit=args.limit,
            rsn_cfg=rsn_cfg,
            out_path=edit_out,
            fewshot=args.fewshot,
        )

    print("\n✅  Done (edited runs).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LM Evaluation Harness (original vs edited) with RSN hooks, using your original-style CLI."
    )
    parser.add_argument("--model", type=str, default="qwen2.5_base")
    parser.add_argument("--model_dir", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--hs", type=str, default="qwen2.5")
    parser.add_argument("--size", type=str, default="7B")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="*", default=None, help="alpha-start-end triplets, e.g., 4-16-22")
    parser.add_argument("--mask_type", type=str, default="nmd", help="Mask type: nmd or random")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--ans_file", type=str, default="tqa_edit_answers")
    parser.add_argument("--tail_len", type=int, default=1, help="Number of last tokens to apply diff")
    parser.add_argument("--tasks", nargs="+", default=["mmlu"], help="Harness task list, e.g., mmlu truthfulqa_mc2")
    parser.add_argument("--fewshot", type=int, default=None, help="K-shot in-context examples per task")
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
