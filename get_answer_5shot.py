#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute length-normalized Negative Log-Likelihood (LN-NLL) for multiple-choice QA
(e.g., MMLU), with optional in-context k-shot prompts. This script is designed
to align with lm-eval-harness "choice" scoring, WITHOUT chat templates or roles.

- Prompt format (few-shot):
  INTRO -> k exemplars (Question + choices + Answer: X) -> TEST (Question + choices + Answer:)
- Scoring:
  For each label (letter or full-option text), compute token-wise log-probs on the
  answer segment only; LN-NLL = -(sum logprobs) / (#answer_tokens); pick the smallest.
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from llms import VicundaModel
from detection.task_list import TASKS
from utils import load_json, build_fewshot_prompt

# ----------------------------- Helpers ---------------------------------

def tokenize(tokenizer, text: str):
    return tokenizer(text, add_special_tokens=False, return_tensors=None).input_ids

def ln_nll_for_candidate_ids(logits: torch.Tensor, ids: List[int], prefix_len: int) -> float:
    """
    Compute length-normalized NLL for the answer segment.
    logits: (L, V) torch.float (on device); corresponds to full sequence (prompt+answer).
    ids: full token ids for the same input
    prefix_len: index where answer tokens start (0-based)
    Returns: scalar LN-NLL = -mean_t log P(token_t | prefix + previous answer tokens)
    """
    # logits[t] predicts ids[t+1], so for token at position i (i >= 1), use logits[i-1]
    # answer tokens positions: prefix_len .. len(ids)-1
    start = prefix_len
    end = len(ids) - 1  # last index that has a previous logit
    if end < start:
        return float("inf")

    # Gather logits rows that predict each answer token (shifted by -1)
    rows = logits[start - 1 : len(ids) - 1] if start > 0 else logits[: len(ids) - 1]
    # Corresponding token ids to score
    tgt = torch.tensor(ids[start:], dtype=torch.long, device=logits.device)

    logprobs = F.log_softmax(rows, dim=-1)
    picked = logprobs.gather(-1, tgt.view(-1, 1)).squeeze(-1)  # (answer_len,)
    nll = -picked.mean()  # length-normalized NLL
    return float(nll.item())

def build_test_query_text(sample: dict, use_E: bool) -> str:
    """Build the test question block ending with 'Answer:' (no role, no chat)."""
    labels = ["A","B","C","D","E"] if use_E else ["A","B","C","D"]
    lines = []
    lines.append(f"Question: {sample['text']}")
    # If sample already contains choice strings, prefer them; else assume embedded in text.
    # Here we follow the prior structure you used: choices may be pre-separated or in text.
    choices = sample.get("choices", None)
    if choices is not None:
        for i, ch in enumerate(choices):
            lines.append(f"{labels[i]}) {ch}")
        if use_E and len(choices) == 4:
            lines.append("E) I am not sure.")
    else:
        # If no explicit choices, assume they are already included in sample['text'] lines.
        if use_E:
            lines.append("E) I am not sure.")
    lines.append("Answer:")
    return "\n".join(lines)

def candidate_texts_for_labels(sample: dict, label_mode: str, use_E: bool) -> List[str]:
    """
    Return candidate completions to append after the test prompt.
    - 'letter' -> [" A", " B", ...]  (space + capital letter)
    - 'full'   -> full option lines like " A) Paris" ...
    """
    labels = ["A","B","C","D","E"] if use_E else ["A","B","C","D"]
    if label_mode == "letter":
        return [f" {lab}" for lab in labels]
    elif label_mode == "full":
        choices = sample.get("choices", None)
        if choices is None:
            # Fall back to just the letter if full text is unavailable
            return [f" {lab}" for lab in labels]
        cands = []
        for i, lab in enumerate(labels):
            if i < len(choices):
                cands.append(f" {lab}) {choices[i]}")
            else:
                # E case without explicit line
                cands.append(" E) I am not sure.")
        return cands
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

def predict_lnnll_for_sample(
    vc: VicundaModel,
    sample: dict,
    fewshot_prompt: str,
    label_mode: str,
    use_E: bool,
) -> Tuple[int, List[float], List[int]]:
    """
    Build test query, then for each candidate (label), compute LN-NLL on the answer segment.
    Returns:
      pred_idx, lnnll_list, answer_token_lengths
    """
    tokenizer = vc.tokenizer
    device = vc.model.device

    test_block = build_test_query_text(sample, use_E)
    prompt = f"{fewshot_prompt}\n\n{test_block}" if fewshot_prompt else test_block

    base_ids = tokenize(tokenizer, prompt)
    base_len = len(base_ids)

    lnnlls: List[float] = []
    alens: List[int] = []

    for cand in candidate_texts_for_labels(sample, label_mode, use_E):
        full_text = prompt + cand
        full_ids = tokenize(tokenizer, full_text)
        # Get token-wise logits for the full sequence
        with torch.no_grad():
            toks = vc.tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(device)
            outputs = vc.model(**toks, return_dict=True, output_hidden_states=False)
            logits = outputs.logits[0]  # (L, V)
        # Compute LN-NLL over the answer tokens only
        nll = ln_nll_for_candidate_ids(logits, full_ids, prefix_len=base_len)
        lnnlls.append(nll)
        alens.append(len(full_ids) - base_len)

    pred_idx = int(np.argmin(lnnlls))
    return pred_idx, lnnlls, alens

# ----------------------------- Main ------------------------------------

def main():
    args = parse_args()

    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()
    tokenizer = vc.tokenizer

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    labels = ["A","B","C","D","E"] if args.use_E else ["A","B","C","D"]

    for task in TASKS if not args.only_tasks else args.only_tasks:
        data_path = Path(args.mmlu_dir) / f"{task}.json"
        if not data_path.exists():
            print(f"[Skip] {data_path} not found")
            continue

        samples = load_json(str(data_path))
        # optional support pool (recommended to avoid leakage)
        support_pool = None
        if args.support_dir:
            sup_path = Path(args.support_dir) / f"{task}.json"
            if sup_path.exists():
                support_pool = load_json(str(sup_path))
            else:
                print(f"[WARN] support file {sup_path} not found. Will fallback to in-file sampling (may leak).")

        stats = {"total": 0, "correct": 0}
        per_label_counts = {lab: {"tp": 0, "pred": 0, "gold": 0} for lab in labels}
        results = []

        for sample in tqdm(samples, desc=f"{task}"):
            true_idx = sample.get("label", -1)
            if not (0 <= true_idx < len(labels)):
                continue

            # Build few-shot section (or empty for 0-shot)
            fewshot_prompt = ""
            if args.fewshot > 0:
                # If no external support, fallback to sampling within the same file (exclude itself)
                pool = support_pool if support_pool is not None else [s for s in samples if s is not sample]
                fewshot_prompt = build_fewshot_prompt(
                    test_sample=sample,
                    support_pool=pool,
                    k=args.fewshot,
                    use_E=args.use_E,
                    tokenizer=tokenizer,
                    max_tokens=args.max_prompt_tokens,
                    global_seed=args.global_seed,
                    subject=task,  # use task name as subject label
                )

            pred_idx, lnnlls, ans_lens = predict_lnnll_for_sample(
                vc=vc,
                sample=sample,
                fewshot_prompt=fewshot_prompt,
                label_mode=args.label_mode,
                use_E=args.use_E,
            )
            pred_label = labels[pred_idx]
            gold_label = labels[true_idx]
            correct = int(pred_idx == true_idx)

            stats["total"] += 1
            stats["correct"] += correct
            per_label_counts[pred_label]["pred"] += 1
            per_label_counts[gold_label]["gold"] += 1
            if correct:
                per_label_counts[gold_label]["tp"] += 1

            results.append({
                "id": sample.get("id", None),
                "task": task,
                "pred_idx": pred_idx,
                "pred_label": pred_label,
                "gold_idx": true_idx,
                "gold_label": gold_label,
                "lnnll": lnnlls,                # per-label LN-NLL
                "answer_token_lens": ans_lens,  # per-label answer token lengths
            })

        acc = (stats["correct"] / stats["total"] * 100.0) if stats["total"] else 0.0
        summary = {
            "task": task,
            "shots": args.fewshot,
            "label_mode": args.label_mode,
            "use_E": args.use_E,
            "global_seed": args.global_seed,
            "stats": {**stats, "accuracy_percentage": round(acc, 2)},
            "per_label": per_label_counts,
        }

        out_path = out_root / f"{task}_{args.size}_lnnll_{args.fewshot}shot_{args.label_mode}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "data": results}, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {out_path}  acc={acc:.2f}%")

    print("\n✅  All tasks finished.")


def parse_args():
    p = argparse.ArgumentParser("Compute LN-NLL for multiple-choice QA (MMLU-style)")
    p.add_argument("--model", "-m", type=str, required=True, help="Name tag for outputs")
    p.add_argument("--size", "-s", type=str, required=True, help="Model size tag, e.g., 8B")
    p.add_argument("--model_dir", type=str, required=True, help="HF model path or local dir")
    p.add_argument("--mmlu_dir", type=str, default="/data2/paveen/RolePlaying/components/mmlu")
    p.add_argument("--support_dir", type=str, default="", help="Optional support pool (dev/train) dir to avoid leakage")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory")
    p.add_argument("--fewshot", type=int, default=5, help="k-shot; 0 for zero-shot")
    p.add_argument("--use_E", action="store_true", help="Use five-choice (A–E)")
    p.add_argument("--label_mode", choices=["letter", "full"], default="letter",
                   help="Answer segment: single letter (recommended) or full option line")
    p.add_argument("--max_prompt_tokens", type=int, default=8192)
    p.add_argument("--global_seed", type=int, default=0)
    p.add_argument("--only_tasks", nargs="*", default=None, help="Subset of TASKS to run")
    return p.parse_args()


if __name__ == "__main__":
    main()
    
    