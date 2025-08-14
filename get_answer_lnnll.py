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
from typing import List
import re

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from llms import VicundaModel
from detection.task_list import TASKS
from utils import load_json, build_fewshot_prefix, build_query_block

# ----------------------------- Helpers ---------------------------------


def extract_choice_lines(text: str, labels: str = "ABCD", use_E: bool = False) -> list[str]:
    """
    Extract choice lines (e.g., "A) ...", "B) ...") directly from question text,
    and return them in the order of the provided labels.
    """
    pat = re.compile(r"^\s*([A-Da-d])\)\s*(.*?)\s*$", re.M)
    pairs = [(m.group(1).upper(), m.group(2)) for m in pat.finditer(text)]
    order = {ch: i for i, ch in enumerate(labels)}
    pairs = sorted([p for p in pairs if p[0] in order], key=lambda x: order[x[0]])

    results = [f" {L}) {body}" for L, body in pairs]
    if use_E:
        results.append(" E) I am not sure.")
    return results


def ln_nll_for_candidate_ids(logits: torch.Tensor, ids: List[int], prefix_len: int) -> float:
    start = prefix_len
    end = len(ids) - 1
    if end < start:
        return float("inf")

    rows = logits[start - 1 : len(ids) - 1] if start > 0 else logits[: len(ids) - 1]
    tgt = torch.tensor(ids[start:], dtype=torch.long, device=logits.device)

    logprobs = F.log_softmax(rows, dim=-1)
    picked = logprobs.gather(-1, tgt.view(-1, 1)).squeeze(-1)
    nll = -picked.mean()
    return float(nll.item())


@torch.no_grad()
def predict_lnnll_for_sample(
    vc: "VicundaModel",
    sample: dict,
    prefix: str,  # Few-shot prefix (empty string means 0-shot)
    use_E: bool,
) -> tuple[int, list[float], list[int]]:
    """
    Build a complete prompt (prefix + query), compute the LN-NLL for each
    candidate answer segment (with add_special_tokens=False everywhere),
    and select the lowest LN-NLL.
    Returns: (pred_idx, lnnll_per_label, answer_token_lengths_per_label)
    """
    # 1) Build the query block that ends with "Answer:"
    query = build_query_block(sample, use_E=use_E)
    prompt_text = f"{prefix}\n\n{query}" if prefix else query

    # 2) Tokenize the base prompt WITHOUT special tokens to define prefix_len
    base_enc = vc.tokenizer(prompt_text, add_special_tokens=False, return_tensors=None)
    base_ids = base_enc["input_ids"] if isinstance(base_enc, dict) else base_enc

    prefix_len = len(base_ids)

    # 3) Build candidate completions (e.g., " A) ...", " B) ...", ...)
    candidates = extract_choice_lines(sample.get("text", ""), use_E=use_E)

    lnnlls: list[float] = []
    answer_lens: list[int] = []

    for cand in candidates:
        full_text = prompt_text + cand

        # 4) Forward pass with add_special_tokens=False to keep indices aligned
        toks = vc.tokenizer(full_text, add_special_tokens=False, return_tensors="pt").to(vc.model.device)
        outputs = vc.model(**toks, return_dict=True)
        logits = outputs.logits[0]  # (L, V)
        full_ids = toks.input_ids[0].tolist()  # length-aligned ids

        # 5) LN-NLL over answer segment only
        nll = ln_nll_for_candidate_ids(logits, full_ids, prefix_len=prefix_len)
        lnnlls.append(nll)
        answer_lens.append(len(full_ids) - prefix_len)

    pred_idx = int(np.argmin(lnnlls))
    return pred_idx, lnnlls, answer_lens


# ----------------------------- Main ------------------------------------


def main():
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    labels = ["A", "B", "C", "D", "E"] if args.use_E else ["A", "B", "C", "D"]

    for task in TASKS:
        data_path = MMLU_DIR / f"{task}.json"
        if not data_path.exists():
            print(f"[Skip] {data_path} not found")
            continue

        samples = load_json(str(data_path))
        # The fixed 5-shot prefixes per task are already prepared in utils.
        # Just retrieve them here. Pass task-related args if your implementation requires them.
        fewshot_prefix = build_fewshot_prefix(
            task=task,
            fewshot_dir=args.fewshot_dir,
            k=args.fewshot,
            use_E=args.use_E,
        )

        stats = {"total": 0, "correct": 0}
        per_label_counts = {lab: {"tp": 0, "pred": 0, "gold": 0} for lab in labels}
        results = []

        for sample in tqdm(samples, desc=f"{task}"):
            true_idx = sample.get("label", -1)
            if not (0 <= true_idx < len(labels)):
                continue

            pred_idx, lnnlls, ans_lens = predict_lnnll_for_sample(vc=vc, sample=sample, prefix=fewshot_prefix, use_E=args.use_E)
            pred_label = labels[pred_idx]
            gold_label = labels[true_idx]
            correct = int(pred_idx == true_idx)

            stats["total"] += 1
            stats["correct"] += correct
            per_label_counts[pred_label]["pred"] += 1
            per_label_counts[gold_label]["gold"] += 1
            if correct:
                per_label_counts[gold_label]["tp"] += 1

            results.append(
                {
                    "task": task,
                    "text": sample.get("text", ""),
                    "label": true_idx,
                    "pred_label": pred_label,
                    "gold_label": gold_label,
                    "lnnll": lnnlls,
                    "answer_token_lens": ans_lens,
                }
            )

        acc = (stats["correct"] / stats["total"] * 100.0) if stats["total"] else 0.0
        summary = {
            "task": task,
            "use_E": args.use_E,
            "stats": {**stats, "accuracy_percentage": round(acc, 2)},
            "per_label": per_label_counts,
        }

        out_path = ANS_DIR / f"{task}_{args.size}_lnnll.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "data": results}, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {out_path}  acc={acc:.2f}%")

    print("\n✅  All tasks finished.")


if __name__ == "__main__":

    p = argparse.ArgumentParser("Compute LN-NLL for multiple-choice QA (MMLU-style)")
    p.add_argument("--model", "-m", type=str, required=True, help="Name tag for outputs")
    p.add_argument("--size", "-s", type=str, required=True, help="Model size tag, e.g., 8B")
    p.add_argument("--model_dir", type=str, required=True, help="HF model path or local dir")
    p.add_argument("--ans_file", type=str, required=True, help="Output directory")
    p.add_argument("--use_E", action="store_true", help="Use five-choice (A–E)")

    args = p.parse_args()

    MMLU_DIR = Path("/data2/paveen/RolePlaying/components/mmlu")
    ANS_DIR = Path(f"/data2/paveen/RolePlaying/components/{args.ans_file}/{args.model}")
    ANS_DIR.mkdir(parents=True, exist_ok=True)
    main()
