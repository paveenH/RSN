#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_answer_classifier_mmlupro_mmlu.py — Classifier-Guided Steering Benchmark (MMLU-Pro style)
===============================================================================================
Three-way comparison on MMLU-Pro style benchmarks:

  A) No Steering    — original forward pass, no diff injection
  B) Always Steer   — apply +alpha steering to every sample
  C) Classifier     — PCA-CNN (MMLU-trained) predicts orig_correct;
                      steer only when predicted wrong (y=0)

Uses the MMLU-trained PCA-CNN classifier (with residual connection) from:
  <clf_dir>/model.pt          — weights + config dict
  <clf_dir>/preprocessor.pkl  — list of scalers + list of PCAs (per layer)

Usage:
  python get_answer_classifier_mmlupro_mmlu.py \\
    --model llama3 \\
    --model_dir meta-llama/Meta-Llama-3-8B-Instruct \\
    --size 8B \\
    --type non \\
    --test_file benchmark/mmlupro_test.json \\
    --ans_file answer_clf_mmlupro_mmlu \\
    --clf_dir /data1/paveen/ConfSteer/models/llama3_pca128_mmlu_cnn_k7 \\
    --hs llama3 \\
    --percentage 0.5 \\
    --configs 4-11-20 \\
    --mask_type nmd \\
    --base_dir /data1/paveen/RolePlaying/components \\
    --roles neutral
"""

import gc
import copy
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from llms import VicundaModel
from template import select_templates_pro
import utils


# ──────────────────────────────────────────────
#  PCA-CNN with residual connection (matches ConfSteer classifier_pca_cnn.py)
# ──────────────────────────────────────────────

class PCA_CNN(nn.Module):
    def __init__(self, num_layers, pca_dim, cnn_channels=64, kernel_size=3, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(pca_dim, cnn_channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size, padding=kernel_size // 2)
        self.proj  = nn.Conv1d(pca_dim, cnn_channels, 1) if pca_dim != cnn_channels else nn.Identity()
        self.act   = nn.GELU()
        self.attn  = nn.Linear(cnn_channels, 1)
        self.head  = nn.Sequential(
            nn.Linear(cnn_channels, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        # x: (B, L, pca_dim)
        x = x.permute(0, 2, 1)                   # (B, pca_dim, L)
        residual = self.proj(x)                   # (B, cnn_ch, L)
        x = self.act(self.conv1(x))               # (B, cnn_ch, L)
        x = self.act(self.conv2(x) + residual)    # residual connection
        x = x.permute(0, 2, 1)                   # (B, L, cnn_ch)
        w = torch.softmax(self.attn(x).squeeze(-1), dim=-1).unsqueeze(-1)
        x = (x * w).sum(dim=1)                   # (B, cnn_ch)
        return self.head(x)


# ──────────────────────────────────────────────
#  Classifier loader + inference
# ──────────────────────────────────────────────

def load_classifier(clf_dir: Path, device: torch.device):
    """Load model.pt + preprocessor.pkl from clf_dir."""
    ckpt = torch.load(clf_dir / "model.pt", map_location="cpu")
    cfg  = ckpt["config"]

    model = PCA_CNN(
        num_layers=cfg["num_layers"],
        pca_dim=cfg["pca_dim"],
        cnn_channels=cfg.get("cnn_channels", 64),
        kernel_size=cfg.get("kernel_size", 3),
        dropout=cfg.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    with open(clf_dir / "preprocessor.pkl", "rb") as f:
        prep = pickle.load(f)
    scalers = prep["scalers"]
    pcas    = prep["pcas"]

    print(f"  Loaded classifier: L={cfg['num_layers']}, pca_dim={cfg['pca_dim']}, "
          f"cnn_ch={cfg.get('cnn_channels', 64)}, kernel={cfg.get('kernel_size', 3)}")
    return model, scalers, pcas, cfg


@torch.no_grad()
def classify(hidden_states: list, scalers: list, pcas: list,
             clf_model: nn.Module, pca_dim: int, device: torch.device) -> int:
    """
    hidden_states: list of L arrays, each (D,)  [last-token, all layers]
    Returns: 0 (predicted wrong) or 1 (predicted correct)
    """
    L = len(hidden_states)
    x = np.zeros((L, pca_dim), dtype=np.float32)
    for l in range(L):
        h = hidden_states[l].reshape(1, -1).astype(np.float32)
        h = scalers[l].transform(h)
        n_comp = pcas[l].n_components_
        x[l, :n_comp] = pcas[l].transform(h)[0]

    t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)  # (1, L, pca_dim)
    logits = clf_model(t)
    return int(logits.argmax(dim=1).item())


# ──────────────────────────────────────────────
#  Per-task runner
# ──────────────────────────────────────────────

def run_task(vc: VicundaModel, task: str, samples: list,
             diff_mtx: np.ndarray, suite: str,
             scalers, pcas, clf_model, pca_dim: int, clf_device: torch.device,
             args):
    """
    Run one task under three conditions: no_steer / always_steer / classifier.
    Returns accuracy dict keyed by (condition, role).
    """
    custom_roles = [r.strip() for r in args.roles.split(",")] if args.roles else None
    roles = utils.make_characters(task.replace(" ", "_"), custom_roles)

    conditions = ["no_steer", "always_steer", "classifier"]
    stats = {
        (cond, role): {"correct": 0, "steered": 0, "total": 0}
        for cond in conditions for role in roles
    }

    for sample in tqdm(samples, desc=task):
        K           = int(sample.get("num_options"))
        base_labels = [chr(ord("A") + i) for i in range(K)]
        templates   = select_templates_pro(suite=suite, labels=base_labels,
                                           use_E=False, cot=args.cot)
        LABELS      = templates["labels"]
        if not args.use_E:
            templates = utils.remove_honest(templates)

        opt_ids  = utils.option_token_ids(vc, LABELS)
        ctx      = sample.get("text", "")
        true_idx = int(sample.get("label", -1))
        true_lab = LABELS[true_idx]

        for role in roles:
            prompt = utils.construct_prompt(vc, templates, ctx, role, args.use_chat)

            # ── Original forward pass ──
            raw_logits_tensor, hidden = vc.get_logits([prompt], return_hidden=True)
            orig_logits = raw_logits_tensor[0, -1].float().cpu().numpy()
            opt_orig    = np.array([orig_logits[i] for i in opt_ids])

            # Extract last-token hidden states: list of L arrays (D,)
            last_hs = [lay[0, -1].float().cpu().numpy() for lay in hidden]

            # ── Steered forward pass ──
            steer_logits_raw = vc.regenerate_logits([prompt], diff_mtx, tail_len=args.tail_len)[0]
            opt_steer        = np.array([steer_logits_raw[i] for i in opt_ids])

            # ── Classifier decision ──
            pred_cls = classify(last_hs, scalers, pcas, clf_model, pca_dim, clf_device)

            for cond in conditions:
                if cond == "no_steer":
                    opt_logits = opt_orig
                    did_steer  = False
                elif cond == "always_steer":
                    opt_logits = opt_steer
                    did_steer  = True
                else:  # classifier
                    if pred_cls == 0:   # predicted wrong → steer
                        opt_logits = opt_steer
                        did_steer  = True
                    else:               # predicted correct → keep
                        opt_logits = opt_orig
                        did_steer  = False

                pred_lab = LABELS[int(opt_logits.argmax())]
                s = stats[(cond, role)]
                s["total"] += 1
                if pred_lab == true_lab:
                    s["correct"] += 1
                if did_steer:
                    s["steered"] += 1

    # Summarize
    accuracy = {}
    for (cond, role), s in stats.items():
        pct = s["correct"] / s["total"] * 100 if s["total"] else 0.0
        accuracy[(cond, role)] = {**s, "accuracy_percentage": round(pct, 2)}
        steer_pct = s["steered"] / s["total"] * 100 if s["total"] else 0.0
        print(f"  [{cond:<14}] {role:<25} acc={pct:5.2f}%  "
              f"steered={steer_pct:5.1f}%  ({s['correct']}/{s['total']})")

    return accuracy


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--model",     required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--size",      required=True)
    parser.add_argument("--type",      default="non")
    # Data
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--ans_file",  required=True)
    parser.add_argument("--suite",     default="default", choices=["default", "vanilla"])
    parser.add_argument("--roles",     type=str, default=None)
    parser.add_argument("--use_E",     action="store_true")
    parser.add_argument("--cot",       action="store_true")
    parser.add_argument("--use_chat",  action="store_true")
    # Steering
    parser.add_argument("--hs",         required=True)
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs",    nargs="*", default=["4-11-20"])
    parser.add_argument("--mask_type",  default="nmd")
    parser.add_argument("--abs",        action="store_true")
    parser.add_argument("--tail_len",   type=int, default=1)
    # Classifier
    parser.add_argument("--clf_dir",    required=True,
                        help="Path to ConfSteer classifier dir (model.pt + preprocessor.pkl)")
    parser.add_argument("--clf_device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Paths
    parser.add_argument("--base_dir",  type=str, default=None)
    parser.add_argument("--data",      default="data1", choices=["data1", "data2"])
    args = parser.parse_args()

    # ── Paths ──
    BASE      = Path(args.base_dir) if args.base_dir else Path(f"/{args.data}/paveen/RolePlaying/components")
    DATA_DIR  = BASE / args.test_file
    MASK_DIR  = BASE / "mask" / f"{args.hs}_{args.type}_logits"
    SAVE_ROOT = BASE / args.model / args.ans_file
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  Classifier-Guided Steering Benchmark  [MMLU-Pro / MMLU-trained clf]")
    print(f"  Model      : {args.model} ({args.size})")
    print(f"  Test file  : {DATA_DIR}")
    print(f"  Clf dir    : {args.clf_dir}")
    print(f"  Configs    : {args.configs}")
    print(f"{'='*65}\n")

    # ── Load classifier ──
    clf_device = torch.device(args.clf_device)
    clf_model, scalers, pcas, clf_cfg = load_classifier(Path(args.clf_dir), clf_device)
    pca_dim = clf_cfg["pca_dim"]

    # ── Load LLM ──
    vc = VicundaModel(model_path=args.model_dir)
    vc.model.eval()

    # ── Load benchmark ──
    all_samples = utils.load_json(DATA_DIR)
    tasks = sorted({s["task"] for s in all_samples})
    print(f"Found {len(tasks)} tasks.")

    ALPHAS_START_END = utils.parse_configs(args.configs)

    for alpha, (st, en) in ALPHAS_START_END:
        mask_suffix = "_abs" if args.abs else ""
        mask_name   = f"{args.mask_type}_{args.percentage}_{st}_{en}_{args.size}{mask_suffix}.npy"
        diff_mtx    = np.load(MASK_DIR / mask_name) * alpha
        TOP         = max(1, int(args.percentage / 100 * diff_mtx.shape[1]))
        print(f"\n=== α={alpha} | layers={st}-{en} | TOP={TOP} ===")

        csv_rows = []

        for task in tasks:
            task_samples = [copy.deepcopy(s) for s in all_samples if s["task"] == task]
            if not task_samples:
                continue

            print(f"\n--- Task: {task} ({len(task_samples)} samples) ---")
            with torch.no_grad():
                accuracy = run_task(
                    vc=vc, task=task, samples=task_samples,
                    diff_mtx=diff_mtx, suite=args.suite,
                    scalers=scalers, pcas=pcas,
                    clf_model=clf_model, pca_dim=pca_dim, clf_device=clf_device,
                    args=args,
                )

            # Save per-task JSON
            out_dir = SAVE_ROOT / f"clf_{alpha}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{task.replace(' ', '_')}_{args.size}_{TOP}_{st}_{en}.json"
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump({"accuracy": {
                    f"{cond}_{role}": v
                    for (cond, role), v in accuracy.items()
                }}, fw, ensure_ascii=False, indent=2)
            print(f"  Saved → {out_path}")

            # Collect CSV rows
            for (cond, role), s in accuracy.items():
                csv_rows.append({
                    "model": args.model, "size": args.size,
                    "alpha": alpha, "start": st, "end": en, "TOP": TOP,
                    "task": task, "role": role, "condition": cond,
                    "correct": s["correct"], "steered": s["steered"],
                    "total": s["total"],
                    "accuracy_percentage": s["accuracy_percentage"],
                })

            del task_samples
            gc.collect()

        # Save summary CSV
        csv_path = SAVE_ROOT / f"clf_{alpha}" / f"summary_{args.model}_{args.size}_{TOP}_{st}_{en}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "model", "size", "alpha", "start", "end", "TOP",
                "task", "role", "condition",
                "correct", "steered", "total", "accuracy_percentage",
            ])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n[Saved CSV] {csv_path}")

        gc.collect()
        torch.cuda.empty_cache()

    print("\n✅  All tasks finished.")


if __name__ == "__main__":
    main()
