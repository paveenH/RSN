#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download FACTOR / FACTors datasets from Hugging Face (if available),
export each available split to JSONL, and also merge into a single JSONL
for convenient downstream use with your converter script.

- Code and comments are in English as requested.
- It tries multiple candidate repo IDs to be robust.
- Outputs live under your cache_dir, so you can set:
    FACTOR_INPUT_PATH  = f"{cache_dir}/factor/factor.jsonl"
    FACTORS_INPUT_PATH = f"{cache_dir}/factors/factors.jsonl"
"""

import os
import json
from typing import List, Optional
from datasets import load_dataset, Dataset

# ----------------------
# Configuration (edit)
# ----------------------
cache_dir = "/data2/paveen/RolePlaying/.cache"
save_root  = "/data2/paveen/RolePlaying/components/"

# Candidate HF repo IDs to try (you can add/remove as needed)
FACTOR_REPO_CANDIDATES  = [
    "AI21Labs/factor",
    "ai21labs/factor",
    "AI21Labs/FACTOR",
]
FACTOR_CONFIGS = [None]  # add config names if needed
FACTOR_SPLITS  = ["train", "validation", "test"]  # we will also try 'train[:100%]' fallback when missing

FACTORS_REPO_CANDIDATES = [
    # Replace with the correct repo when you know it.
    # Keeping candidates to avoid hard failure up front.
    "AI21Labs/factors",
    "ai21labs/factors",
    # If you know the exact repo later, put it at the top of this list.
]
FACTORS_CONFIGS = [None]
FACTORS_SPLITS  = ["train", "validation", "test"]

# Output paths (final merged JSONL per dataset)
FACTOR_OUT_MERGED_JSONL  = os.path.join(save_root, "factor", "factor.jsonl")
FACTORS_OUT_MERGED_JSONL = os.path.join(save_root, "factors", "factors.jsonl")


# ----------------------
# Helpers
# ----------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def dataset_to_jsonl(ds: Dataset, out_path: str) -> None:
    """Write a HuggingFace Dataset to JSONL."""
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

def merge_jsonls(input_paths: List[str], out_path: str) -> None:
    """Concatenate multiple JSONL files into one."""
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as w:
        for p in input_paths:
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as r:
                for line in r:
                    w.write(line)

def try_load_any(repo_candidates: List[str],
                 configs: List[Optional[str]],
                 split: Optional[str],
                 cache_dir: str):
    """
    Try to load a dataset from multiple (repo, config) pairs.
    Returns (repo, config, dataset) on success; raises last exception on failure.
    """
    last_err = None
    for repo in repo_candidates:
        for cfg in configs:
            try:
                ds = load_dataset(repo, cfg, split=split, cache_dir=cache_dir)
                return repo, cfg, ds
            except Exception as e:
                last_err = e
                continue
    raise last_err if last_err else RuntimeError("No repo/config worked.")

def export_all_splits(repo_candidates: List[str],
                      configs: List[Optional[str]],
                      splits: List[str],
                      cache_dir: str,
                      save_dir: str,
                      merged_out_path: str) -> None:
    """
    For each split in 'splits', try to load and export to JSONL.
    Also write a merged JSONL combining all exported splits.
    """
    ensure_dir(save_dir)
    exported_paths = []

    # Probe available splits; some datasets may not have all 'splits'.
    for sp in splits:
        try:
            repo, cfg, ds = try_load_any(repo_candidates, configs, sp, cache_dir)
            out_path = os.path.join(save_dir, f"{sp}.jsonl")
            dataset_to_jsonl(ds, out_path)
            print(f"[OK] Exported {repo} ({'config='+cfg if cfg else 'no-config'}) split='{sp}' → {out_path} (n={len(ds)})")
            exported_paths.append(out_path)
        except Exception:
            # Try a common fallback style (e.g., 'train[:100%]') to force materialization in some repos
            try:
                repo, cfg, ds = try_load_any(repo_candidates, configs, f"{sp}[:100%]", cache_dir)
                out_path = os.path.join(save_dir, f"{sp}.jsonl")
                dataset_to_jsonl(ds, out_path)
                print(f"[OK] Exported {repo} ({'config='+cfg if cfg else 'no-config'}) split='{sp}[:100%]' → {out_path} (n={len(ds)})")
                exported_paths.append(out_path)
            except Exception as e2:
                print(f"[SKIP] Could not load split '{sp}': {e2}")

    if not exported_paths:
        raise SystemExit(f"[ERROR] No splits were exported for candidates: {repo_candidates}")

    # Merge into one JSONL
    merge_jsonls(exported_paths, merged_out_path)
    print(f"[MERGED] Wrote merged JSONL → {merged_out_path}")

def main():
    # FACTOR
    factor_save_dir = os.path.join(save_root, "factor")
    try:
        export_all_splits(
            repo_candidates=FACTOR_REPO_CANDIDATES,
            configs=FACTOR_CONFIGS,
            splits=FACTOR_SPLITS,
            cache_dir=cache_dir,
            save_dir=factor_save_dir,
            merged_out_path=FACTOR_OUT_MERGED_JSONL,
        )
        print(f"[INFO] Set this in your converter:\n  FACTOR_INPUT_PATH = '{FACTOR_OUT_MERGED_JSONL}'")
    except Exception as e:
        print(f"[WARN] FACTOR download/export failed: {e}\n"
              f"→ Please verify the exact HF repo id/config and update FACTOR_REPO_CANDIDATES.")

    # FACTors
    factors_save_dir = os.path.join(save_root, "factors")
    try:
        export_all_splits(
            repo_candidates=FACTORS_REPO_CANDIDATES,
            configs=FACTORS_CONFIGS,
            splits=FACTORS_SPLITS,
            cache_dir=cache_dir,
            save_dir=factors_save_dir,
            merged_out_path=FACTORS_OUT_MERGED_JSONL,
        )
        print(f"[INFO] Set this in your converter:\n  FACTORS_INPUT_PATH = '{FACTORS_OUT_MERGED_JSONL}'")
    except Exception as e:
        print(f"[WARN] FACTors download/export failed: {e}\n"
              f"→ Please fill the correct HF repo id in FACTORS_REPO_CANDIDATES.")

if __name__ == "__main__":
    main()