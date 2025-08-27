#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inspect MedQA (English, source) test split.

Dataset: bigbio/med_qa
Config : med_qa_en_source
Split  : test

"""

from datasets import load_dataset
from collections import Counter
import random
import json
import ast
from typing import List, Any, Optional

CFG = "med_qa_en_source"
SPLIT = "test"
REPO = "bigbio/med_qa"


def _normalize_option_item(item: Any) -> str:

    if isinstance(item, dict):
        if "value" in item:
            return str(item["value"])
        if "text" in item:
            return str(item["text"])
        return str(item)

    if isinstance(item, str):
        s = item.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                d = ast.literal_eval(s)
                if isinstance(d, dict):
                    if "value" in d:
                        return str(d["value"])
                    if "text" in d:
                        return str(d["text"])
                    return str(d)
            except Exception:
                pass
        return s

    return str(item)


def _get_options(row) -> List[str]:

    opts = row.get("options", None)
    if isinstance(opts, list):
        return [_normalize_option_item(o) for o in opts]

    for k in ("choices", "options_text", "answers"):
        v = row.get(k, None)
        if isinstance(v, list):
            return [_normalize_option_item(o) for o in v]

    return []


def _letter_to_index(letter: str) -> Optional[int]:
    letter = (letter or "").strip()
    if len(letter) == 1 and "A" <= letter <= "Z":
        return ord(letter) - ord("A")
    return None


def _get_answer_idx(row, options: List[str]) -> Optional[int]:
    if "answer_idx" in row and row["answer_idx"] is not None:
        try:
            return int(row["answer_idx"])
        except Exception:
            pass

    ans = row.get("answer", None)
    if ans is not None:
        idx = _letter_to_index(str(ans).strip().upper())
        if idx is not None:
            return idx
        ans_str = str(ans).strip()
        try:
            return options.index(ans_str)
        except ValueError:
            pass

    return None


def _format_example(q: str, options: List[str]) -> str:
    letters = [chr(ord("A") + i) for i in range(min(10, len(options)))]
    lines = [q.strip()]
    for i, opt in enumerate(options[:len(letters)]):
        lines.append(f"{letters[i]}) {opt}")
    return "\n".join(lines)


def main():
    print("=" * 80)
    print(f"Loading MedQA: repo='{REPO}', config='{CFG}', split='{SPLIT}'")
    ds = load_dataset(REPO, CFG, split=SPLIT)
    print(ds)
    print("Features:", ds.features)

    n = len(ds)
    print(f"\nTotal rows: {n}")

    opt_len_counter = Counter()
    ans_idx_counter = Counter()
    missing_gold = 0

    for i in range(n):
        row = ds[i]
        opts = _get_options(row)
        opt_len_counter[len(opts)] += 1

        ai = _get_answer_idx(row, opts)
        if ai is None:
            missing_gold += 1
        else:
            ans_idx_counter[ai] += 1

    print("\n#options distribution (count by number of options):")
    print(dict(sorted(opt_len_counter.items())))

    print("\nAnswer index distribution (0-based):")
    print(dict(sorted(ans_idx_counter.items())))
    if missing_gold:
        print(f"⚠️  Rows with missing/unavailable gold answer: {missing_gold} "
              f"(测试集通常不提供金标，这属于正常情况)")

    k = min(3, n)
    idxs = random.sample(range(n), k) if n >= k else list(range(n))
    examples = []
    for i in idxs:
        row = ds[i]
        q = row.get("question", "")
        opts = _get_options(row)
        ai = _get_answer_idx(row, opts)
        gold = opts[ai] if (ai is not None and 0 <= ai < len(opts)) else None
        examples.append({
            "id": row.get("id", i),
            "question": q,
            "num_options": len(opts),
            "options": opts[:6] + (["..."] if len(opts) > 6 else []),
            "answer_idx": ai,
            "gold_answer_text": gold,
        })

    print("\nRandom examples (3):")
    print(json.dumps(examples, ensure_ascii=False, indent=2))

    if n > 0:
        row0 = ds[0]
        q0 = row0.get("question", "")
        opts0 = _get_options(row0)
        ai0 = _get_answer_idx(row0, opts0)
        fmt0 = _format_example(q0, opts0)
        print("\nFormatted example (row 0):")
        print(fmt0)
        gold0 = opts0[ai0] if ai0 is not None and 0 <= ai0 < len(opts0) else "N/A"
        print(f"\nGold index: {ai0}  |  Gold text: {gold0}")


if __name__ == "__main__":
    main()