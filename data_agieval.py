"""
AGIEval (MCQ only) → unified MMLU-Pro-like JSON (top-level list).

- Auto-discovers tasks under data/v1_1 via GitHub API
- Skips Cloze/fill-in tasks: ["gaokao-math-cloze", "MATH"]
- Downloads train/dev/test if present (jsonl or json)
- Builds items:
    {
      "task": "AGIEval-<task_name>",
      "text": "<passage?>\\n\\n<question>\\nA) ...\\nB) ...\\n...",
      "label": <0-based correct index>,
      "num_options": K
    }
- By default preserves original option order (no shuffle)

Usage:
  python agieval_mcq_to_mmlupro.py
"""

import os
import json
import time
import random
import urllib.request
from typing import List, Dict, Any, Optional

# ========== CONFIG ==========
SAVE_DIR = "/data2/paveen/RolePlaying/components/agieval"
OUT_PATH = os.path.join(SAVE_DIR, "agieval_mcq_all.json")

# Deterministic option shuffling?
SHUFFLE_OPTIONS = False
SEED = 42

# Exclude known cloze/fill-in tasks (non-MCQ)
EXCLUDE_TASKS = {"gaokao-math-cloze", "MATH"}

# API endpoints
API_BASE = "https://api.github.com/repos/ruixiangcui/AGIEval/contents"
V11_DIR  = "data/v1_1"

# Acceptable split filenames to try (in order)
SPLIT_FILES = ["train.jsonl", "dev.jsonl", "test.jsonl",
               "train.json",  "dev.json",  "test.json"]
# ===========================

rnd = random.Random(SEED)
LETTER = [chr(ord("A") + i) for i in range(26)]

def _http_get(url: str, retry: int = 3, sleep: float = 1.0) -> bytes:
    last_err = None
    for _ in range(retry):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except Exception as e:
            last_err = e
            time.sleep(sleep)
    raise last_err

def _json_get(url: str) -> Any:
    data = _http_get(url)
    return json.loads(data.decode("utf-8", errors="replace"))

def _list_dir(path: str) -> List[Dict[str, Any]]:
    """
    List a GitHub repo path via Contents API.
    Returns a list of dicts with keys including: name, type, download_url, url
    """
    url = f"{API_BASE}/{path}"
    obj = _json_get(url)
    if not isinstance(obj, list):
        return []
    return obj

def _download_text(url: str) -> str:
    b = _http_get(url)
    # Try UTF-8 first
    return b.decode("utf-8", errors="replace")

def _is_mcq_record(rec: Dict[str, Any]) -> bool:
    # MCQ records should have question + options + label (letter)
    return ("question" in rec) and ("options" in rec) and ("label" in rec)

def _label_to_index(label: Any, nopt: int) -> Optional[int]:
    """
    Convert label to 0-based index. Label is typically a letter "A/B/C/D".
    """
    if label is None:
        return None
    if isinstance(label, int):
        return label if 0 <= label < nopt else None
    s = str(label).strip()
    # Allow forms: "A", "(A)", "A)", "(A) xxx"
    if s:
        # take first alpha char as the letter
        for ch in s:
            if "A" <= ch <= "Z":
                idx = ord(ch) - ord("A")
                return idx if 0 <= idx < nopt else None
    return None

def _build_text(passage: Optional[str], question: str, options: List[str]) -> str:
    lines: List[str] = []
    passage = (passage or "").strip()
    if passage:
        lines.append(passage)
        lines.append("")  # blank line
    lines.append((question or "").strip())
    for i, opt in enumerate(options):
        lines.append(f"{LETTER[i]}) {str(opt).strip()}")
    return "\n".join(lines) + "\n"

def _parse_jsonl(text: str) -> List[Dict[str, Any]]:
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            # skip malformed lines
            continue
    return out

def _load_split_from_download_url(url: str) -> List[Dict[str, Any]]:
    raw = _download_text(url)
    # Heuristic: jsonl vs json
    if "\n" in raw and raw.strip().startswith("{") is False and raw.strip().startswith("[") is False:
        # likely JSONL (each line is a JSON object)
        return _parse_jsonl(raw)
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj
        elif isinstance(obj, dict):
            # some repos may wrap in {"data":[...]}
            if "data" in obj and isinstance(obj["data"], list):
                return obj["data"]
    except Exception:
        pass
    # fallback: try line-by-line parse
    return _parse_jsonl(raw)

def _collect_items_from_task(task_name: str, task_api_url: str) -> List[Dict[str, Any]]:
    """
    For one task folder (e.g., data/v1_1/lsat-ar), try to read train/dev/test.
    Build MMLU-Pro-like items.
    """
    items: List[Dict[str, Any]] = []
    listing = _json_get(task_api_url)  # list of files in this folder

    # Map file name → download_url for quick lookup
    filemap = {ent["name"]: ent.get("download_url") for ent in listing if ent.get("type") == "file"}

    for fname in SPLIT_FILES:
        dl = filemap.get(fname)
        if not dl:
            continue

        records = _load_split_from_download_url(dl)
        for rec in records:
            if not isinstance(rec, dict):
                continue
            if not _is_mcq_record(rec):
                # skip non-MCQ (e.g., cloze items with "answer")
                continue

            question = rec.get("question", "")
            options_raw = rec.get("options") or []
            options = [str(x).strip() for x in options_raw if str(x).strip()]
            if len(options) < 2 or not question:
                continue

            # Some tasks also include a 'passage' (may be None)
            passage = rec.get("passage", None)

            # Label → index
            gold = _label_to_index(rec.get("label"), len(options))
            if gold is None:
                continue

            # Optional deterministic shuffle
            if SHUFFLE_OPTIONS:
                perm = list(range(len(options)))
                rnd.shuffle(perm)
                options_shuf = [options[j] for j in perm]
                label = perm.index(gold)
            else:
                options_shuf = options
                label = gold

            text = _build_text(passage, question, options_shuf)
            items.append({
                "task": f"AGIEval-{task_name}",
                "text": text,
                "label": int(label),
                "num_options": len(options_shuf),
            })
    return items

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1) List tasks under data/v1_1
    root_entries = _list_dir(V11_DIR)
    task_dirs = [ent for ent in root_entries if ent.get("type") == "dir"]
    task_names = [ent["name"] for ent in task_dirs if ent.get("name")]

    # 2) Filter out non-MCQ tasks
    task_names = [t for t in task_names if t not in EXCLUDE_TASKS]

    merged: List[Dict[str, Any]] = []
    per_task_count: Dict[str, int] = {}

    for t in sorted(task_names):
        api_url = f"{API_BASE}/{V11_DIR}/{t}"
        try:
            items = _collect_items_from_task(t, api_url)
        except Exception as e:
            print(f"[WARN] Skip task {t} due to error: {e}")
            continue

        if items:
            merged.extend(items)
            per_task_count[t] = len(items)
            print(f"[INFO] {t}: {len(items)} items")
        else:
            print(f"[INFO] {t}: 0 items (no MCQ splits found)")

    # 3) Save (top-level list)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    total = sum(per_task_count.values())
    print(f"✅ Saved AGIEval MCQ merged → {OUT_PATH}")
    print(f"[INFO] Total items: {total}")
    if per_task_count:
        top5 = sorted(per_task_count.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print("[INFO] Top tasks:", ", ".join(f"{k}:{v}" for k, v in top5))

    if merged:
        print("[Preview top-2]")
        print(json.dumps(merged[:2], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()