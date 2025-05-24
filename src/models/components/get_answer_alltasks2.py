import os
import json
import re
from tqdm import tqdm

from vicuna import VicundaModel      # Your model wrapper

# ────────────────────── ① Configuration ──────────────────────────────────────────────
TASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_medicine",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions"
]

MODEL      = "falcon3"
SIZE       = "10B"
NUM_GPUS   = 1

PATH_MMLU  = "/data2/paveen/RolePlaying/src/models/components/mmlu"
SAVE_BASE  = "/data2/paveen/RolePlaying/src/models/components/answer_phi"
MODEL_DIR  = f"/data2/paveen/RolePlaying/shared/{MODEL}/{SIZE}"

LABELS     = ["A", "B", "C", "D"]          # Option labels

# ────────────────────── ② Character Set ────────────────────────────────────────────
def make_characters(task_name: str):
    task_name = task_name.replace("_", " ")
    return [
        f"non-{task_name}",
        f"{task_name}",
        # f"{task_name} expert",
        # "person",
    ]

# ────────────────────── ③ General Cleaning / Extraction / Generation ────────────────────────────
RE_ASSISTANT = re.compile(r"<\|?assistant\|?>", re.I)
RE_PAREN     = re.compile(r"\b([A-E])\s*\)", re.I)   # D) / (D
RE_SINGLE    = re.compile(r"\b([A-E])\b")

def sanitize(text: str) -> str:
    """Remove special tags and excessive spaces, convert to uppercase"""
    text = RE_ASSISTANT.sub("", text)
    text = text.strip().replace("\n", " ")
    return text.upper()

def extract_choice(raw: str):
    """Try to extract A~E; return None if fails"""
    txt = sanitize(raw)
    m   = RE_PAREN.search(txt) or RE_SINGLE.search(txt)
    return m.group(1) if m else None

def generate_choice(vc, prompt,
                    short_tokens: int = 6,
                    long_tokens:  int = 8):
    """
    First attempt with short generation → extraction; rescue with long generation if failed.
    Returns:
      - "A"/"B"/"C"/"D"       → Valid answer
      - "E"                   → Skip option
      - "[ADD]..." / "[INV]..."  → Debug text
    """
    # First attempt
    out   = vc.generate([prompt], max_new_tokens=short_tokens)[0]
    pick  = extract_choice(out)
    if pick:               return pick
    if "I AM NOT SURE" in sanitize(out):  return "E"

    # Rescue attempt
    out2  = vc.generate([prompt], max_new_tokens=long_tokens)[0]
    pick2 = extract_choice(out2)
    if pick2:              return f"[ADD]{pick2} ORIGINAL:{sanitize(out2)}"
    if "I AM NOT SURE" in sanitize(out2): return "E"

    return f"[INV]{sanitize(out2)}"

# ────────────────────── ④ Other Utility Functions ─────────────────────────────────────
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_correct_text(question: str, label_idx: int):
    prefix = f"{LABELS[label_idx]})"
    for line in question.split("\n"):
        s = line.strip()
        if s.upper().startswith(prefix):
            return s[len(prefix):].strip().lower()
    return None

def update(stat, ch, tag):
    stat[ch][tag] += 1
    stat[ch]["total"] += 1

# ────────────────────── ⑤ Run Task Once ─────────────────────────────────────
def run_task(vc, template, task_name: str):
    samples = load_json(os.path.join(PATH_MMLU, f"{task_name}.json"))
    chars   = make_characters(task_name)
    stat    = {c: {"correct":0, "E":0, "invalid":0, "total":0} for c in chars}

    for idx, s in enumerate(tqdm(samples, desc=task_name)):
        ctx, gold_idx = s["text"], s["label"]
        if not (0 <= gold_idx < len(LABELS)):  continue
        gold_label = LABELS[gold_idx]
        # gold_text  = extract_correct_text(ctx, gold_idx)

        for ch in chars:
            prompt = template.format(character=ch, context=ctx)
            ans    = generate_choice(vc, prompt)

            # Determine label
            if ans in LABELS:                     tag = "correct" if ans == gold_label else "invalid"
            elif ans == "E":                      tag = "E"
            else:                                 tag = "invalid"

            # Statistics & recording
            update(stat, ch, tag)
            s[f"answer_{ch.replace(' ','_')}"] = ans

            # Optional debug output
            if tag == "invalid":
                tqdm.write(f"Sample {idx}, Char '{ch}': invalid '{ans}'")

            # If rescue hit correct or E, prompt
            if ans.startswith("[ADD]") and tag == "correct":
                tqdm.write(f"[{idx}][{ch}] salvage hit -> Correct")

    # Summarize as percentage
    summary = {
        ch: {
            **v,
            "accuracy%": round(100 * v["correct"] / v["total"], 2) if v["total"] else 0
        } for ch, v in stat.items()
    }
    return samples, summary

# ────────────────────── ⑥ Main Process ───────────────────────────────────────────
def main():
    print(f"Loading model  {MODEL}/{SIZE} ...")
    vc       = VicundaModel(model_path=MODEL_DIR, num_gpus=NUM_GPUS)
    template = vc.template
    save_dir = os.path.join(SAVE_BASE, MODEL); os.makedirs(save_dir, exist_ok=True)

    for task in TASKS:
        print(f"\n=== {task} ===")
        data, acc = run_task(vc, template, task)

        # Save results
        fn = os.path.join(save_dir, f"{task}_{SIZE}_answers.json")
        with open(fn, "w", encoding="utf-8") as f:
            json.dump({"data": data, "accuracy": acc}, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {fn}")

        # Print summary
        for ch, r in acc.items():
            print(f"{ch:>22}: {r['accuracy%']}% "
                  f"(✔ {r['correct']}/{r['total']}  "
                  f"E {r['E']}  ✗ {r['invalid']})")

    print("\nAll tasks finished!")

if __name__ == "__main__":
    main()