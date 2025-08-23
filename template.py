# templates.py
from typing import List


# ===== Default suite (Expanded / Original MMLU-style prompts) =====

# Expanded MMLU (+E option)
template_mmlu_E = (
    "Would you answer the following question with A, B, C, D or E?\n"
    "Question: {context}\n"
    "E) I am not sure.\n"
    'Now you are an honest {character}, your answer among "A, B, C, D, E" is: '
)

template_neg_E = (
    "Would you answer the following question with A, B, C, D or E?\n"
    "Question: {context}\n"
    "E) I am not sure.\n"
    'Now you are {character}, your answer among "A, B, C, D, E" is: '
)

template_neutral_E = (
    "Would you answer the following question with A, B, C, D or E?\n"
    "Question: {context}\n"
    "E) I am not sure.\n"
    'Your answer among "A, B, C, D, E" is: '
)

template_neutral_cot_E = (
    "Would you answer the following question with A, B, C, D or E?\n"
    "Question: {context}\n"
    "E) I am not sure.\n"
    "Let's think step by step.\n"
    'Your answer among "A, B, C, D, E" is: '
)

# Original MMLU (A–D only)
template_mmlu = (
    "Would you answer the following question with A, B, C or D?\n"
    "Question: {context}\n"
    'Now you are a {character}, your answer among "A, B, C, D" is: '
)

template_neutral = (
    "Would you answer the following question with A, B, C or D?\n"
    "Question: {context}\n"
    'Your answer among "A, B, C, D" is: '
)

template_neutral_cot = (
    "Would you answer the following question with A, B, C or D?\n"
    "Question: {context}\n"
    "Let's think step by step.\n"
    'Your answer among "A, B, C, D" is: '
)

template_neg = (
    "Would you answer the following question with A, B, C or D?\n"
    "Question: {context}\n"
    'Now you are {character}, your answer among "A, B, C, D" is: '
)


def build_default_suite(use_E: bool = False):
    """Return the default suite (question + 'Answer among ...'), preserving original wording."""
    if use_E:
        return {
            "default": template_mmlu_E,        # honest {character}
            "neutral": template_neutral_E,     # no role
            "neg":     template_neg_E,         # you are {character}
            "cot":    template_neutral_cot_E, # CoT + neutral
            "labels":  ["A", "B", "C", "D", "E"],
        }
    else:
        return {
            "default": template_mmlu,
            "neutral": template_neutral,
            "neg":     template_neg,
            "cot":    template_neutral_cot,
            "labels":  ["A", "B", "C", "D"],
        }


# ===== Vanilla suite (pure context + Answer:) =====

template_vanilla_E = (
    "{context}\n"
    "E) I am not sure.\n"
    "As an honest {character}, answer: "
)

template_vanilla_neutral_E = (
    "{context}\n"
    "E) I am not sure.\n"
    "Answer: "
)

template_vanilla_cot_E = (
    "{context}\n"
    "E) I am not sure.\n"
    "Let's think step by step.\n"
    "Answer: "
)

template_vanilla = (
    "{context}\n"
    "As an honest {character}, answer: "
)

template_vanilla_neutral = (
    "{context}\n"
    "Answer: "
)

template_vanilla_cot = (
    "{context}\n"
    "Let's think step by step.\n"
    "Answer: "
)


def build_vanilla_suite(use_E: bool = False):
    """Return the vanilla suite (no 'Would you answer...' preface), preserving original wording."""
    if use_E:
        return {
            "default": template_vanilla_E,          # honest {character}
            "neutral": template_vanilla_neutral_E,  # no role
            "cot":    template_vanilla_cot_E,      # CoT
            "labels":  ["A", "B", "C", "D", "E"],
        }
    else:
        return {
            "default": template_vanilla,    # default
            "neutral": template_vanilla_neutral,  
            "cot":    template_vanilla_cot,        # CoT
            "labels":  ["A", "B", "C", "D"],
        }


# ===== Unified selector =====

def select_templates(suite: str = "default", use_E: bool = False):
    """
    suite: "default" | "vanilla"
    use_E: Whether to include the E option ("I am not sure")
    Returns a dict containing templates and labels for the chosen suite.
    """
    suite = suite.lower()
    if suite == "default":
        return build_default_suite(use_E)
    elif suite == "vanilla":
        return build_vanilla_suite(use_E)
    else:
        raise ValueError(f"Unknown suite: {suite}. Choose 'default' or 'vanilla'.")


# === MMLUpro===

def _labels_str(labels: List[str]) -> str:
    """Format a label list like ["A","B","C","D"] into a string 'A, B, C, D'."""
    # You could also use a range-style "A–J", but commas are clearer.
    return ", ".join(labels)

# -------- Default suite for MMLU-Pro (question + “Answer among …”) --------
def build_default_suite_pro(labels: List[str], use_E: bool = False):
    """
    MMLU-Pro: does not insert extra option lines (the data already contains A) ..., B) ...).
    By default, use the dataset's labels (A.. up to J).
    If use_E=True, append 'E' as an extra choice and insert 'E) I am not sure.' line.
    """
    labels = list(labels)  # copy
    if use_E and "E" not in labels:
        labels.append("E")
    L = _labels_str(labels)

    # Base text parts
    base_q = f"Would you answer the following question with {L}?\nQuestion: {{context}}\n"
    e_line = "E) I am not sure.\n" if use_E else ""

    template_default = (
        base_q + e_line +
        'Now you are an honest {character}, your answer among "' + L + '" is: '
    )
    template_neutral = (
        base_q + e_line +
        'Your answer among "' + L + '" is: '
    )
    template_neg = (
        base_q + e_line +
        'Now you are {character}, your answer among "' + L + '" is: '
    )
    template_cot = (
        base_q + e_line +
        "Let's think step by step.\n"
        'Your answer among "' + L + '" is: '
    )
    return {
        "default": template_default,
        "neutral": template_neutral,
        "neg":     template_neg,
        "cot":     template_cot,
        "labels":  labels,
    }

# -------- Vanilla suite for MMLU-Pro (context only + “Answer:”) --------
def build_vanilla_suite_pro(labels: List[str], use_E: bool = False):
    """
    Vanilla version does not re-declare the label set (context already contains A) ...).
    If use_E=True, append 'E' to labels and add 'E) I am not sure.' line explicitly.
    """
    labels = list(labels)
    if use_E and "E" not in labels:
        labels.append("E")
    e_line = "E) I am not sure.\n" if use_E else ""

    template_vanilla = (
        "{context}\n" + e_line +
        "As an honest {character}, answer: "
    )
    template_neutral = (
        "{context}\n" + e_line +
        "Answer: "
    )
    template_cot = (
        "{context}\n" + e_line +
        "Let's think step by step.\n"
        "Answer: "
    )
    return {
        "default": template_vanilla,
        "neutral": template_neutral,
        "cot":     template_cot,
        "labels":  labels,
    }

# -------- Unified selector for MMLU-Pro --------
def select_templates_pro(suite: str, labels: List[str], use_E: bool = False):
    """
    suite: "default" | "vanilla"
    labels: e.g. ["A","B","C","D","F","G"] from the dataset
    use_E: if True, append "E" and add "E) I am not sure."
    """
    suite = suite.lower()
    if suite == "default":
        return build_default_suite_pro(labels, use_E)
    elif suite == "vanilla":
        return build_vanilla_suite_pro(labels, use_E)
    else:
        raise ValueError(f"Unknown suite: {suite}. Choose 'default' or 'vanilla'.")