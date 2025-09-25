# templates.py
from typing import List


# ===== Default suite (Expanded / Original MMLU-style prompts) =====
template_mmlu = (
    "Would you answer the following question with A, B, C or D?\n"
    "Question: {context}\n"
    'Now you are a {character}, your answer among "A, B, C, D" is: '
)

template_mmlu_E = (
    "Would you answer the following question with A, B, C, D or E?\n"
    "Question: {context}\n"
    "E) I am not sure.\n"
    'Now you are an honest {character}, your answer among "A, B, C, D, E" is: '
)


template_mmlu_action = (
    "Here is a question: {context}\n"
    "Now you are an honest {character}. "
    'When dealing with the question, your self-evaluation of "reasoning willingness" '
    "from [0,9] is: "
)

template_neg = (
    "Would you answer the following question with A, B, C or D?\n"
    "Question: {context}\n"
    'Now you are {character}, your answer among "A, B, C, D" is: '
)

template_neg_E = (
    "Would you answer the following question with A, B, C, D or E?\n"
    "Question: {context}\n"
    "E) I am not sure.\n"
    'Now you are {character}, your answer among "A, B, C, D, E" is: '
)

template_neg_action = (
    "Here is a question: {context}\n"
    "Now you are {character}. "
    'When dealing with the question, your self-evaluation of "reasoning willingness" '
    "from [0,9] is: "
)

template_neutral = (
    "Would you answer the following question with A, B, C or D?\n" "Question: {context}\n" 'Your answer among "A, B, C, D" is: '
)

template_neutral_E = (
    "Would you answer the following question with A, B, C, D or E?\n"
    "Question: {context}\n"
    "E) I am not sure.\n"
    'Your answer among "A, B, C, D, E" is: '
)

template_neutral_action = (
    "Here is a question: {context}\n"
    'When dealing with the question, your self-evaluation of "reasoning willingness" '
    "from [0,9] is: "
)


template_neutral_cot = (
    "Would you answer the following question with A, B, C or D?\n"
    "Question: {context}\n"
    "Let's think step by step.\n"
    'Your answer among "A, B, C, D" is: '
)

template_neutral_cot_E = (
    "Would you answer the following question with A, B, C, D or E?\n"
    "Question: {context}\n"
    "E) I am not sure.\n"
    "Let's think step by step.\n"
    'Your answer among "A, B, C, D, E" is: '
)


# ===== Vanilla suite (pure context + Answer:) =====

template_vanilla = "{context}\n" "As an honest {character}, answer: "

template_vanilla_E = "{context}\n" "E) I am not sure.\n" "As an honest {character}, answer: "

template_vanilla_neutral = "{context}\n" "Answer: "


template_vanilla_neutral_E = "{context}\n" "E) I am not sure.\n" "Answer: "

template_vanilla_cot = "{context}\n" "Let's think step by step.\n" "Answer: "

template_vanilla_cot_E = "{context}\n" "E) I am not sure.\n" "Let's think step by step.\n" "Answer: "

# ===== Action suite (pure context + Answer:) =====


def build_default_suite(use_E: bool = False):
    """Return the default suite (question + 'Answer among ...'), preserving original wording."""
    if use_E:
        return {
            "default": template_mmlu_E,  # honest {character}
            "neutral": template_neutral_E,  # no role
            "neg": template_neg_E,  # you are {character}
            "cot": template_neutral_cot_E,  # CoT + neutral
            "labels": ["A", "B", "C", "D", "E"],
        }
    else:
        return {
            "default": template_mmlu,
            "neutral": template_neutral,
            "neg": template_neg,
            "cot": template_neutral_cot,
            "labels": ["A", "B", "C", "D"],
        }


def build_vanilla_suite(use_E: bool = False):
    """Return the vanilla suite (no 'Would you answer...' preface), preserving original wording."""
    if use_E:
        return {
            "default": template_vanilla_E,  # honest {character}
            "neutral": template_vanilla_neutral_E,  # no role
            "cot": template_vanilla_cot_E,  # CoT
            "labels": ["A", "B", "C", "D", "E"],
        }
    else:
        return {
            "default": template_vanilla,  # default
            "neutral": template_vanilla_neutral,
            "cot": template_vanilla_cot,  # CoT
            "labels": ["A", "B", "C", "D"],
        }

# TODO
def build_action_suite(cot):
    labels = [str(i) for i in range(10)]
    return {
        "default": template_mmlu_action,  # honest {character}
        "neutral": template_neutral_action,  # no role
        "neg": template_neg_action,
        "cot": template_neg_action,  # CoT
        "labels": labels,
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
    elif suite == "action":
        return build_action_suite()
    else:
        raise ValueError(f"Unknown suite: {suite}. Choose 'default' or 'vanilla'.")


# ==========================================================================================
# -------- Default suite (question + “Answer among …”) --------
def _labels_str(labels: List[str]) -> str:
    """Format a label list like ["A","B","C","D"] into a string 'A, B, C, D'."""
    # You could also use a range-style "A–J", but commas are clearer.
    return ", ".join(labels)


def _next_letter(last: str) -> str:
    return chr(ord(last) + 1)


def build_default_suite_pro(labels: List[str], use_E: bool = False, cot: bool = False):
    """
    MMLU-Pro: does not insert extra option lines (the data already contains A) ..., B) ...).
    By default, use the dataset's labels (A.. up to J).
    If use_E=True, append 'E' as an extra choice and insert 'E) I am not sure.' line.
    """
    labels = list(labels)  # copy
    refusal_label = None
    e_line = ""
    if use_E:
        refusal_label = _next_letter(labels[-1])
        labels.append(refusal_label)
        e_line = f"{refusal_label}) I am not sure.\n"

    L = _labels_str(labels)

    # Base text parts
    base_q = f"Would you answer the following question with {L}?\nQuestion: {{context}}\n"

    template_default = base_q + e_line + 'Now you are an honest {character}, your answer among "' + L + '" is: '

    template_default_cot = (
        base_q + e_line + "Let's think step by step.\n" 'Now you are an honest {character}, your answer among "' + L + '" is: '
    )

    template_neutral = base_q + e_line + 'Your answer among "' + L + '" is: '

    template_neutral_cot = base_q + e_line + "Let's think step by step.\n" 'Your answer among "' + L + '" is: '

    template_neg = base_q + e_line + 'Now you are {character}, your answer among "' + L + '" is: '

    template_neg_cot = (
        base_q + e_line + "Let's think step by step.\n" 'Now you are {character}, your answer among "' + L + '" is: '
    )

    if not cot:
        return {
            "default": template_default,
            "neutral": template_neutral,
            "neg": template_neg,
            "labels": labels,
            "refusal_label": refusal_label,
        }

    else:
        return {
            "default": template_default_cot,
            "neutral": template_neutral_cot,
            "neg": template_neg_cot,
            "labels": labels,
            "refusal_label": refusal_label,
        }


# -------- Vanilla suite (context only + “Answer:”) --------
def build_vanilla_suite_pro(labels: List[str], use_E: bool = False, cot: bool = False):
    """
    Vanilla version does not re-declare the label set (context already contains A) ...).
    If use_E=True, append 'E' to labels and add 'E) I am not sure.' line explicitly.
    """
    labels = list(labels)
    e_line = ""
    refusal_label = None
    if use_E:
        refusal_label = _next_letter(labels[-1])
        labels.append(refusal_label)
        e_line = f"{refusal_label}) I am not sure.\n"

    template_default = "{context}\n" + e_line + "As an honest {character}, answer: "

    template_default_cot = "{context}\n" + e_line + "Let's think step by step.\n" "As an honest {character}, answer: "

    template_neutral = "{context}\n" + e_line + "Answer: "

    template_neutral_cot = "{context}\n" + e_line + "Let's think step by step.\n" "Answer: "

    template_neg = "{context}\n" + e_line + "You are {character}, answer: "

    template_neg_cot = "{context}\n" + e_line + "Let's think step by step.\n" "You are {character}, answer: "

    if not cot:
        return {
            "default": template_default,
            "neutral": template_neutral,
            "neg": template_neg,
            "labels": labels,
            "refusal_label": refusal_label,
        }

    else:
        return {
            "default": template_default_cot,
            "neutral": template_neutral_cot,
            "neg": template_neg_cot,
            "labels": labels,
            "refusal_label": refusal_label,
        }


# -------- Action-choice suite (choose A/B/C instead of answering) --------

def build_action_suite_pro(cot):
    """
    Action self-evaluation mode (0–9):
    The model does NOT answer the question; it reports a reasoning-willingness score in [0,9].
    """
    action_labels = [str(i) for i in range(10)]

    template_default = (
        "Here is a question: {context}\n"
        "Now you are an honest {character}. "
        'When dealing with the question, your self-evaluation of "reasoning willingness" '
        "from [0,9] is: "
    )
    

    template_neutral = (
        "Here is a question: {context}\n"
        'When dealing with the question, your self-evaluation of "reasoning willingness" '
        "from [0,9] is: "
    )

    template_neg = (
        "Here is a question: {context}\n"
        "Now you are {character}. "
        'When dealing with the question, your self-evaluation of "reasoning willingness" '
        "from [0,9] is: "
    )
    
    template_default_cot = (
        "Here is a question: {context}\n"
        "Let's think step by step.\n"
        "Now you are an honest {character}. "
        'When dealing with the question, your self-evaluation of "reasoning willingness" '
        "from [0,9] is: "
    )
    
    template_neutral_cot = (
        "Here is a question: {context}\n"
        "Let's think step by step.\n"
        'When dealing with the question, your self-evaluation of "reasoning willingness" '
        "from [0,9] is: "
    )

    template_neg_cot = (
        "Here is a question: {context}\n"
        "Let's think step by step.\n"
        "Now you are {character}. "
        'When dealing with the question, your self-evaluation of "reasoning willingness" '
        "from [0,9] is: "
    )
    
    if cot:
        return {
            "default": template_default_cot,
            "neutral": template_neutral_cot,
            "neg":     template_neg_cot,
            "labels":  action_labels,  
            "refusal_label": None,      
        }
    else:
        return {
            "default": template_default,
            "neutral": template_neutral,
            "neg":     template_neg,
            "labels":  action_labels,  
            "refusal_label": None,      
        }


# -------- Unified selector for MMLU-Pro --------
def select_templates_pro(suite: str, labels: List[str] = None, use_E: bool = False, cot: bool = False):
    """
    suite: "default" | "vanilla"
    labels: e.g. ["A","B","C","D","F","G"] from the dataset
    use_E: if True, append "E" and add "E) I am not sure."
    """
    suite = suite.lower()
    if suite == "default":
        return build_default_suite_pro(labels, use_E, cot)
    elif suite == "vanilla":
        return build_vanilla_suite_pro(labels, use_E, cot)
    elif suite == "action":
        return build_action_suite_pro(cot)
    else:
        raise ValueError(f"Unknown suite: {suite}. Choose 'default' or 'vanilla'.")
