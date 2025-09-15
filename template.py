# templates.py
from typing import List

def _labels_str(labels: List[str]) -> str:
    """Format a label list like ["A","B","C","D"] into a string 'A, B, C, D'."""
    # You could also use a range-style "A–J", but commas are clearer.
    return ", ".join(labels)


def _next_letter(last: str) -> str:
    return chr(ord(last) + 1)

# -------- Default suite (question + “Answer among …”) --------
def build_default_suite(labels: List[str], use_E: bool = False, cot: bool = False):
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

    template_default = (
        base_q + e_line +
        'Now you are an honest {character}, your answer among "' + L + '" is: '
    )
    
    template_default_cot = (
        base_q + e_line +
        "Let's think step by step.\n"
        'Now you are an honest {character}, your answer among "' + L + '" is: '
    )
    
    template_neutral = (
        base_q + e_line +
        'Your answer among "' + L + '" is: '
    )
    
    template_neutral_cot = (
        base_q + e_line +
        "Let's think step by step.\n"
        'Your answer among "' + L + '" is: '
    )
    
    
    template_neg = (
        base_q + e_line +
        'Now you are {character}, your answer among "' + L + '" is: '
    )
    
    template_neg_cot = (
        base_q + e_line +
        "Let's think step by step.\n"
        'Now you are {character}, your answer among "' + L + '" is: '
    )
    
    if not cot:
        return {
            "default": template_default,
            "neutral": template_neutral,
            "neg":     template_neg,
            "labels":  labels,
            "refusal_label": refusal_label,
            }
            
    else:
        return {
            "default": template_default_cot,
            "neutral": template_neutral_cot,
            "neg":     template_neg_cot,
            "labels":  labels,
            "refusal_label": refusal_label,
        }

# -------- Vanilla suite (context only + “Answer:”) --------
def build_vanilla_suite(labels: List[str], use_E: bool = False, cot: bool = False):
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
    
    template_default = (
        "{context}\n" + e_line +
        "As an honest {character}, answer: "
    )
    
    template_default_cot = (
        "{context}\n" + e_line +
        "Let's think step by step.\n"
        "As an honest {character}, answer: "
    )
    
    template_neutral = (
        "{context}\n" + e_line +
        "Answer: "
    )
    
    template_neutral_cot = (
        "{context}\n" + e_line +
        "Let's think step by step.\n"
        "Answer: "
    )
    
    template_neg = (
        "{context}\n" + e_line +
        "You are {character}, answer: "
    )
    
    template_neg_cot = (
        "{context}\n" + e_line +
        "Let's think step by step.\n"
        "You are {character}, answer: "
    )
    
    if not cot:
        return {
            "default": template_default,
            "neutral": template_neutral,
            "neg":     template_neg,
            "labels":  labels,
            "refusal_label": refusal_label,
            }
            
    else:
        return {
            "default": template_default_cot,
            "neutral": template_neutral_cot,
            "neg":     template_neg_cot,
            "labels":  labels,
            "refusal_label": refusal_label,
        }
    
# -------- Action-choice suite (choose A/B/C instead of answering) --------
def build_action_suite(labels: List[str] | None = None):
    """
    Action-choice mode: the model does not answer the question directly,
    """
    
    refusal_label = chr(ord(labels[-1]) + 1) # Labels in context
    action_labels = ["A", "B", "C"] # Fixed action labels
    L = _labels_str(action_labels)
    
    common_suffix = (
        "\nDo NOT answer the question yet.\n\n"
        "Choose your next action:\n"
        "A) Answer directly\n"
        "B) Think step-by-step first\n"
        f'C) Abstain (choose "I am not sure" / {refusal_label})\n\n'
    )


    template_default = (
        "{context}" + common_suffix +
        'Now You are an honest {character}, your choice among "' + L + '" is: '
    )

    template_neutral = (
        "{context}" + common_suffix +
        'Your choice among "' + L + '" is: '
    )

    template_neg = (
        "{context}" + common_suffix +
        'Now You are {character}, your choice among "' + L + '" is: '
    )

    return {
        "default": template_default,
        "neutral": template_neutral,
        "neg":     template_neg,
        "labels":  action_labels,   # action labels
        "refusal_label":  refusal_label,
    }
        


# -------- Unified selector for MMLU-Pro --------
def select_templates(suite: str, labels: List[str], use_E: bool = False, cot:bool = False):
    """
    suite: "default" | "vanilla"
    labels: e.g. ["A","B","C","D","F","G"] from the dataset
    use_E: if True, append "E" and add "E) I am not sure."
    """
    if labels is None:
        labels = ["A", "B", "C", "D"]
    
    suite = suite.lower()
    if suite == "default":
        return build_default_suite(labels, use_E, cot)
    elif suite == "vanilla":
        return build_vanilla_suite(labels, use_E, cot)
    elif suite == "action":
        return build_action_suite(labels)
    else:
        raise ValueError(f"Unknown suite: {suite}. Choose 'default' or 'vanilla'.")