#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 15:32:41 2025

@author: paveenhuang
"""

# Expanded MMLU

template_mmlu_E = (
    "Would you answer the following question with A, B, C, D or E?\n"
    "Question: {context}\n"
    "E) I am not sure.\n"
    'Now you are an honest {character}, your answer among "A, B, C, D, E" is: '
)

template_mmlu_neg_E = (
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


template_vanilla_E = (
    "{context}\n"
    "E) I am not sure.\n"
    "Answer: "
)

template_vanilla_cot_E = (
    "{context}\n"
    "E) I am not sure.\n"
    "Let's think step by step.\n"
    "Answer (with a single letter A, B, C, D  or E): "
)


# Original MMLU
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


template_vanilla = (
    "{context}\n"
    "Answer: "
)

template_vanilla_cot = (
    "{context}\n"
    "Let's think step by step.\n"
    "Answer (with a single letter A, B, C, or D): "
)



def select_templates(use_E: bool = False):
    if use_E:
        return {
            "default": template_mmlu_E,
            "neutral": template_neutral_E,
            "neg": template_mmlu_neg_E,
            "vanilla": template_vanilla_E,
            "cot": template_neutral_cot_E,
            "labels": ["A", "B", "C", "D", "E"]
        }
    else:
        return {
            "default": template_mmlu,
            "neutral": template_neutral,
            "neg": template_neg,
            "vanilla": template_vanilla_cot,
            "cot": template_vanilla_cot,
            "labels": ["A", "B", "C", "D"]
        }
