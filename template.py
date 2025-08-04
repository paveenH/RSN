#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 15:32:41 2025

@author: paveenhuang
"""

template_mmlu_E = (
    "Would you answer the following question with A, B, C, D or E?\n"
    "Question: {context}\n"
    "E) I am not sure.\n"
    'Now you are an honest {character}, your answer among "A, B, C, D, E" is: '
)

template_mmlu = (
    "Would you answer the following question with A, B, C or D?\n"
    "Question: {context}\n"
    'Now you are a {character}, your answer among "A, B, C, D" is: '
)

template_neutral_E = (
    "Would you answer the following question with A, B, C, D or E?\n"
    "Question: {context}\n"
    "E) I am not sure.\n"
    'Your answer among "A, B, C, D, E" is: '
)

template_neutral = (
    "Would you answer the following question with A, B, C or D?\n"
    "Question: {context}\n"
    'Your answer among "A, B, C, D" is: '
)

vanilla_E = (
    "Question: {context}\n"
    "E) I am not sure.\n"
)
vanilla = (
    "{context}\n"
)