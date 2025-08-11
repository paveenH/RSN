#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 18:34:35 2025

@author: paveenhuang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from transformers import AutoTokenizer
from template import select_templates  

def render_norole_with_E(model_dir: str, context: str) -> str:
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)

    # 1) vanilla E
    templates = select_templates(use_E=True)
    user_text = templates["vanilla"].format(context=context)

    # 2) norole messages
    messages = [{"role": "user", "content": user_text}]

    # 3) chat_template and assistant
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return rendered

def main():
    ap = argparse.ArgumentParser(description="Inspect norole + chat_template(+E) rendered prompt.")
    ap.add_argument("--model_dir", required=True, help="HF model id or local path, e.g. meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--context", default="Question: Which of the following is correct?\nA) ...\nB) ...\nC) ...\nD) ...")
    ap.add_argument("--show_ids", action="store_true", help="同时打印前若干个token ids（可选）")
    args = ap.parse_args()

    text = render_norole_with_E(args.model_dir, args.context)

    print("\n===== RENDERED PROMPT (first 600 chars) =====")
    print(text[:600])
    print("\n===== FULL (repr, shows special chars clearly) =====")
    print(repr(text))

    if args.show_ids:
        tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False, trust_remote_code=True)
        ids = tok(text, add_special_tokens=False).input_ids
        print("\n===== TOKEN IDS (first 128) =====")
        print(ids[:128])

if __name__ == "__main__":
    main()