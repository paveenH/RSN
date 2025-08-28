#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run lm-eval-harness with Vicunda + RSN-style neuron editing.

- Extends HuggingFace HFLM class with forward hooks
- Injects diff matrices into decoder layers
- Supports evaluation on tasks like TruthfulQA MC1/MC2
"""

import argparse, os, json
import numpy as np
import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


class HFLMWithRSN(HFLM):
    def __init__(self, *args, rsn_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._printed = False
        self.rsn_cfg = rsn_cfg or {}
        self._hooks = []


    # Register forward hooks for each decoder layer
    def _register_hooks(self):
        if self._hooks:
            return
        diff_matrices = self.rsn_cfg["diff_matrices"]  # list[np.ndarray] or shape [n_layers, H]
        decoder_layers = self._find_decoder_layers()
        for layer, dm in zip(decoder_layers, diff_matrices):
            h = layer.register_forward_hook(self._make_hook(dm))
            self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # Override forward pass (used for MC tasks)    
    def _model_call(self, inps):
        self._register_hooks()
        try:
            return super()._model_call(inps)
        finally:
            self._remove_hooks()
    

    # Override generate (used for generation tasks)
    def _model_generate(self, context, max_length, eos_token_id):
        self._register_hooks()
        try:
            return super()._model_generate(context, max_length, eos_token_id)
        finally:
            self._remove_hooks()

    # Locate decoder layers (adjust patterns for model family)
    def _find_decoder_layers(self):
        layers = []
        for name, module in self.model.named_modules():
            if (name.startswith("model.layers.") and name.count(".") == 2) or \
               (name.startswith("model.decoder.layers.") and name.count(".") == 3) or \
               (name.startswith("gpt_neox.layers.") and name.count(".") == 2):
                layers.append(module)
        if not layers:
            raise ValueError("No decoder layers found; adjust name patterns.")
        return layers

    # Build a hook to add diff vector to last token hidden state
    def _make_hook(self, diff_matrix_np):
        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output  # [B, L, H]
            B, L, H = hs.shape
            diff = torch.as_tensor(diff_matrix_np, device=hs.device, dtype=hs.dtype)
            if diff.ndim == 1:
                diff = diff.unsqueeze(0).expand(B, -1)  # [B, H]
            # Add diff to the last token (or tail_len tokens if extended)
            last_pos = torch.full((B,), L - 1, device=hs.device, dtype=torch.long)
            add_buf = torch.zeros_like(hs)
            add_buf[torch.arange(B, device=hs.device), last_pos, :] = diff
            hs = hs + add_buf
            return (hs,) + output[1:] if isinstance(output, tuple) else hs
        return hook


def pyify(obj):
    if isinstance(obj, dict):
        return {k: pyify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [pyify(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.dtype):
        return str(obj)
    return obj
    
    
def main():
    
    # 1) baseline
    m0 = HFLM(pretrained=args.model_dir, dtype="float16", device_map="auto")
    res0 = evaluator.simple_evaluate(model=m0, 
                                      tasks=TASKS, 
                                      num_fewshot=0, 
                                      batch_size="auto")
    
    print("ORIGINAL:", res0["results"])
    
    del m0
    torch.cuda.empty_cache()

    # 2) editing
    for cfg in args.configs:
        alpha, start, end = map(int, cfg.split("-"))
        mask_name = f"{args.mask_type}_{args.percentage}_{start}_{end}_{args.size}{'_abs' if args.abs else ''}.npy"
        mask_path = f"/data2/paveen/RolePlaying/components/mask/{args.hs}_non_logits/{mask_name}"

        diff = np.load(mask_path) * alpha
        if diff.ndim == 2:
            diff_list = [diff[i] for i in range(diff.shape[0])]
        else:
            diff_list = diff

        m1 = HFLMWithRSN(
            pretrained=args.model_dir,
            dtype="float16",
            device_map="auto",
            rsn_cfg={"diff_matrices": diff_list, "tail_len": args.tail_len}
        )
        
        res1 = evaluator.simple_evaluate(model=m1, 
                                         tasks=TASKS, 
                                         num_fewshot=0, 
                                         batch_size="auto")
        print(f"EDITED α={alpha}, layers={start}-{end}:", res1["results"])

        # save result
        res0 = pyify(res0)
        res1 = pyify(res1)
        out_dir = f"results/{args.model}_{args.size}"
        os.makedirs(out_dir, exist_ok=True)
        # with open(os.path.join(out_dir, "original.json"), "w") as f:
        #     json.dump(res0, f, indent=2)
        with open(os.path.join(out_dir, f"edited_{alpha}_{start}_{end}.json"), "w") as f:
            json.dump(res1, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TruthfulQA (MC1/MC2) with neuron editing and logits output.")
    parser.add_argument("--model", type=str, default="qwen2.5_base")
    parser.add_argument("--model_dir", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--hs", type=str, default="qwen2.5")
    parser.add_argument("--size", type=str, default="7B")
    parser.add_argument("--percentage", type=float, default=0.5)
    parser.add_argument("--configs", nargs="*", default=["4-16-22"], help="alpha-start-end triplets, e.g., 4-16-22")
    parser.add_argument("--mask_type", type=str, default="nmd", help="Mask type: nmd or random")
    parser.add_argument("--abs", action="store_true")
    parser.add_argument("--tail_len", type=int, default=1, help="Number of last tokens to apply diff")
    args = parser.parse_args()
    
    TASKS = ["truthfulqa_mc1", "truthfulqa_mc2"]
    
    main()