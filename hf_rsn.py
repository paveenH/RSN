#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HFLMWithRSN: A HuggingFace LM wrapper for LM Evaluation Harness that injects
per-layer diff vectors via forward hooks to edit hidden states at the tail tokens.

Usage (programmatic):
    from lm_eval import evaluator
    from hf_rsn import HFLMWithRSN
    import numpy as np

    diff = np.load("/path/to/diff.npy")  # shape: [n_layers, H] or [H]
    model = HFLMWithRSN(
        pretrained="meta-llama/Llama-3.1-8B-Instruct",
        rsn_cfg={"diff_matrices": diff, "tail_len": 1},
        # ... any HFLM kwargs
    )
    results = evaluator.simple_evaluate(model=model, tasks=["mmlu"])

Design:
- Overrides _model_call to wrap a single forward with temporary forward_hooks.
- Works for both loglikelihood and generation codepaths since both call _model_call.
- Does not move batch inputs to a single device (safe for device_map="auto").
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

from lm_eval.models.huggingface import HFLM


class HFLMWithRSN(HFLM):
    """
    HFLM subclass that supports neuron/hidden-state editing via forward hooks.
    Config via rsn_cfg:
        - "diff_matrices": np.ndarray or list[np.ndarray]
              Either [n_layers, H] or per-layer list of [H].
              Or a 1D [H] vector to be broadcast to all layers.
        - "tail_len": int, number of last positions per sequence to edit (default 1)
        - "layer_indices": Optional[List[int]], choose subset of decoder layers to edit.
                           If None, apply to all layers; if provided, len must match
                           diff_matrices when diff is a list.
        - "alpha": Optional[float], multiply diffs by alpha (applied once at load time).
    """

    def __init__(self, *args, rsn_cfg: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rsn_cfg: Dict[str, Any] = rsn_cfg or {}
        self._rsn_enabled: bool = bool(self.rsn_cfg)

        # Prepared (lazy) fields
        self._decoder_layers_cache: Optional[List[torch.nn.Module]] = None
        self._rsn_diff_per_layer: Optional[List[np.ndarray]] = None  # numpy, to be tensor-ized inside hook
        self._tail_len: int = int(self.rsn_cfg.get("tail_len", 1))
        self._layer_indices: Optional[List[int]] = self.rsn_cfg.get("layer_indices")

        # Load / normalize diff matrices if provided
        if self._rsn_enabled:
            raw = self.rsn_cfg.get("diff_matrices", None)
            if raw is None:
                raise ValueError("rsn_cfg provided but 'diff_matrices' missing.")
            alpha = float(self.rsn_cfg.get("alpha", 1.0))
            self._rsn_diff_per_layer = self._normalize_diffs(raw, alpha=alpha)

    # --------------- Core HF/Harness plumbing ---------------
    
    def _model_call(self, inputs):
        # ---- normalize inputs to a dict ----
        if isinstance(inputs, torch.Tensor):
            batch = {"input_ids": inputs}
        elif isinstance(inputs, (list, tuple)):
            if len(inputs) == 1 and isinstance(inputs[0], torch.Tensor):
                batch = {"input_ids": inputs[0]}
            elif len(inputs) == 2 and all(isinstance(x, torch.Tensor) for x in inputs):
                batch = {"input_ids": inputs[0], "attention_mask": inputs[1]}
            else:
                if len(inputs) == 1 and isinstance(inputs[0], dict):
                    batch = dict(inputs[0])
                else:
                    raise TypeError(f"Unsupported _model_call inputs type/structure: {type(inputs)}")
        elif isinstance(inputs, dict):
            batch = dict(inputs)
        else:
            raise TypeError(f"Unsupported _model_call inputs type: {type(inputs)}")

        # ---- RSN off: baseline path ----
        if not self._rsn_enabled:
            with torch.no_grad():
                return self._forward_outputs(**batch)

        # ---- get / synthesize attention_mask ----
        attn = batch.get("attention_mask", None)
        if attn is None:
            input_ids = batch["input_ids"]
            attn = torch.ones_like(input_ids, dtype=torch.long)
            batch["attention_mask"] = attn

        last_idx = attn.sum(dim=1).to(dtype=torch.long) - 1  # [B]

        # ---- per-layer hooks ----
        decoder_layers = self._get_decoder_layers()
        diffs = self._align_diffs_to_layers(self._rsn_diff_per_layer, decoder_layers)

        hooks = []
        try:
            for layer, diff in zip(decoder_layers, diffs):
                h = layer.register_forward_hook(
                    self._make_tail_add_hook(diff_matrix=diff, last_indices=last_idx, tail_len=self._tail_len)
                )
                hooks.append(h)

            with torch.no_grad():
                return self._forward_outputs(**batch)
        finally:
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass
    

    def _forward_outputs(self, **model_inputs) -> torch.Tensor:
        """
        Unified forward that returns logits tensor (both for scoring and generation priming).
        Keep return_dict True to be safe across models; extract .logits.
        """
        outputs = self.model(**model_inputs, return_dict=True, use_cache=False)
        # Some models may not return logits in edge cases
        logits = getattr(outputs, "logits", None)
        if logits is None:
            # Fallback: certain models might return tensor directly
            if isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                raise RuntimeError("Model forward produced no logits.")
        return logits

    # --------------- RSN utilities ---------------

    def _get_decoder_layers(self) -> List[torch.nn.Module]:
        """
        Heuristic discovery of decoder blocks across popular HF decoder-only models.
        Caches the list for reuse.
        """
        if self._decoder_layers_cache is not None:
            return self._decoder_layers_cache

        candidates: List[List[torch.nn.Module]] = []

        # Common paths across families
        paths = [
            "model.layers",               # LLaMA/Mistral
            "model.decoder.layers",       # some seq2seq decoders
            "transformer.h",              # GPT-2/GPT-NeoX style
            "gpt_neox.layers",            # GPT-NeoX
            "model.gpt_neox.layers",
        ]

        named = dict(self.model.named_modules())
        for p in paths:
            if p in named:
                seq = []
                # children of this container in order
                for name, module in self.model.named_modules():
                    if name.startswith(p + ".") and name.count(".") == p.count(".") + 1:
                        seq.append(module)
                if seq:
                    candidates.append(seq)

        if not candidates:
            # Last-resort: scan shallow module tree for a list-like container of uniform blocks
            layers = []
            for name, module in self.model.named_modules():
                # Heuristic: block modules often end with ".layers" or ".h"
                if name.endswith(".layers") or name.endswith(".h"):
                    # collect its direct children
                    sub = []
                    for n2, m2 in self.model.named_modules():
                        if n2.startswith(name + ".") and n2.count(".") == name.count(".") + 1:
                            sub.append(m2)
                    if sub:
                        candidates.append(sub)

        if not candidates:
            # Print a short tree for debugging
            print("Decoder layer discovery failed. Top-level modules:")
            for i, (n, m) in enumerate(self.model.named_children()):
                print(f"  [{i}] {n} -> {m.__class__.__name__}")
            raise ValueError("No decoder layers found. Please adapt _get_decoder_layers() for your model.")

        # Prefer the first candidate (most specific path first)
        layers = candidates[0]
        self._decoder_layers_cache = layers
        return layers

    def _normalize_diffs(self, raw: Any, alpha: float = 1.0) -> List[np.ndarray]:
        """
        Normalize user-provided diffs to a per-layer list of 1D numpy arrays [H].
        - raw can be np.ndarray [n_layers, H], or [H], or list of [H]
        - applies alpha scaling
        """
        if isinstance(raw, np.ndarray):
            if raw.ndim == 1:
                arrs = [raw.astype(np.float32) * alpha]  # broadcast later
            elif raw.ndim == 2:
                arrs = [(raw[i].astype(np.float32) * alpha) for i in range(raw.shape[0])]
            else:
                raise ValueError(f"Unsupported diff ndarray shape {raw.shape}, expect [H] or [n_layers,H].")
        elif isinstance(raw, list):
            arrs = []
            for i, x in enumerate(raw):
                x = np.asarray(x)
                if x.ndim != 1:
                    raise ValueError(f"diff list item {i} has shape {x.shape}, expect 1D [H].")
                arrs.append(x.astype(np.float32) * alpha)
        else:
            raise ValueError("diff_matrices must be a numpy array or a list of 1D arrays.")
        return arrs

    def _align_diffs_to_layers(self, diffs: List[np.ndarray], layers: List[torch.nn.Module]) -> List[np.ndarray]:
        """
        Broadcast / select diffs so that len(return) == len(layers_to_edit).
        If layer_indices is provided, we edit only those indices.
        """
        # Choose which layers to edit
        if self._layer_indices is not None:
            # select subset of layers
            edit_layers = [layers[i] for i in self._layer_indices]
        else:
            edit_layers = layers

        # Align the number of diff vectors
        if len(diffs) == 1:
            out = [diffs[0]] * len(edit_layers)
        elif len(diffs) == len(edit_layers):
            out = diffs
        elif len(diffs) == len(layers) and self._layer_indices is not None:
            # Provided per-all-layers diffs, but we only edit a subset
            out = [diffs[i] for i in self._layer_indices]
        else:
            raise ValueError(
                f"Mismatch between diffs ({len(diffs)}) and edit layers ({len(edit_layers)}); "
                f"either provide a single [H] vector or a per-layer list matching the target layers."
            )
        return out

    def _make_tail_add_hook(
        self,
        diff_matrix: np.ndarray,
        last_indices: torch.Tensor,
        tail_len: int = 1,
    ):
        """
        Build a forward hook that adds diff_matrix onto the last 'tail_len' hidden positions.
        diff_matrix: numpy [H]
        last_indices: torch.LongTensor [B]
        """
        # cache by (device, dtype)
        _cache: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}

        def get_diff_1h(hs: torch.Tensor) -> torch.Tensor:
            key = (hs.device, hs.dtype)
            if key not in _cache:
                dt = torch.as_tensor(diff_matrix, device=hs.device, dtype=hs.dtype)
                if dt.ndim != 1:
                    raise ValueError(f"Expected 1D [H] diff, got {tuple(dt.shape)}")
                _cache[key] = dt
            return _cache[key]  # [H]

        def add_at_tail(hs: torch.Tensor) -> torch.Tensor:
            # hs: [B, L, H]
            B, L, H = hs.shape
            n = max(int(tail_len), 1)

            # positions to modify: last, last-1, ..., last-(n-1)
            last_pos = last_indices.to(device=hs.device, dtype=torch.long)  # [B]
            offs = torch.arange(n, device=hs.device, dtype=torch.long)      # [n]
            pos_raw = last_pos.unsqueeze(1) - offs.unsqueeze(0)             # [B, n]
            valid_mask = (pos_raw >= 0)                                     # [B, n]
            pos = pos_raw.clamp_min(0)                                      # [B, n]

            # build [B, n, H] diff, mask invalid
            diff_bh = get_diff_1h(hs).unsqueeze(0).expand(B, -1)            # [B, H]
            diff_bnh = diff_bh.unsqueeze(1).expand(B, n, H)                 # [B, n, H]
            diff_bnh = diff_bnh * valid_mask.unsqueeze(-1)                  # zero out invalid

            # in-place scatter_add on L dimension
            index = pos.unsqueeze(-1).expand(B, n, H)                        # [B, n, H]
            hs.scatter_add_(dim=1, index=index, src=diff_bnh)
            return hs

        def hook(module, inputs, output):
            # module: decoder layer block; output: Tensor or tuple(Tensor, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
                hidden_states = add_at_tail(hidden_states)
                return (hidden_states,) + output[1:]
            else:
                hidden_states = output
                hidden_states = add_at_tail(hidden_states)
                return hidden_states

        return hook