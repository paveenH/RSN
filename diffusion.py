#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 11:29:12 2025

@author: paveenhuang
"""

# diffusion.py
import torch
import numpy as np
import torch.nn.functional as F

def _add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise  = torch.rand_like(logits, dtype=torch.float64)
    g_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / g_noise

def _get_num_transfer(mask_idx, steps):
    mask_num = mask_idx.sum(dim=1, keepdim=True)
    base, rem = mask_num // steps, mask_num % steps
    out = torch.zeros(mask_num.size(0), steps, dtype=torch.int64,
                      device=mask_idx.device) + base
    for i in range(mask_num.size(0)):
        out[i, :rem[i]] += 1
    return out


@torch.no_grad()
def diffusion_generate(
        model,
        prompt_ids: torch.Tensor,          # shape (1, L_prompt)
        gen_len: int            = 128,
        steps: int              = 128,
        block_len: int          = 128,
        temperature: float      = 0.0,
        cfg_scale: float        = 0.0,
        remask: str             = "low_confidence",
        mask_id: int            = 126336,
    ):
        
        # Construct the initial full MASK sequence
        x = torch.full(
            (1, prompt_ids.shape[1] + gen_len),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        x[:, : prompt_ids.shape[1]] = prompt_ids
        prompt_mask = (x != mask_id)

        assert gen_len % block_len == 0
        n_blocks = gen_len // block_len
        assert steps % n_blocks == 0
        inner_steps = steps // n_blocks

        for blk in range(n_blocks):
            blk_mask_idx = (
                x[:, prompt_ids.shape[1] + blk * block_len : prompt_ids.shape[1] + (blk + 1) * block_len]
                == mask_id
            )
            n_transfer = _get_num_transfer(blk_mask_idx, inner_steps)

            for s in range(inner_steps):
                mask_idx = (x == mask_id)
                # ---------- forward ----------
                if cfg_scale > 0.0:
                    uncond = x.clone()
                    uncond[prompt_mask] = mask_id
                    cat_in = torch.cat([x, uncond], dim=0)      # (2, L)
                    logits = model(cat_in).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits
                # ---------- sampling ----------
                logits = _add_gumbel_noise(logits, temperature)
                x_hat  = logits.argmax(dim=-1)

                # Calculate confidence
                if remask == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    conf = p.gather(-1, x_hat.unsqueeze(-1)).squeeze(-1)
                elif remask == "random":
                    conf = torch.rand_like(x_hat, dtype=torch.float32)
                else:
                    raise NotImplementedError(remask)

                # Only allow the current block to participate in remasking
                conf[:, prompt_ids.shape[1] + (blk + 1) * block_len :] = -np.inf

                # Update token
                x_hat = torch.where(mask_idx, x_hat, x)
                conf  = torch.where(mask_idx, conf,  torch.full_like(conf, -np.inf))
                select = torch.zeros_like(x_hat, dtype=torch.bool)
                for b in range(conf.shape[0]):
                    _, idx = torch.topk(conf[b], k=n_transfer[b, s])
                    select[b, idx] = True
                x[select] = x_hat[select]

        return x

@torch.no_grad()
def get_logits(
    model, 
    prompt_ids: torch.LongTensor, 
    gen_length: int = 1, 
    mask_id: int = 126336
) -> torch.Tensor:
    """
    Obtain logits for the masked positions in a single forward pass.

    Args:
        model:       A mask-predictor-style model (e.g., LLaDA).
        prompt_ids:  Tensor of shape (1, L) containing token IDs for the prompt.
        gen_length:  Number of masked tokens to predict (default: 1).
        mask_id:     The token ID used for [MASK] (default: 126336).

    Returns:
        A tensor of shape (1, gen_length, vocab_size) containing the raw logits 
        at the masked positions.
    """
    device = prompt_ids.device
    seq_len = prompt_ids.size(1)

    # 1) Prepare input: prefix with prompt_ids, suffix with gen_length mask tokens
    x = torch.full((1, seq_len + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :seq_len] = prompt_ids

    # 2) Forward pass to get logits
    outputs = model(x)  # should return a ModelOutput with .logits
    logits = outputs.logits  # shape (1, seq_len + gen_length, vocab_size)

    # 3) Extract the logits at the masked positions
    masked_logits = logits[:, seq_len:, :]  # shape (1, gen_length, vocab_size)
    return masked_logits
    