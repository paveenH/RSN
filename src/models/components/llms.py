#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 10:46:09 2025

@author: paveenhuang
"""

import logging
import torch
import numpy as np
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from fastchat.conversation import get_conv_template
from fastchat.utils import get_gpu_memory
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from transformers.configuration_utils import PretrainedConfig

# Monkey-patch to avoid NoneType quantization_config errors
PretrainedConfig.__repr__ = lambda self: self.__class__.__name__

log = logging.getLogger(__name__)


class VicundaModel:
    """
    Wrapper around a CausalLM to provide a consistent interface,
    support for quantization, multi-GPU loading, and role-based prompts.
    """
    task: str = "text2text-generation"

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        num_gpus: int = None,
        quantized: bool = False,
    ) -> None:
        self.model_path = model_path
        self.system_prompt = self._infer_system_prompt(model_path)
        
        # v3
        self.template = (
            'Would you answer the following question with A, B, C, D or E?\n'
            'Question: {context}\n'
            'E) I am not sure.\n'
            'Now you are an honest {character} expert, your answer among "A, B, C, D, E" is: '
        )

        # Prepare quantization config if needed
        bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            if quantized
            else None
        )

        # Default to single GPU if unspecified
        num_gpus = num_gpus or 1

        # Multi-GPU loading (not recommended for StableLM)
        if num_gpus > 1:
            if quantized:
                log.warning("Multi-GPU quantization not supported. Loading unquantized model.")
            assert device == "cuda", "Multi-GPU only supported on CUDA devices."

            config = AutoConfig.from_pretrained(self.model_path)
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
            model.tie_weights()

            available_memory = get_gpu_memory(num_gpus)
            sorted_ids = sorted(range(len(available_memory)), key=lambda i: -available_memory[i])
            max_memory = {i: f"{int(available_memory[i]*0.95)}GiB" for i in sorted_ids[:num_gpus]}

            self.model = load_checkpoint_and_dispatch(
                model,
                self.model_path,
                device_map="auto",
                max_memory=max_memory,
                no_split_module_classes=["LlamaDecoderLayer"],
            )

        # Single-GPU or default loading
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                quantization_config=bnb_config,
            )
            if not quantized:
                self.model = self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=False,
            trust_remote_code=True,
        )
        self._ensure_padding_token()

    def _infer_system_prompt(self, path: str) -> str | None:
        """
        Determine the system prompt template based on model name.
        """
        lower = path.lower()
        if "vicuna" in lower:
            return "vicuna_v1.1"
        if "koala" in lower:
            return "koala_v1"
        if "llama2" in lower:
            return "llama-2"
        # Llama3 and others default to None
        return None

    def _ensure_padding_token(self) -> None:
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "</s>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        
    def _apply_diff_hooks(self, diff_matrices: list[np.ndarray], forward_fn):
        """
        Helper to register hooks on all decoder layers, apply diff matrices to last token, run forward_fn, then remove hooks.
        """
        decoder_layers = [
            module for name, module in self.model.named_modules()
            if name.startswith("model.layers.") and name.count(".") == 2
        ]
        if not decoder_layers:
            raise ValueError("No decoder layers found. Check naming convention.")
        if len(decoder_layers) != len(diff_matrices):
            raise ValueError(
                f"Mismatch: {len(diff_matrices)} diff matrices vs {len(decoder_layers)} layers"
            )

        def create_hook(diff_matrix):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    idx = hidden_states.shape[1] - 1
                    if diff_matrix.shape[-1] != hidden_states.shape[-1]:
                        raise ValueError(
                            f"Diff hidden_size {diff_matrix.shape[-1]} != {hidden_states.shape[-1]}"
                        )
                    delta = torch.tensor(diff_matrix, device=hidden_states.device).unsqueeze(0)
                    hidden_states[:, idx, :] += delta
                    return (hidden_states,) + output[1:]
                else:
                    idx = output.shape[1] - 1
                    if diff_matrix.shape[-1] != output.shape[-1]:
                        raise ValueError(
                            f"Diff hidden_size {diff_matrix.shape[-1]} != {output.shape[-1]}"
                        )
                    output[:, idx, :] += diff_matrix
                    return output
            return hook

        hooks = [layer.register_forward_hook(create_hook(dm)) for layer, dm in zip(decoder_layers, diff_matrices)]
        try:
            return forward_fn()
        finally:
            for h in hooks:
                h.remove()

    def _apply_replace_hooks(self, replace_matrices: list[np.ndarray], forward_fn, start: int = 0, end: int = None):
        """
        Register hooks on Transformer decoder layers **only** in [start, end) range,
        replacing the last token's hidden state with the given replacement_matrix.

        Args:
            replace_matrices (list[np.ndarray]): shape = (num_layers, hidden_size).
                Typically you'd pass the entire array for all layers, but only the slice
                [start, end) will actually be used for replacement.
            forward_fn (function): The forward pass function (e.g. `generate`) to execute after hooking.
            start (int): Start layer index (0-based, inclusive).
            end (int): End layer index (0-based, exclusive). If None, defaults to total decoder layers.

        Returns:
            The output of forward_fn().
        """
        # 1) Find all decoder layers
        decoder_layers = [
            module for name, module in self.model.named_modules() if name.startswith("model.layers.") and name.count(".") == 2
        ]
        if not decoder_layers:
            for name, module in self.model.named_modules():
                print(name)
            raise ValueError("No decoder layers found in the model. Please check the layer naming convention.")

        num_layers = len(decoder_layers)
        # If end is None, default to the total number of layers
        if end is None or end > num_layers:
            end = num_layers

        # 2) Basic sanity checks
        if len(replace_matrices) < num_layers:
            raise ValueError(
                f"replace_matrices has length {len(replace_matrices)}, "
                f"but we found {num_layers} decoder layers. Need at least >= num_layers."
            )
        if start < 0 or start >= num_layers:
            raise ValueError(f"Invalid start layer index: {start}. Must be in [0, {num_layers-1}].")
        if end <= start:
            raise ValueError(f"Invalid range: start={start}, end={end}. Must have end > start.")

        # 3) Hook factory function: direct replacement
        def create_replace_hook(replace_matrix):
            def hook(module, module_input, module_output):
                # module_output could be a tuple or a single tensor
                if isinstance(module_output, tuple):
                    hidden_states = module_output[0]
                    last_token_idx = hidden_states.shape[1] - 1
                    if replace_matrix.shape[-1] != hidden_states.shape[-1]:
                        raise ValueError(
                            f"Replacement hidden_size ({replace_matrix.shape[-1]}) "
                            f"!= model hidden_size ({hidden_states.shape[-1]})."
                        )
                    # Create a replacement tensor on the GPU with dimensions [1, hidden_size]
                    rep_tensor = torch.tensor(replace_matrix, device=hidden_states.device).unsqueeze(0)
                    # Direct overwrite
                    hidden_states[:, last_token_idx, :] = rep_tensor
                    return (hidden_states,) + module_output[1:]
                else:
                    # If output is a single tensor
                    last_token_idx = module_output.shape[1] - 1
                    if replace_matrix.shape[-1] != module_output.shape[-1]:
                        raise ValueError(
                            f"Replacement hidden_size ({replace_matrix.shape[-1]}) "
                            f"!= model hidden_size ({module_output.shape[-1]})."
                        )
                    rep_tensor = torch.tensor(replace_matrix, device=module_output.device).unsqueeze(0)
                    module_output[:, last_token_idx, :] = rep_tensor
                    return module_output

            return hook

        # 4) Register hooks ONLY for layers in [start, end)
        hooks = []
        for layer_idx in range(num_layers):
            if start <= layer_idx < end:
                # replace_matrices[layer_idx] 是 (hidden_size,)
                layer = decoder_layers[layer_idx]
                rep_matrix = replace_matrices[layer_idx]

                hook = layer.register_forward_hook(create_replace_hook(rep_matrix))
                hooks.append(hook)
            else:
                pass

        # 5) Perform forward pass
        try:
            outputs = forward_fn()
        finally:
            # 6) Remove hooks
            for hook in hooks:
                hook.remove()

        return outputs


    def generate(
        self,
        inputs: list[str],
        max_new_tokens: int = 1,
        top_p: float = 0.9,
        temperature: float = 0.0,
    ) -> list[str]:
        """
        Generate responses for a batch of input prompts.
        """
        do_sample = temperature > 0
        top_p = top_p if do_sample else None
        temperature = temperature if do_sample else None

        results = []
        for prompt in inputs:
            tokens = self.tokenizer([prompt], return_tensors="pt", padding="longest")
            input_ids = tokens.input_ids.to(self.model.device)
            attention_mask = tokens.attention_mask.to(self.model.device)

            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            # Strip input prompt tokens
            gen_ids = output_ids[0][input_ids.shape[1]:]
            text = self.tokenizer.decode(
                gen_ids, skip_special_tokens=True, spaces_between_special_tokens=False
            )
            results.append(text.strip())

        return results
    
    def get_logits(
        self,
        inputs: list[str],
        postfix_token=None,
        llm_start_msg=None,
        character=None,
        change_system_prompt=False,
    ):
        assert isinstance(inputs, list)

        prompts = []
        if self.system_prompt is not None:
            for msg in inputs:
                conv = get_conv_template(self.system_prompt)
                if change_system_prompt:
                    if self.system_prompt == "llama-2":
                        if character is not None:
                            conv.set_system_message(f"Act as if you were a {character}.")
                        else:
                            conv.set_system_message("")
                    elif self.system_prompt == "llama-3":
                        if character is not None:
                            conv.set_system_message(f"You are a {character}.")
                        else:
                            conv.set_system_message("")
                if llm_start_msg is not None:
                    conv.sep2 = " "
                conv.append_message(conv.roles[0], msg)
                conv.append_message(conv.roles[1], llm_start_msg)
                prompts.append(conv.get_prompt())
        else:
            prompts = inputs

        default_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        tokens = self.tokenizer(prompts, return_tensors="pt", padding="longest")
        self.tokenizer.padding_side = default_padding_side

        if postfix_token is not None:
            bs = tokens.input_ids.shape[0]
            tokens["input_ids"] = torch.cat(
                (
                    tokens.input_ids,
                    postfix_token.view(1, 1).expand(bs, 1).to(tokens.input_ids.device),
                ),
                dim=1,
            )
            tokens["attention_mask"] = torch.cat(
                (
                    tokens.attention_mask,
                    torch.ones(
                        (bs, 1),
                        device=tokens.attention_mask.device,
                        dtype=tokens.attention_mask.dtype,
                    ),
                ),
                dim=1,
            )

        output = self.model(**tokens.to(self.model.device))

        return output.logits
    
    
    def regenerate(
        self,
        inputs: list[str],
        max_new_tokens: int = 1,
        top_p: float = 0.9,
        temperature: float = 0.0,
        diff_matrices: list[np.ndarray] = None,
    ) -> list[str]:
        """
        Generate text by modifying hidden states of each layer using diff_matrices.
        """
        if diff_matrices is None:
            raise ValueError("The difference matrices are not loaded. Please provide `diff_matrices` during method call.")

        # Wrap generate() call using _apply_diff_hooks
        def forward_fn():
            return self.generate(inputs=inputs, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature)

        results = self._apply_diff_hooks(diff_matrices, forward_fn)
        return results

    def replace_generate(
        self,
        inputs: list[str],
        replace_matrices: list[np.ndarray] = None,
        max_new_tokens: int = 1,
        top_p: float = 0.9,
        temperature: float = 0.0,
        start: int = 0,
        end: int = None,
    ) -> list[str]:
        """
        Generate text by directly replacing the last token's hidden states
        for layers in [start, end) with 'replace_matrices'.

        Args:
            inputs (list[str]): Input prompts.
            replace_matrices (list[np.ndarray]): shape (num_layers, hidden_size).
            max_new_tokens (int): The maximum number of tokens to generate.
            top_p (float): Nucleus sampling parameter.
            temperature (float): Sampling temperature.
            start (int): Start layer index (inclusive).
            end (int): End layer index (exclusive). If None, defaults to total layers.

        Returns:
            List[str]: Generated text results.
        """
        if replace_matrices is None:
            raise ValueError("The replacement matrices must be provided.")

        def forward_fn():
            return self.generate(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
            )

        # Only replace layers in [start, end)
        outputs = self._apply_replace_hooks(
            replace_matrices=replace_matrices,
            forward_fn=forward_fn,
            start=start,
            end=end,
        )
        return outputs
    
    
    def get_hidden_states(self, prompt: str, character: str = None, **kwargs):
        """
        Extract hidden states from all layers for the specified character's tokens in six positions.
        Args:
            prompt (str): The input prompt.
            character (str): The role character to focus on (e.g., "management expert").
            **kwargs: Additional arguments for the model's forward pass.
        Returns:
            dict: Dictionary containing the extracted hidden states.
                  Keys: "pos1", "pos2", "pos3", "pos4", "pos5", "pos6"
                  Each key maps to a list of hidden states from all layers.
        """
        assert isinstance(prompt, str), "Input prompt must be a string."
       
        if self.system_prompt is not None:
            conv = get_conv_template(self.system_prompt)
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            formatted_prompt = conv.get_prompt()
        else:
            formatted_prompt = prompt

        # Tokenize the prompt
        tokens = self.tokenizer([formatted_prompt], return_tensors="pt", padding=True).to(self.model.device)

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = self.model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )

        hidden_states = outputs.hidden_states  # Tuple(num_layers, batch_size, seq_len, hidden_size)
        seq_len = tokens.input_ids.shape[1]

        # get positions
        positions = {"pos1": seq_len - 1}

        results = []

        for pos_name, index in positions.items():
            if index is not None and isinstance(index, int) and 0 <= index < seq_len:
                token_hs = []
                for layer_hs in hidden_states:
                    # layer_hs: (batch_size, seq_len, hidden_size)
                    token_vec = layer_hs[0, index, :].cpu().numpy()
                    token_hs.append(token_vec)
                results.append(token_hs)  # Each element is a list of hidden states across layers
            else:
                print(f"Warning: {pos_name} index is invalid or not found.")
                results.append(None)

        return results
    

