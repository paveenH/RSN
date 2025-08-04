import logging
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from fastchat.conversation import get_conv_template
from diffusion import diffusion_generate

log = logging.getLogger(__name__)

import template as tmp

class VicundaModel:
    """
    Wrapper around a CausalLM to provide a consistent interface,
    support for quantization, multi–GPU loading, and role–based prompts.
    """

    task: str = "text2text-generation"

    def __init__(
        self,
        model_path: str,
        diffusion_mode: str = None,  # diffusion using dream or not
    ) -> None:
        self.model_path = model_path
        self.diffusion_mode = diffusion_mode
        self.system_prompt = self._infer_system_prompt(model_path)
        
        self.template_mmlu_E = tmp.template_mmlu_E
        self.template_mmlu = tmp.template_mmlu
        self.template_neutral_E = tmp.template_neutral_E
        self.template_neutral = tmp.template_neutral
        self.vanilla_E = tmp.vanilla_E
        self.vanilla = tmp.vanilla

        if diffusion_mode == "dream":
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )

        # Tokenizer
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
        return None

    def _ensure_padding_token(self) -> None:
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "</s>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _apply_diff_hooks(self, diff_matrices: list[np.ndarray], forward_fn):
        """
        Helper function: Register hooks on all Transformer decoder layers,
        adding the corresponding layer's diff matrix to the last token's hidden state,
        then performing forward_fn() for the forward pass, and finally removing the hook and returning the output.

        Args:
            diff_matrices (list[np.ndarray]): List of difference matrices for each layer
            forward_fn (function): The forward pass function

        Returns:
            Output: The return value of forward_fn()
        """
        # Locate all Transformer decoder layers
        decoder_layers = [
            module for name, module in self.model.named_modules() if name.startswith("model.layers.") and name.count(".") == 2
        ]
        if not decoder_layers:
            for name, module in self.model.named_modules():
                print(name)
            raise ValueError("No decoder layers found in the model. Please check the layer naming convention.")
        if len(decoder_layers) != len(diff_matrices):
            raise ValueError(
                f"Number of difference matrices ({len(diff_matrices)}) does not match number of decoder layers ({len(decoder_layers)})."
            )

        # Define hook factory function
        def create_hook(diff_matrix):
            def hook(module, input, output):
                # Supports both tuple and tensor output formats
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    last_token_idx = hidden_states.shape[1] - 1
                    if diff_matrix.shape[-1] != hidden_states.shape[-1]:
                        raise ValueError(
                            f"Diff matrix hidden_size ({diff_matrix.shape[-1]}) does not match model hidden_size ({hidden_states.shape[-1]})."
                        )
                    diff_tensor = torch.tensor(diff_matrix, device=hidden_states.device).unsqueeze(0)
                    hidden_states[:, last_token_idx, :] += diff_tensor
                    return (hidden_states,) + output[1:]
                else:
                    last_token_idx = output.shape[1] - 1
                    if diff_matrix.shape[-1] != output.shape[-1]:
                        raise ValueError(
                            f"Diff matrix hidden_size ({diff_matrix.shape[-1]}) does not match model hidden_size ({output.shape[-1]})."
                        )

                    output[:, last_token_idx, :] += diff_matrix
                    return output

            return hook

        # Register hooks
        hooks = []
        for layer, diff_matrix in zip(decoder_layers, diff_matrices):
            hook = layer.register_forward_hook(create_hook(diff_matrix))
            hooks.append(hook)

        try:
            outputs = forward_fn()
        finally:
            for hook in hooks:
                hook.remove()
        return outputs

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

    def _apply_lesion_hooks(self, neuron_indices: list[int], forward_fn, start: int = 0, end: int = None):
        """
        Register hooks on Transformer decoder layers in [start, end) such that
        for each forward pass, the entire column (neuron indices) in the hidden
        states output is zeroed out.

        Args:
            neuron_indices (list[int]): The neuron indices to set to zero in the last dimension.
            forward_fn (function): The forward pass function (e.g. self.generate(...)) to execute.
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
        if end is None or end > num_layers:
            end = num_layers

        if start < 0 or start >= num_layers:
            raise ValueError(f"Invalid start layer index: {start}, must be in [0, {num_layers-1}]")
        if end <= start:
            raise ValueError(f"Invalid range: start={start}, end={end}, must have end>start")

        # 2) Hook factory function: zero out the entire column for specified neuron indices
        def create_lesion_hook(neuron_ids: list[int]):
            def hook(module, module_input, module_output):
                # module_output could be a tuple (hidden_states, ...) or a single tensor
                if isinstance(module_output, tuple):
                    hidden_states = module_output[0]  # shape: (batch, seq_len, hidden_size)
                    if hidden_states.shape[-1] <= max(neuron_ids):
                        raise ValueError("Some neuron index is out of range for the hidden_size.")
                    hidden_states[..., neuron_ids] = 0.0
                    return (hidden_states,) + module_output[1:]
                else:
                    if module_output.shape[-1] <= max(neuron_ids):
                        raise ValueError("Some neuron index is out of range for the hidden_size.")
                    module_output[..., neuron_ids] = 0.0
                    return module_output

            return hook

        # 3) Register hooks for [start, end)
        hooks = []
        for layer_idx in range(start, end):
            layer = decoder_layers[layer_idx]
            hook = layer.register_forward_hook(create_lesion_hook(neuron_indices))
            hooks.append(hook)

        # 4) Run the forward pass
        try:
            outputs = forward_fn()
        finally:
            # 5) Remove hooks
            for h in hooks:
                h.remove()

        return outputs
    
    @torch.no_grad()
    def get_logits(self, prompts: list[str], return_hidden: bool = False
                   ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """
        Tokenize prompts, run the model, and return logits.
        If return_hidden is True, also return hidden_states from all layers.

        Returns:
          - logits: Tensor of shape (batch_size, seq_len, vocab_size)
          - hidden_states (optional): Tuple of length (num_layers+1), each Tensor
            of shape (batch_size, seq_len, hidden_size)
        """
        tokens = self.tokenizer(prompts, return_tensors="pt", padding="longest").to(self.model.device)        
        output = self.model(**tokens, return_dict=True, output_hidden_states=return_hidden)

        if return_hidden:
            return output.logits, output.hidden_states
        return output.logits
    
        
    @torch.no_grad()
    def regenerate_logits(
            self,
            prompts: list[str],
            diff_matrices: list[np.ndarray],
        ):
            if diff_matrices is None:
                raise ValueError("diff_matrices required")

            def forward_fn():
                return self.get_logits(prompts)  # (B, L, V)
            full_logits = self._apply_diff_hooks(diff_matrices, forward_fn)
            last_logits = full_logits[:, -1, :]    # shape = (B, V)
            return last_logits.cpu().numpy()
    
    @torch.no_grad()
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
                use_cache=False,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            gen_ids = output_ids[0][input_ids.shape[1] :]
            text = self.tokenizer.decode(
                gen_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )
            results.append(text.strip())

        return results


    @torch.no_grad()
    def generate_diffusion_llada(
        self,
        inputs: list[str],
        max_new_tokens: int = 4,
        steps: int = 50,
        block_len: int = 32,
        temperature: float = 0.0,
        guidance: float = 0.0,
    ) -> list[str]:
        """
        Use LLaDA's built-in diffusion sampling instead of the HF autoregressive generate method.
        """
        # mask_id = self.tokenizer.mask_token_id
        results = []

        for prompt in inputs:
            tok = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            full_ids = diffusion_generate(
                model=self.model,
                prompt_ids=tok.input_ids,
                gen_len=max_new_tokens,
                steps=steps,
                block_len=block_len,
                temperature=temperature,
                cfg_scale=guidance,
                remask="low_confidence",
                # mask_id     = mask_id,
            )
            gen_ids = full_ids[0, tok.input_ids.shape[1] :]  # Only get the answer part
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            results.append(text)

        return results

    @torch.no_grad()
    def generate_diffusion_dream(
        self,
        inputs: list[str],
        max_new_tokens: int = 4,
        steps: int = 50,
        temperature: float = 0.0,
        top_p: float = 0,
        alg: str = "entropy",
        alg_temp: float = 0.0,
        output_history: bool = False,
        return_dict: bool = False,
    ) -> list[str]:
        """
        Dream-org/Dream-v0-Instruct-7B
        """
        results = []
        for prompt in inputs:
            toks = self.tokenizer(
                prompt,
                return_tensors="pt",
                # return_dict=True,
                # add_generation_prompt=True,
            ).to(self.model.device)

            out = self.model.diffusion_generate(
                toks.input_ids,
                attention_mask=toks.attention_mask,
                max_new_tokens=max_new_tokens,
                steps=steps,
                temperature=temperature,
                top_p=top_p,
                alg=alg,
                alg_temp=alg_temp,
                output_history=output_history,
                return_dict_in_generate=return_dict,
            )

            if return_dict:
                seqs = out.sequences
            else:
                seqs = out

            gen_ids = seqs[0, toks.input_ids.shape[1] :]
            text = self.tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True).strip()
            results.append(text)
        return results

    @torch.no_grad()
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

    @torch.no_grad()
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

    @torch.no_grad()
    def generate_lesion(
        self,
        inputs: list[str],
        neuron_indices: list[int],
        start: int = 0,
        end: int = None,
        max_new_tokens: int = 1,
        top_p: float = 0.9,
        temperature: float = 0.0,
    ) -> list[str]:
        """
        Generate text while zeroing out the specified neuron indices in the
        last dimension for layers in [start, end).

        Args:
            inputs (list[str]): A batch of input prompts.
            neuron_indices (list[int]): The hidden-dim neuron indices to set to zero.
            start (int): Start layer index (0-based, inclusive).
            end (int): End layer index (0-based, exclusive). If None, defaults to total layers.
            max_new_tokens (int): The maximum number of tokens to generate.
            top_p (float): Nucleus sampling parameter.
            temperature (float): Sampling temperature.

        Returns:
            list[str]: The generated output strings for each prompt.
        """

        def forward_fn():
            return self.generate(inputs=inputs, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature)

        # Only zero out the specified neuron indices in [start, end) layers
        outputs = self._apply_lesion_hooks(neuron_indices=neuron_indices, forward_fn=forward_fn, start=start, end=end)
        return outputs

    @torch.no_grad()
    def get_hidden_states_mdf(self, prompt: str, diff_matrices: list[np.ndarray], **kwargs):
        """
        Similar to get_hidden_states, but during the forward pass, inject diff_matrices into the last token's hidden state
        of each decoder layer to obtain the modified hidden states (h'), allowing tracking of the propagated changes.

        Args:
            prompt (str): The input prompt.
            diff_matrices (list[np.ndarray]): The difference matrices for each layer (zero matrices for layers that don't need modification).

        Returns:
            list: A list of hidden states for each target position (e.g., pos1, pos2, ...) for each layer.
        """
        formatted_prompt = prompt

        tokens = self.tokenizer([formatted_prompt], return_tensors="pt", padding=True).to(self.model.device)
        seq_len = tokens.input_ids.shape[1]

        # Use _apply_diff_hooks to execute the forward pass and obtain modified hidden states
        def forward_fn():
            return self.model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )

        outputs = self._apply_diff_hooks(diff_matrices, forward_fn)
        hidden_states = outputs.hidden_states  # Tuple(num_layers, batch_size, seq_len, hidden_size)

        positions = {"pos1": seq_len - 1}

        results = []
        for pos_name, index in positions.items():
            if index is not None and isinstance(index, int) and 0 <= index < seq_len:
                token_hs = []
                for layer_hs in hidden_states:
                    token_vec = layer_hs[0, index, :].detach().cpu().numpy()
                    token_hs.append(token_vec)
                results.append(token_hs)
            else:
                print(f"Warning: {pos_name} index is invalid or not found.")
                results.append(None)
        return results

    @torch.no_grad()
    def get_hidden_states_rpl(self, prompt: str, replace_matrices: list[np.ndarray], start: int = 0, end: int = None, **kwargs):
        """
        Similar to get_hidden_states, but we replace the last token's hidden states
        in [start, end) layers using replace_matrices during the forward pass,
        then return the final hidden states for the positions of interest.

        Args:
            prompt (str): The input prompt for the model.
            replace_matrices (list[np.ndarray]): shape = (num_layers, hidden_size).
                We'll only apply them to layer indices in [start, end).
            start (int): Start layer index (0-based, inclusive).
            end (int): End layer index (0-based, exclusive). If None, defaults to total decoder layers.
            **kwargs: Additional arguments for the model's forward pass
                      (e.g., `attention_mask`, `past_key_values`, etc.).

        Returns:
            list: A list of shape [num_positions], each element is a list of shape [num_layers, hidden_size].
                  For example, if we only track "pos1", then it returns [[layer0_vec, layer1_vec, ...],].
        """
        # 1) Construct prompt
        formatted_prompt = prompt

        # 2) Tokenize
        tokens = self.tokenizer([formatted_prompt], return_tensors="pt", padding=True).to(self.model.device)
        seq_len = tokens.input_ids.shape[1]

        # 3) Define forward_fn, which is used to register in _apply_replace_hooks
        def forward_fn():
            return self.model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )

        # 4) Use _apply_replace_hooks to complete the replacement
        outputs = self._apply_replace_hooks(replace_matrices=replace_matrices, forward_fn=forward_fn, start=start, end=end)
        hidden_states = outputs.hidden_states  # Tuple(num_layers, batch_size, seq_len, hidden_size)

        # 5) Find position
        positions = {"pos1": seq_len - 1}

        # 6) Collect and return the hidden states corresponding to the positions
        results = []
        for pos_name, index in positions.items():
            if index is not None and 0 <= index < seq_len:
                token_hs = []
                # hidden_states: tuple of length num_layers
                # each layer_hs shape: (batch_size, seq_len, hidden_size)
                for layer_hs in hidden_states:
                    # 取 batch=0, seq_pos=index
                    token_vec = layer_hs[0, index, :].detach().cpu().numpy()
                    token_hs.append(token_vec)
                results.append(token_hs)
            else:
                print(f"Warning: {pos_name} index is invalid or not found.")
                results.append(None)

        return results

    @torch.no_grad()
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
