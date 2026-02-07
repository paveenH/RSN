import logging
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, AutoConfig
from diffusion import diffusion_generate

log = logging.getLogger(__name__)


def _is_mistral3_model(model_path: str) -> bool:
    """Check if the model is a Mistral3 multimodal model."""
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return config.model_type == "mistral3"
    except Exception:
        # Fallback: check model path name
        return "mistral3" in model_path.lower() or "ministral-3" in model_path.lower()


class VicundaModel:
    """
    Wrapper around a CausalLM to provide a consistent interface,
    support for quantization, multi–GPU loading, and role–based prompts.
    """

    task: str = "text2text-generation"

    def __init__(
        self,
        model_path: str,
        diffusion_mode: str = None,  # whether to use diffusion with dream mode
    ) -> None:
        self.model_path = model_path
        self.diffusion_mode = diffusion_mode

        # Model
        if diffusion_mode == "dream":
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,  # or use torch.float32 if needed
                device_map="auto",
            )
        elif _is_mistral3_model(model_path):
            # Mistral3 is a multimodal model, requires specific class
            from transformers import Mistral3ForConditionalGeneration
            log.info(f"Detected Mistral3 model, using Mistral3ForConditionalGeneration")
            self.model = Mistral3ForConditionalGeneration.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,  # or use torch.float32 if needed
                device_map="auto",
            )

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=False,
            trust_remote_code=True,
        )
        self._ensure_padding_token()

    # ───────────────────── Core helpers ───────────────────── #

    def _ensure_padding_token(self) -> None:
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "</s>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _find_decoder_layers(self):
        """
        Collect all decoder layers. Supports multiple naming conventions:
        - Standard CausalLM: 'model.layers.*'
        - Mistral3 (multimodal): 'model.language_model.layers.*'
        Called by all hook functions to preserve original behavior.
        """
        # Try standard CausalLM naming first
        decoder_layers = [
            module for name, module in self.model.named_modules()
            if name.startswith("model.layers.") and name.count(".") == 2
        ]

        # Try Mistral3/multimodal naming (model.language_model.layers.*)
        if not decoder_layers:
            decoder_layers = [
                module for name, module in self.model.named_modules()
                if name.startswith("model.language_model.layers.") and name.count(".") == 3
            ]

        if not decoder_layers:
            # Print all module names for debugging
            print("Available module names:")
            for name, module in self.model.named_modules():
                if "layer" in name.lower():
                    print(f"  {name}")
            raise ValueError("No decoder layers found in the model. Please check the layer naming convention.")
        return decoder_layers

    # ───────────────────── Hook framework ───────────────────── #

    def _apply_diff_hooks(
        self,
        diff_matrices: list[np.ndarray],
        forward_fn,
        last_indices: torch.Tensor | None = None,
        tail_len: int = 1,
    ):
        """
        Add diff_matrices to last token (or tail_len tokens) for each layer.
        """
        decoder_layers = self._find_decoder_layers()

        if len(decoder_layers) != len(diff_matrices):
            raise ValueError(
                f"Number of difference matrices ({len(diff_matrices)}) "
                f"does not match number of decoder layers ({len(decoder_layers)})."
            )

        def create_hook(diff_matrix):
            def hook(module, input, output):
                def prepare_diff(hs: torch.Tensor) -> torch.Tensor:
                    B, _, H = hs.shape
                    diff_t = torch.as_tensor(diff_matrix, device=hs.device, dtype=hs.dtype)
                    if diff_t.ndim == 1:
                        diff_t = diff_t.unsqueeze(0).expand(B, -1)  # expand to [B, H]
                    elif diff_t.ndim == 2 and diff_t.shape[0] == 1:
                        diff_t = diff_t.expand(B, -1)  # expand [1, H] to [B, H]
                    else:
                        assert diff_t.shape == (B, H), f"diff shape {diff_t.shape} != (B, {H})"
                    return diff_t  # return [B, H]

                def add_at_tail(hs: torch.Tensor) -> torch.Tensor:
                    # hs: [B, L, H] (batch, sequence_length, hidden_size)
                    B, L, H = hs.shape
                    n = max(int(tail_len), 1)

                    if last_indices is not None:
                        last_pos = last_indices.to(device=hs.device, dtype=torch.long)  # [B]
                    else:
                        last_pos = torch.full((B,), L - 1, device=hs.device, dtype=torch.long)  # [B]

                    offs = torch.arange(n, device=hs.device, dtype=torch.long)  # [n]
                    pos_raw = last_pos.unsqueeze(1) - offs.unsqueeze(0)  # [B, n]
                    valid_mask = pos_raw >= 0  # [B, n]
                    pos_mat = pos_raw.clamp_min(0)  # [B, n]

                    diff_bh = prepare_diff(hs).unsqueeze(1)  # [B, 1, H] -> [B, n, H]
                    diff_bh = diff_bh * valid_mask.unsqueeze(-1)  # [B, n, H]

                    add_buf = torch.zeros_like(hs)  # [B, L, H]
                    batch_idx = torch.arange(B, device=hs.device).unsqueeze(1).expand(B, n)  # [B, n]
                    add_buf[batch_idx, pos_mat, :] = diff_bh
                    hs += add_buf
                    return hs

                if isinstance(output, tuple):
                    hidden_states = output[0]
                    hidden_states = add_at_tail(hidden_states)
                    return (hidden_states,) + output[1:]
                else:
                    hidden_states = output
                    hidden_states = add_at_tail(hidden_states)
                    return hidden_states

            return hook

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
        Replace the last token's hidden state with replace_matrices for layers in [start, end).
        """
        decoder_layers = self._find_decoder_layers()
        num_layers = len(decoder_layers)

        if end is None or end > num_layers:
            end = num_layers

        if len(replace_matrices) < num_layers:
            raise ValueError(
                f"replace_matrices has length {len(replace_matrices)}, "
                f"but we found {num_layers} decoder layers. Need at least >= num_layers."
            )
        if start < 0 or start >= num_layers:
            raise ValueError(f"Invalid start layer index: {start}. Must be in [0, {num_layers-1}].")
        if end <= start:
            raise ValueError(f"Invalid range: start={start}, end={end}. Must have end > start.")

        def create_replace_hook(replace_matrix: np.ndarray):
            def hook(module, module_input, module_output):
                if isinstance(module_output, tuple):
                    hidden_states = module_output[0]
                    last_token_idx = hidden_states.shape[1] - 1
                    if replace_matrix.shape[-1] != hidden_states.shape[-1]:
                        raise ValueError(
                            f"Replacement hidden_size ({replace_matrix.shape[-1]}) "
                            f"!= model hidden_size ({hidden_states.shape[-1]})."
                        )
                    rep_tensor = torch.tensor(replace_matrix, device=hidden_states.device).unsqueeze(0)
                    hidden_states[:, last_token_idx, :] = rep_tensor
                    return (hidden_states,) + module_output[1:]
                else:
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

        hooks = []
        for layer_idx in range(num_layers):
            if start <= layer_idx < end:
                layer = decoder_layers[layer_idx]
                rep_matrix = replace_matrices[layer_idx]
                hook = layer.register_forward_hook(create_replace_hook(rep_matrix))
                hooks.append(hook)

        try:
            outputs = forward_fn()
        finally:
            for hook in hooks:
                hook.remove()

        return outputs

    def _apply_index_lesion_hooks(self, neuron_indices: list[int], forward_fn, start: int = 0, end: int = None):
        """
        Zero out given neuron_indices for layers in [start, end).
        """
        decoder_layers = self._find_decoder_layers()
        num_layers = len(decoder_layers)

        if end is None or end > num_layers:
            end = num_layers
        if start < 0 or start >= num_layers:
            raise ValueError(f"Invalid start layer index: {start}, must be in [0, {num_layers-1}]")
        if end <= start:
            raise ValueError(f"Invalid range: start={start}, end={end}, must have end>start")

        def create_lesion_hook(neuron_ids: list[int]):
            def hook(module, module_input, module_output):
                if len(neuron_ids) == 0:
                    return module_output
                if isinstance(module_output, tuple):
                    hidden_states = module_output[0]
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

        hooks = []
        for layer_idx in range(start, end):
            layer = decoder_layers[layer_idx]
            hook = layer.register_forward_hook(create_lesion_hook(neuron_indices))
            hooks.append(hook)

        try:
            outputs = forward_fn()
        finally:
            for h in hooks:
                h.remove()

        return outputs

    def _apply_rsn_hooks(
        self,
        rsn_indices_per_layer: list[list[int]],
        forward_fn,
        mode: str = "lesion",  # "lesion" or "complement"
    ):
        """
        - lesion:
            rsn_ids = []     → do nothing for this layer
            rsn_ids = [ids]  → zero-out ONLY these neurons
            
        - complement:
            rsn_ids = []     → zero-out ENTIRE layer
            rsn_ids = [ids]  → keep-only these neurons, zero-out others
        """

        decoder_layers = self._find_decoder_layers()
        num_layers = len(decoder_layers)

        if len(rsn_indices_per_layer) != num_layers:
            raise ValueError(
                f"rsn_indices_per_layer has {len(rsn_indices_per_layer)}, "
                f"but model has {num_layers} layers."
            )

        def create_layer_hook(rsn_ids):
            rsn_ids = np.array(rsn_ids, dtype=int)

            def hook(module, module_input, module_output):

                if isinstance(module_output, tuple):
                    hs = module_output[0]
                    tail = module_output[1:]
                else:
                    hs = module_output
                    tail = None

                H = hs.shape[-1]

                # -------- LESION mode --------
                if mode == "lesion":
                    # if rsn_ids is empty, skip; otherwise zero out specified neurons
                    if rsn_ids.size > 0:
                        hs[..., rsn_ids] = 0.0

                # -------- COMPLEMENT mode --------
                elif mode == "complement":

                    # if rsn_ids is empty, zero out entire layer; otherwise keep only rsn_ids
                    if rsn_ids.size == 0:
                        hs[..., :] = 0.0

                    else:
                        # keep only rsn_ids, zero out all others
                        drop_ids = np.setdiff1d(np.arange(H), rsn_ids)
                        if drop_ids.size > 0:
                            hs[..., drop_ids] = 0.0

                else:
                    raise ValueError(f"Unknown mode={mode}")

                if tail is None:
                    return hs
                return (hs,) + tail

            return hook

        # Register hooks for each layer
        hooks = []
        for L in range(num_layers):
            rsn_ids = rsn_indices_per_layer[L]   # per-layer neuron indices list
            hook = decoder_layers[L].register_forward_hook(create_layer_hook(rsn_ids))
            hooks.append(hook)

        try:
            outputs = forward_fn()
        finally:
            for h in hooks:
                h.remove()

        return outputs

    def _apply_rsn_lesion_hooks(
        self,
        rsn_indices_per_layer: list[list[int]],
        forward_fn,
    ):
        return self._apply_rsn_hooks(
            rsn_indices_per_layer=rsn_indices_per_layer,
            forward_fn=forward_fn,
            mode="lesion",
        )

    def _apply_rsn_complement_hooks(
        self,
        rsn_indices_per_layer: list[list[int]],
        forward_fn,
    ):
        return self._apply_rsn_hooks(
            rsn_indices_per_layer=rsn_indices_per_layer,
            forward_fn=forward_fn,
            mode="complement",
        )

    # ───────────────────── Generate Logits ───────────────────── #

    @torch.no_grad()
    def get_logits(
        self, prompts: list[str], return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        tokens = self.tokenizer(prompts, return_tensors="pt", padding="longest")
        tokens = tokens.to(self.model.device)

        output = self.model(**tokens, return_dict=True, output_hidden_states=return_hidden, use_cache=False)

        if return_hidden:
            return output.logits, output.hidden_states
        return output.logits

    @torch.no_grad()
    def regenerate_logits(self, prompts: list[str], diff_matrices: list[np.ndarray], tail_len: int = 1):
        if diff_matrices is None:
            raise ValueError("diff_matrices required")

        tokens = self.tokenizer(prompts, return_tensors="pt", padding="longest").to(self.model.device)
        attn = tokens.attention_mask
        last_idx = attn.sum(dim=1) - 1  # (B,)

        def forward_fn():
            return self.model(**tokens, return_dict=True, output_hidden_states=False, use_cache=False).logits

        full_logits = self._apply_diff_hooks(diff_matrices, forward_fn, last_indices=last_idx, tail_len=tail_len)  # shape: (B, L, V)

        B, L, V = full_logits.shape
        idx = last_idx.view(B, 1, 1).expand(B, 1, V)
        last_logits = full_logits.gather(dim=1, index=idx).squeeze(1)  # shape: (B, V)
        return last_logits.detach().cpu().to(torch.float32).numpy()

    def regenerate_rsn_lesion(
        self,
        prompts: list[str],
        rsn_indices_per_layer: list[list[int]],
    ):
        """
        Lesion RSNs per layer and return last-token logits.
        """
        tokens = self.tokenizer(prompts, return_tensors="pt", padding="longest").to(self.model.device)
        attn = tokens.attention_mask
        last_idx = attn.sum(dim=1) - 1  # (B,)

        def forward_fn():
            return self.model(
                **tokens,
                return_dict=True,
                output_hidden_states=False,
                use_cache=False,
            ).logits  # shape: (B, L, V)

        full_logits = self._apply_rsn_lesion_hooks(
            rsn_indices_per_layer=rsn_indices_per_layer,
            forward_fn=forward_fn,
        )  # shape: (B, L, V)

        B, L, V = full_logits.shape
        idx = last_idx.view(B, 1, 1).expand(B, 1, V)
        last_logits = full_logits.gather(dim=1, index=idx).squeeze(1)  # shape: (B, V)

        return last_logits.detach().cpu().to(torch.float32).numpy()

    @torch.no_grad()
    def regenerate_rsn_complement(
        self,
        prompts: list[str],
        rsn_indices_per_layer: list[list[int]],
    ):
        """
        Complement Ablation:
        Keep only RSN neurons; zero out all other neurons.
        Return last-token logits for each prompt.
        """
        tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
        ).to(self.model.device)

        attn = tokens.attention_mask
        last_idx = attn.sum(dim=1) - 1  # (B,)

        def forward_fn():
            return self.model(
                **tokens,
                return_dict=True,
                output_hidden_states=False,
                use_cache=False,
            ).logits  # shape: (B, L, V)

        full_logits = self._apply_rsn_complement_hooks(
            rsn_indices_per_layer=rsn_indices_per_layer,
            forward_fn=forward_fn,
        )  # shape: (B, L, V)

        B, L, V = full_logits.shape
        gather_idx = last_idx.view(B, 1, 1).expand(B, 1, V)
        last_logits = full_logits.gather(dim=1, index=gather_idx).squeeze(1)

        return last_logits.detach().cpu().to(torch.float32).numpy()

    # ───────────────────── Generate answer ───────────────────── #
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
        top_p_val = top_p if do_sample else None
        temperature_val = temperature if do_sample else None

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
                temperature=temperature_val,
                use_cache=False,
                top_p=top_p_val,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            gen_ids = output_ids[0][input_ids.shape[1]:]  # extract generated token ids
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
            )
            gen_ids = full_ids[0, tok.input_ids.shape[1] :]
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

            seqs = out.sequences if return_dict else out
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

        def forward_fn():
            return self.generate(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
            )

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

        outputs = self._apply_replace_hooks(
            replace_matrices=replace_matrices,
            forward_fn=forward_fn,
            start=start,
            end=end,
        )
        return outputs

    @torch.no_grad()
    def regenerate_index_lesion(
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
        Generate text while zeroing out specified neuron indices in [start, end).
        """

        def forward_fn():
            return self.generate(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
            )

        outputs = self._apply_index_lesion_hooks(
            neuron_indices=neuron_indices,
            forward_fn=forward_fn,
            start=start,
            end=end,
        )
        return outputs

    # ───────────────────── Hidden state extractors ───────────────────── #

    @torch.no_grad()
    def get_hidden_states_mdf(self, prompt: str, diff_matrices: list[np.ndarray], **kwargs):
        """
        Get hidden states under diff_matrices (mdf), using the same diff-hook
        mechanism as regenerate_logits.
        """
        formatted_prompt = prompt
        tokens = self.tokenizer([formatted_prompt], return_tensors="pt", padding=True).to(self.model.device)
        seq_len = tokens.input_ids.shape[1]

        def forward_fn():
            return self.model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )

        outputs = self._apply_diff_hooks(diff_matrices, forward_fn)
        hidden_states = outputs.hidden_states  # tuple of (num_layers, B, L, H) tensors

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
    def get_hidden_states_rpl(
        self,
        prompt: str,
        replace_matrices: list[np.ndarray],
        start: int = 0,
        end: int = None,
        **kwargs,
    ):
        """
        Get hidden states when replacing last token's hidden state for layers in [start, end).
        """
        formatted_prompt = prompt
        tokens = self.tokenizer([formatted_prompt], return_tensors="pt", padding=True).to(self.model.device)
        seq_len = tokens.input_ids.shape[1]

        def forward_fn():
            return self.model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )

        outputs = self._apply_replace_hooks(
            replace_matrices=replace_matrices,
            forward_fn=forward_fn,
            start=start,
            end=end,
        )
        hidden_states = outputs.hidden_states  # tuple of (num_layers, B, L, H) tensors

        positions = {"pos1": seq_len - 1}
        results = []

        for pos_name, index in positions.items():
            if index is not None and 0 <= index < seq_len:
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
    def get_hidden_states(self, prompt: str, character: str = None, **kwargs):
        """
        Basic hidden state extractor (no editing).
        """
        tokens = self.tokenizer([prompt], return_tensors="pt", padding=True).to(self.model.device)

        outputs = self.model(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs.hidden_states  # tuple of (num_layers, B, L, H) tensors
        seq_len = tokens.input_ids.shape[1]
        positions = {"pos1": seq_len - 1}  # extract last token position
        results = []

        for pos_name, index in positions.items():
            if index is not None and isinstance(index, int) and 0 <= index < seq_len:
                token_hs = []
                for layer_hs in hidden_states:
                    token_vec = layer_hs[0, index, :].cpu().numpy()
                    token_hs.append(token_vec)
                results.append(token_hs)
            else:
                print(f"Warning: {pos_name} index is invalid or not found.")
                results.append(None)

        return results
