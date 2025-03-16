import logging
import numpy as np
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from fastchat.conversation import get_conv_template
from fastchat.utils import get_gpu_memory
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

log = logging.getLogger(__name__)


class VicundaModel:
    # with this we aim to look like a huggingface pipeline such that we can use langchain
    task: str = "text2text-generation"

    def __init__(
        self, model_path: str = "/path/to/vicuna/13B", device: str = "cuda", num_gpus: int = None, quantized: bool = False
    ) -> None:
        self.model_path = model_path
        if "vicuna" in model_path.lower():
            self.system_prompt = "vicuna_v1.1"
        elif "koala" in model_path.lower():
            self.system_prompt = "koala_v1"
        elif "llama2" in model_path.lower():
            self.system_prompt = "llama-2"
        elif "llama3" in model_path.lower():
            # self.system_prompt = "llama-3"
            self.system_prompt = None
        else:
            self.system_prompt = None

        # v1
        # self.template = "Would you answer the following question with A, B, C, D or E?\nQuestion: {context}\nE) I am not sure.\nNow you are an honest {character} expert, your answer among{{A, B, C, D, E}} is: "
        # v2
        # self.template = "Would you answer the following question with A, B, C, D or E?\nQuestion: {context}\nE) I am not sure.\nNow you are an honest {character} expert, your answer among (A, B, C, D, E) is: "
        if "phi" in model_path.lower() or "qwen" in model_path.lower():
            # v4
            self.template = 'Would you answer the following question with A, B, C, D or E?\nQuestion: {context}\nE) I am not sure.\nNow you are an honest {character} expert, your only answer with one token among "A, B, C, D, E" is: '
        else:
            # v3
            # self.template = 'Would you answer the following question with A, B, C, D or E?\nQuestion: {context}\nE) I am not sure.\nNow you are an honest {character} expert, your answer among "A, B, C, D, E" is: '
            # v3 students
            self.template = 'Would you answer the following question with A, B, C, D or E?\nQuestion: {context}\nE) I am not sure.\nNow you are an honest {character} student, your answer among "A, B, C, D, E" is: '
            # v5
            # self.template = 'Would you answer the following question with A, B, C, D or E?\nQuestion: {context}\nE) I am not sure.\nNow your answer among "A, B, C, D or E" as an honest {character} expert is: '
            # Else for v5
            # self.template = 'Would you answer the following question with A, B, C, D or E?\nQuestion: {context}\nE) I am not sure.\nNow you need to answer with "A, B, C, D or E" as an honest {character} expert: '
        if quantized:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        if num_gpus is None:
            num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            if quantized:
                log.warn("Multi-GPU quantization not supported. Using unquantized model.")
            assert device == "cuda"
            config = AutoConfig.from_pretrained(self.model_path)
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
            model.tie_weights()
            available_gpu_memory = get_gpu_memory(num_gpus)
            max_gpu_memory = {i: str(int(available_gpu_memory[i] * 0.85)) + "GiB" for i in range(num_gpus)}
            self.model = load_checkpoint_and_dispatch(
                model,
                self.model_path,
                device_map="auto",
                max_memory=max_gpu_memory,
                no_split_module_classes=["LlamaDecoderLayer"],
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                quantization_config=bnb_config,
            )
            if not quantized:
                self.model = self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        # set a padding token
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "</s>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if "koala" in model_path.lower():
            self.tokenizer.pad_token = " "

        # # Print module name
        # print("Module Name:")
        # for name, module in self.model.named_modules():
        #     print(name)
        
    
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
            module for name, module in self.model.named_modules()
            if name.startswith("model.layers.") and name.count(".") == 2
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
        for layer_idx in range(num_layers):
            if start <= layer_idx < end:
                layer = decoder_layers[layer_idx]
                hook = layer.register_forward_hook(create_lesion_hook(neuron_indices))
                hooks.append(hook)
            else:
                pass

        # 4) Run the forward pass
        try:
            outputs = forward_fn()
        finally:
            # 5) Remove hooks
            for h in hooks:
                h.remove()

        return outputs

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

    def generate(
        self,
        inputs: list[str],
        max_new_tokens: int = 1,
        # temperature: float = 0.1, # 0.7
        top_p: float = 0.9,
        temperature: float = 0,  # 0.7
    ):
        assert isinstance(inputs, list)

        # Determine sampling mode
        do_sample = temperature > 0

        # Adjust parameters based on sampling mode
        top_p = top_p if do_sample else None
        temperature = temperature if do_sample else None

        # # Print parameters for debugging
        # print(f"  do_sample: {do_sample}")
        # print(f"  temperature: {temperature}")
        # print(f"  top_p: {top_p}")

        # Support Batching?
        results = []
        for msg in inputs:
            if isinstance(msg, list) and len(msg) == 1 and isinstance(msg[0], str):
                msg = msg[0]
            if self.system_prompt is not None:
                conv = get_conv_template(self.system_prompt)
                conv.append_message(conv.roles[0], msg)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
            else:
                prompt = msg

            tokens = self.tokenizer([prompt], return_tensors="pt", padding="longest")
            input_ids = tokens.input_ids
            attention_mask = tokens.attention_mask

            input_tensor = input_ids.to(next(self.model.parameters()).device)
            attention_mask = attention_mask.to(next(self.model.parameters()).device)

            output_ids = self.model.generate(
                input_tensor,
                attention_mask=attention_mask,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if self.model.config.is_encoder_decoder:
                output_ids = output_ids[0]
            else:
                output_ids = output_ids[0][len(input_ids[0]) :]
            outputs = self.tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )

            results.append(outputs.strip())

        return results


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
                return self.generate(
                    inputs=inputs,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    temperature=temperature
                )

            # Only zero out the specified neuron indices in [start, end) layers
            outputs = self._apply_lesion_hooks(
                neuron_indices=neuron_indices,
                forward_fn=forward_fn,
                start=start,
                end=end
            )
            return outputs
    

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
        # Construct the prompt
        if self.system_prompt is not None:
            conv = get_conv_template(self.system_prompt)
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            formatted_prompt = conv.get_prompt()
        else:
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
        if self.system_prompt is not None:
            conv = get_conv_template(self.system_prompt)
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            formatted_prompt = conv.get_prompt()
        else:
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

    def find_subsequence(self, tokens, subseq):
        """Find all starting positions of the subsequence subseq in the tokens list."""
        matches = []
        l = len(subseq)
        for i in range(len(tokens) - l + 1):
            if tokens[i : i + l] == subseq:
                matches.append(i)
        return matches

    def get_position_mmlu(self, token_ids, text_tokens, character, tokenizer):
        """
        Get target positions
        """

        positions = {}

        char_words = character.split()
        # role_seq = ['You', 'Ġare', 'Ġa'] + ['Ġ'+w for w in char_words] + [',']
        role_seq = [f"Ġ{w}" for w in char_words]

        occ = self.find_subsequence(text_tokens, role_seq)
        if len(occ) < 3:
            print(f"Warning: Found only {len(occ)} occurrences of the role line for '{character}'.")
            for pos_num in range(len(occ) + 1, 4):
                positions[f"pos{pos_num}"] = None
        else:
            positions["pos1"] = occ[0] + len(role_seq) - 1
            positions["pos2"] = occ[1] + len(role_seq) - 1
            positions["pos3"] = occ[2] + len(role_seq) - 1

        pos3 = positions.get("pos3", None)
        pos4 = None
        start_i = pos3 + 1 if pos3 is not None else 0
        for i in range(start_i, len(text_tokens) - 1):
            if text_tokens[i] == "ĠD" and text_tokens[i + 1] == "?":
                pos4 = i + 1
                break
        if pos4 is None:
            print("Warning: '?' not found for pos4.")
            positions["pos4"] = None
        else:
            positions["pos4"] = pos4

        answer_seq = ["ĠAnswer", ":"]
        ans_occ = self.find_subsequence(text_tokens, answer_seq)
        if len(ans_occ) == 0:
            print("Warning: 'Answer:' not found for pos6.")
            positions["pos6"] = None
            positions["pos5"] = None
        else:
            ans_start = ans_occ[0]
            pos6 = ans_start + len(answer_seq) - 1
            positions["pos6"] = pos6
            if pos6 - 1 >= 0:
                positions["pos5"] = pos6 - 1
            else:
                print("Warning: pos5 index is out of range.")
                positions["pos5"] = None

        return positions

    def get_position_description(self, token_ids, text_tokens, tokenizer):
        """
        Get target positions for markers corresponding to:
        pos1: End of context description (after '.\nQuestion:')
        pos2: End of options (after '\nB) medical genetics expert\n')
        pos3: After 'Answer:'

        Args:
            token_ids (list[int]): List of token IDs.
            text_tokens (list[str]): List of token strings.
            character (str): The role character (not used here).
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer instance.

        Returns:
            dict: Dictionary containing the positions.
                  Keys: "pos1", "pos2", "pos3"
                  Values: Token index or None
        """
        positions = {}
        marker_sequences = {"pos1": "Question", "pos2": "\nB) medical genetics expert\n", "pos3": "\nAnswer"}

        for pos_name, marker in marker_sequences.items():
            # Tokenize the marker
            marker_tokens = tokenizer.tokenize(marker)
            # Find all occurrences of the marker in the text_tokens
            occ = self.find_subsequence(text_tokens, marker_tokens)
            if len(occ) == 0:
                print(f"Warning: Marker '{marker}' not found.")
                positions[pos_name] = None
            else:
                # Assuming only one occurrence per marker
                # Store the position after the marker sequence
                positions[pos_name] = occ[0] + len(marker_tokens)

        return positions

    def get_hidden_states(self, prompt: str, character: str = None, temptype: str = "description", **kwargs):
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
        if temptype == "mmlu" and not character:
            raise ValueError("Character must be provided for mmlu temptype.")

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

        # Convert tokens to list for processing
        token_ids = tokens.input_ids[0].tolist()
        text_tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

        # get positions
        if temptype == "mmlu":
            positions = self.get_position_mmlu(token_ids, text_tokens, character, self.tokenizer)
        elif temptype == "description":
            positions = self.get_position_description(token_ids, text_tokens, self.tokenizer)
        elif temptype == "abcde":
            positions = {"pos1": seq_len - 1}
        else:
            print("Type error")
            return None

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
