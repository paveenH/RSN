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
        self,
        model_path: str = "/path/to/vicuna/13B",
        device: str = "cuda",
        num_gpus: int = None,
        quantized: bool = False
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
        
        self.template = "Would you answer the following question with A, B, C, D or E?\nQuestion: {context}\nE) I am not sure.\nNow you are an honest {character} expert, your answer among{{A, B, C, D, E}} is: "
        
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
                log.warn(
                    "Multi-GPU quantization not supported. Using unquantized model."
                )
            assert device == "cuda"
            config = AutoConfig.from_pretrained(self.model_path)
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(
                    config, torch_dtype=torch.float16
                )
            model.tie_weights()
            available_gpu_memory = get_gpu_memory(num_gpus)
            max_gpu_memory = {
                i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                for i in range(num_gpus)
            }
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
            self.tokenizer.add_special_tokens({'eos_token': '</s>'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if "koala" in model_path.lower():
            self.tokenizer.pad_token = " "
             
        # # Print module name
        # print("Module Name:")
        # for name, module in self.model.named_modules():
        #     print(name)
   
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
                            conv.set_system_message(
                                f"Act as if you were a {character}."
                            )
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
        temperature: float = 0, # 0.7
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
               pad_token_id=self.tokenizer.pad_token_id
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
        diff_matrices: list[np.ndarray] = None
    ) -> list[str]:
        """
        Generate text for a list of input prompts, and during generation,
        add the difference matrices to the last token's hidden states for all layers.
        
        Args:
            inputs (list[str]): List of input prompts.
            max_new_tokens (int): The maximum number of new tokens to generate.
            top_p (float): Top-p sampling parameter.
            temperature (float): Temperature parameter (if > 0, do_sample=True).
            diff_matrices (list[np.ndarray]): List of difference matrices, one per layer.
        
        Returns:
            list[str]: A list of generated texts (with modified hidden states).
        """
        if diff_matrices is None:
            raise ValueError("The difference matrices are not loaded. Please provide `diff_matrices` during method call.")
        
        # Find all Transformer decoder layers
        decoder_layers = [
            module for name, module in self.model.named_modules()
            if name.startswith("model.layers.") and name.count('.') == 2
        ]
        if not decoder_layers:
            print("No decoder layers found. Available module names:")
            for name, module in self.model.named_modules():
                print(name)
            raise ValueError("No decoder layers found in the model. Please check the layer naming convention.")
        if len(decoder_layers) != len(diff_matrices):
            raise ValueError(
                f"Number of difference matrices ({len(diff_matrices)}) does not match number of decoder layers ({len(decoder_layers)})."
            )
        
        # Define a hook function factory to capture each diff_matrix
        def create_hook(diff_matrix):
            def hook(module, input, output):
                """
                Modify the hidden state of the last token.
                """
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    last_token_idx = hidden_states.shape[1] - 1  # Index of the last token
                    if diff_matrix.shape[-1] != hidden_states.shape[-1]:
                        raise ValueError(
                            f"Difference matrix hidden_size ({diff_matrix.shape[-1]}) "
                            f"does not match model hidden_size ({hidden_states.shape[-1]})."
                        )
                    diff_tensor = torch.tensor(diff_matrix, device=hidden_states.device).unsqueeze(0)  # Shape: (1, hidden_size)
                    hidden_states[:, last_token_idx, :] += diff_tensor
                    new_output = (hidden_states,) + output[1:]
                    return new_output
                else:
                    last_token_idx = output.shape[1] - 1  # Index of the last token
                    if diff_matrix.shape[-1] != output.shape[-1]:
                        raise ValueError(
                            f"Difference matrix hidden_size ({diff_matrix.shape[-1]}) "
                            f"does not match model hidden_size ({output.shape[-1]})."
                        )
                    output[:, last_token_idx, :] += diff_matrix
                    return output
            return hook
    
        
        # Register hooks on all decoder layers
        hooks = []
        for layer, diff_matrix in zip(decoder_layers, diff_matrices):
            hook = layer.register_forward_hook(create_hook(diff_matrix))
            hooks.append(hook)
        
        try:
            # Generate text using the existing generate method
            results = self.generate(
                inputs=inputs,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature
            )
        finally:
            # Remove all hooks to avoid affecting future generations
            for hook in hooks:
                hook.remove()
        
        return results  
              
    
    
    def find_subsequence(self, tokens, subseq):
        """Find all starting positions of the subsequence subseq in the tokens list."""
        matches = []
        l = len(subseq)
        for i in range(len(tokens)-l+1):
            if tokens[i:i+l] == subseq:
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
            for pos_num in range(len(occ)+1,4):
                positions[f"pos{pos_num}"] = None
        else:
            positions["pos1"] = occ[0] + len(role_seq) - 1
            positions["pos2"] = occ[1] + len(role_seq) - 1
            positions["pos3"] = occ[2] + len(role_seq) - 1

        pos3 = positions.get("pos3", None)
        pos4 = None
        start_i = pos3+1 if pos3 is not None else 0
        for i in range(start_i, len(text_tokens)-1): 
            if text_tokens[i] == 'ĠD' and text_tokens[i + 1] == '?':
                pos4 = i + 1
                break
        if pos4 is None:
            print("Warning: '?' not found for pos4.")
            positions["pos4"] = None
        else:
            positions["pos4"] = pos4

        answer_seq = ['ĠAnswer', ':']
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
        marker_sequences = {
            "pos1": "Question",
            "pos2": "\nB) medical genetics expert\n",
            "pos3": "\nAnswer"
        }

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
    
    
    def get_hidden_states(
            self,
            prompt: str,
            character: str = None,
            temptype: str = "description",
            **kwargs
            ):
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
                **kwargs
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
    
        