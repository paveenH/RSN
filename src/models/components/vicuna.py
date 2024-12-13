import logging

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from fastchat.conversation import get_conv_template
from fastchat.utils import get_gpu_memory
from tqdm import tqdm
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
        quantized: bool = False,
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
        temperature: float = 0.1, # 0.7
        top_p: float = 0.9,
    ):
        assert isinstance(inputs, list)
        
        do_sample = True if temperature > 0 else False

        # Support Batching?
        results = []
        for msg in tqdm(inputs):
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

            # print(f"{conv.roles[0]}: {msg}")
            # print(f"{conv.roles[1]}: {outputs}")
            results.append(outputs.strip())

        return results
    
    
    def find_subsequence(self, tokens, subseq):
        """Find all starting positions of the subsequence subseq in the tokens list."""
        matches = []
        l = len(subseq)
        for i in range(len(tokens)-l+1):
            if tokens[i:i+l] == subseq:
                matches.append(i)
        return matches
    
    def get_position(self, token_ids, text_tokens, character, tokenizer):
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
        for i in range(start_i, len(text_tokens)):
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
    
    
    def get_hidden_states(
            self,
            prompt: str,
            character: str,
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
        positions = self.get_position(token_ids, text_tokens, character, self.tokenizer)  

        results = [] 
        
        for pos_name, index in zip(["pos1", "pos2", "pos3", "pos4", "pos5", "pos6"], positions.values()):
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
 

if __name__ == "__main__":
    import json
    import argparse
    import os
    import numpy as np

    # Define the path to the JSON files
    PATH = "/data2/paveen/RolePlaying/src/models/components/mmlu"

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run VicundaModel on a specific task.")
    parser.add_argument("task", type=str, help="The name of the task to process.")
    parser.add_argument("size", type=str, help="The size of the model (e.g., 13B).")
    args = parser.parse_args()
    
    task = args.task
    size = args.size

    # Define model path
    model_path = f"/data2/paveen/RolePlaying/shared/llama3/{size}"   

    # Define JSON file path
    json_path = os.path.join(PATH, f"{task}.json")

    # Define save directory for hidden states
    save_dir = os.path.join("/data2/paveen/RolePlaying/src/models/components/hidden", f"hidden_states_{task}")
    os.makedirs(save_dir, exist_ok=True)

    # Initialize VicundaModel
    vc = VicundaModel(model_path=model_path)
    template = "You are a {character}, You are a {character}, You are a {character}, would you answer the following question with A, B, C or D? \n Question: {context}\n Answer: "
    characters = ["management expert", "medical genetics expert"]

    # Load JSON data
    print(f"Loading JSON data from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Total samples loaded: {len(data)}")

    # Initialize storage for hidden states
    hidden_states_storage = {character: [] for character in characters}

    # Iterate through each sample
    print("Starting hidden state extraction...")
    for idx, sample in enumerate(tqdm(data, desc="Processing Samples")):
        context = sample.get("text", "")
        if not context:
            print(f"Sample {idx} is missing 'text' field. Skipping.")
            continue

        for character in characters:
            # Generate prompt
            prompt = template.format(character=character, context=context)

            # Extract hidden states
            hidden_states = vc.get_hidden_states(prompt=prompt, character=character)

            # Check if all positions were found
            if any(pos is None for pos in hidden_states):
                print(f"Sample {idx} for character '{character}' has missing hidden states. Skipping.")
                continue

            hidden_states_array = np.stack([np.stack(pos, axis=0) for pos in hidden_states], axis=0)

            # Append to storage
            hidden_states_storage[character].append(hidden_states_array)
    
    print("Finished extracting hidden states.")

    # Save hidden states
    print("Saving hidden states to disk...")
    for character, hs_list in hidden_states_storage.items():
        if not hs_list:
            print(f"No hidden states collected for character '{character}'. Skipping save.")
            continue
        # Convert list to numpy array: shape (num_samples, 6, num_layers, hidden_size)
        hs_array = np.stack(hs_list, axis=0)
        # Define save path
        character_safe = character.replace(' ', '_')
        save_path = os.path.join(save_dir, f"{character_safe}_hidden_states.npy")
        # Save as .npy file
        np.save(save_path, hs_array)
        print(f"Saved hidden states for '{character}' to {save_path}")

    print("All hidden states have been saved successfully.")
        
        
        