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

    def __call__(self, prompt: str, **kwargs):
        response = self.generate([prompt], **kwargs)
        return [{"generated_text": response[0]}]
    
    
    def get_hidden_states(
            self,
            prompt: str,
            character: str,
            extract_last_token: bool = True,
            extract_last_character_token: bool = False,
            **kwargs
            ):
        """
        Extract hidden states from the last token of the specified character.

        Args:
            prompt (str): The input prompt.
            character (str): The role character to focus on (e.g., "management expert").
            extract_last_token (bool): Whether to extract the hidden state of the last token in the prompt.
            extract_last_character_token (bool): Whether to extract the hidden state of the last token of the specified character.
            **kwargs: Additional arguments for the model's forward pass.

        Returns:
            dict: Dictionary containing the extracted hidden states.
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
        tokens = self.tokenizer([formatted_prompt],return_tensors="pt",padding=True).to(self.model.device)

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = self.model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **kwargs
                )

        hidden_states = outputs.hidden_states  # Tuple of (num_layers, batch_size, seq_len, hidden_size)
        last_layer = hidden_states[-1]  # Last layer's hidden states
        
        # Convert tokens to list for processing
        token_ids = tokens.input_ids[0].tolist()
        
        # Construct the role string in the prompt
        role_str = f"You are a {character},"
        
        # Tokenize the role string
        role_tokens = self.tokenizer.tokenize(role_str)
        role_token_ids = self.tokenizer.convert_tokens_to_ids(role_tokens)
        role_length = len(role_token_ids)

        # Find all occurrences of the role string
        occurrences = []
        for j in range(len(token_ids) - role_length + 1):
            if token_ids[j:j + role_length] == role_token_ids:
                occurrences.append(j + role_length - 1)  # Last token index of the role

        if not occurrences:
            print(f"Role string '{role_str}' not found in prompt.")
            return {}
        
        # Get the index of the last occurrence
        last_role_token_index = occurrences[-1]
        
        results = {}
        
        if extract_last_token:
            # Extract hidden state of the last token in the prompt
            last_token_hidden = last_layer[0, -1, :].cpu().numpy()
            results["last_token"] = last_token_hidden

        if extract_last_character_token:
            # Extract hidden state of the last token of the specified character's role
            last_character_token_hidden = last_layer[0, last_role_token_index, :].cpu().numpy()
            results["last_character_token"] = last_character_token_hidden

        return results


if __name__ == "__main__":
    
    import json
    import argparse
    import os
    import numpy as np

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run VicundaModel on a specific task.")
    parser.add_argument("task", type=str, help="The name of the task to process.")
    parser.add_argument("size", type=str, help="The size of the model (e.g., 1B, 3B).")
    args = parser.parse_args()
    
    task = args.task
    size = args.size

    model_path = f"/data2/paveen/RolePlaying/shared/llama3/{size}"   
    json_path = f"/data2/paveen/RolePlaying/src/models/components/{task}.json"
    
    vc = VicundaModel(model_path=model_path)
    template = "You are a {character}, You are a {character}, You are a {character}, would you answer the following question with A, B, C or D? \n Question: {context}\n Answer: "
    characters = ["management expert", "medical genetics expert"]  # 添加第二个角色
    
    # Create a hidden state storage directory
    hidden_states_dir = f"{task}_hidden_states"
    os.makedirs(hidden_states_dir, exist_ok=True)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        mmlu_questions = json.load(f)
        
    for character in characters:
        formatted_prompts = []
        for question in mmlu_questions:
            formatted_prompt = template.format(character=character, context=question['text'])
            formatted_prompts.append(formatted_prompt)
        
        # Generate response
        results = vc.generate(formatted_prompts)
        
        output = []
        correct_count = 0
        total_count = len(results)
        
        # Create a hidden state storage subdirectory for the current character
        character_hidden_states_dir = os.path.join(hidden_states_dir, character.replace(" ", "_"))
        os.makedirs(character_hidden_states_dir, exist_ok=True)
        
        for idx, response in enumerate(results):
            question_text = mmlu_questions[idx]["text"]
            label = mmlu_questions[idx]["label"]
            is_correct = response.strip() == chr(65 + label)  # 检查响应是否与正确答案匹配（A, B, C, D）
            if is_correct:
                correct_count += 1

            print(f"Character: {character}")
            print(f"Question {idx+1}: {question_text}")
            print(f"Response: {response}\n")
            print("Ground Truth:", label)
            print("Correct:" if is_correct else "Incorrect:", is_correct)
            print()
            
            # extract hidden states
            prompt = formatted_prompts[idx]
            hidden_states = vc.get_hidden_states(
                prompt=prompt,
                character=character,
                extract_last_token=True,
                extract_last_character_token=True
            )
            
            # Define hidden state file path
            hidden_state_file = os.path.join(character_hidden_states_dir, f"hidden_state_{idx+1}.npy")
            
            # Save hidden state
            np.save(hidden_state_file, hidden_states)
            
            output.append({
                "character": character,
                "question": question_text,
                "response": response,
                "ground_truth": label,
                "is_correct": is_correct,
                "hidden_state_file": hidden_state_file  # 可选：包含隐藏状态文件路径
            })
            
        accuracy = (correct_count / total_count) * 100
        print(f"Character: {character} - Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})\n")
    
        output_path = f"{task}_results_{character.replace(' ', '_')}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        print(f"Results for {character} saved to {output_path}\n")

    print(f"All results and hidden states saved under directory: {hidden_states_dir}")
        
        
        