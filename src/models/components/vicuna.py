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
            self.system_prompt = "llama-3"
            # self.system_prompt = None
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
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        assert isinstance(inputs, list)

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
               # top_p=top_p,
               max_new_tokens=max_new_tokens,
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
            results.append(outputs)

        return results

    def __call__(self, prompt: str, **kwargs):
        response = self.generate([prompt], **kwargs)
        return [{"generated_text": response[0]}]


if __name__ == "__main__":
    
    import json

    def load_mmlu_questions(json_path: str) -> list:
        with open(json_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        return questions

    model_path = "/data2/paveen/RolePlaying/shared/llama3/3B"
    json_path = "mmlu_questions.json"
    mmlu_questions = load_mmlu_questions(json_path)
    
    vc = VicundaModel(model_path = model_path)
    template = """You are a {character}. Please answer the following multiple-choice question by selecting only one of the options: A, B, C, or D.
Do not provide any additional explanations or text.

Question: {question}
A) {A}
B) {B}
C) {C}
D) {D}
Answer:"""
    character = "Computer Science expert"
    
    formatted_prompts = []
    for item in mmlu_questions:
        prompt = template.format(
            character=character,
            question=item['question'],
            A=item['options']['A'],
            B=item['options']['B'],
            C=item['options']['C'],
            D=item['options']['D']
        )
        formatted_prompts.append(prompt)
    
    results = vc.generate(formatted_prompts)
    
    for idx, response in enumerate(results):
        question = mmlu_questions[idx]["question"]
        options = mmlu_questions[idx]["options"]
        print(f"Question {idx+1}: {question}")
        for key, value in options.items():
            print(f"  {key}) {value}")
        print(f"Model's Answer: {response}\n")