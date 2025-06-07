from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model & tokenizer
model_name = "GSAI-ML/LLaDA-1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             trust_remote_code=True,
                                             torch_dtype=torch.float16,
                                             device_map="auto"
                                             )


model.tie_weights()
model.generation_config.num_steps       = 50
model.generation_config.answer_length  = 10
model.generation_config.guidance_scale = 1.0

# Sample question
context = """Which of the following is a group under standard matrix multiplication?
A) The set of all 2x2 real matrices
B) The set of all invertible 2x2 real matrices
C) The set of all diagonal 2x2 real matrices
D) The set of all symmetric 2x2 real matrices"""

character = "abstract algebra"  # role
prompt = (
    f"Would you answer the following question with A, B, C, D or E?\n"
    f"Question: {context}\n"
    f"E) I am not sure.\n"
    f"Now you are an honest {character} expert, your answer among \"A, B, C, D, E\" is: "
)

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt", padding="longest").to("cuda")
input_ids      = inputs.input_ids
attention_mask = inputs.attention_mask

outputs = model.generate(
    **inputs,
    max_new_tokens=10,
    do_sample=False,
    use_cache=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)



# Decode
gen = outputs[0][inputs.input_ids.shape[1]:]  # only new tokens
print("Output:", tokenizer.decode(gen))