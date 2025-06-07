from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "GSAI-ML/LLaDA-1.5"
# 1) Load
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# 2) configuration
model.tie_weights()
model.generation_config.num_steps       = 10   
model.generation_config.answer_length  = 10    
model.generation_config.guidance_scale = 1.0   

# 3)  prompt
question = """Which of the following is a group under standard matrix multiplication?
A) The set of all 2x2 real matrices
B) The set of all invertible 2x2 real matrices
C) The set of all diagonal 2x2 real matrices
D) The set of all symmetric 2x2 real matrices"""
prompt = (
    "Would you answer the following question with A, B, C or D?\n"
    f"Question: {question}\n"
    "Now you are an expert in abstract algebra, your answer is: "
)

# 4) Tokenize 
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=10,    
    do_sample=False,      
    use_cache=False,      
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

gen_ids = outputs[0, inputs.input_ids.shape[1]:]
print("Answer:", tokenizer.decode(gen_ids).strip())