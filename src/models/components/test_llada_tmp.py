from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model & tokenizer
model_name = "GSAI-ML/LLaDA-1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             trust_remote_code=True,
                                             device_map="auto"
                                             )

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
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate
outputs = model.generate(
    **inputs,
    max_new_tokens=5,
    do_sample=False,
    temperature=0,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)


# Decode
gen = outputs[0][inputs.input_ids.shape[1]:]  # only new tokens
print("Output:", tokenizer.decode(gen))