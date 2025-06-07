from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "GSAI-ML/LLaDA-1.5"
tokenizer   = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model       = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
).eval()                

MASK          = tokenizer.mask_token      # 通常就是 "<mask>"
ANSWER_LEN    = 10                        # 想让模型生成多少个 token
NUM_STEPS     = 10                        # diffusion 步数，经验上设成 = ANSWER_LEN 即可
GUIDANCE      = 1.0                       # 0~2 之间自行调

# ----------------------------------------------------------------------------
# ⚠️ 关键 1：在 prompt 末尾追加 N 个 <mask>，让模型知道要填多少空
# ----------------------------------------------------------------------------
question = (
    "Which of the following is a group under standard matrix multiplication?\n"
    "A) The set of all 2x2 real matrices\n"
    "B) The set of all invertible 2x2 real matrices\n"
    "C) The set of all diagonal 2x2 real matrices\n"
    "D) The set of all symmetric 2x2 real matrices\n"
)
prompt = (
    "Would you answer the following question with A, B, C or D?\n"
    f"Question: {question}\n"
    "Now you are an expert in abstract algebra, your answer is: "
    + (MASK + " ") * ANSWER_LEN            # <mask> <mask> … 共 ANSWER_LEN 个
)

# ----------------------------------------------------------------------------
# ⚠️ 关键 2：把 diffusion 采样相关超参写进 generation_config
# ----------------------------------------------------------------------------
model.tie_weights()                        # 官方要求
gc = model.generation_config
gc.num_steps       = NUM_STEPS             # diffusion 步数
gc.answer_length   = ANSWER_LEN            # 和 <mask> 数保持一致
gc.guidance_scale  = GUIDANCE              # classifier-free guidance
# 其他采样超参
top_p       = 0.95
temperature = 1.0                          # do_sample=True 时才生效

# ----------------------------------------------------------------------------
# 生成
# ----------------------------------------------------------------------------
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out_ids = model.generate(
    **inputs,
    max_new_tokens=ANSWER_LEN,             # 让 HF 框架知道最多补多少
    do_sample=True,                        # Diffusion 建议打开
    temperature=temperature,
    top_p=top_p,
    use_cache=False,                       # MDM/KV-cache 不支持
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

gen_ids = out_ids[0, inputs.input_ids.shape[1]:]   # 只看补出的部分
print("Raw tokens :", gen_ids.tolist())
print("Answer     :", tokenizer.decode(gen_ids).strip())