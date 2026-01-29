from huggingface_hub import snapshot_download
import os

# 1. Configure paths
model_id = "meta-llama/Llama-3.3-70B-Instruct"
local_dir = "/work/d12922004/models/Llama-3.3-70B-Instruct"
cache_dir = "/work/d12922004/hf_cache"

# 2. Ensure directories exist
os.makedirs(local_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

print(f"🚀 开始下载 {model_id}...")
print(f"📂 目标目录: {local_dir}")
print(f"📦 缓存目录: {cache_dir}")

try:
    # 3. Perform download
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        cache_dir=cache_dir,
        local_dir_use_symlinks=True, # Use symlinks to save space
        token=True,                   # Will automatically read your previously logged-in token
        max_workers=8                 # Enable multithreaded download
    )
    print("\n✅ 下载成功！模型已就位。")
except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    print("💡 提示：请确保你已经运行过 'huggingface-cli login' 或者设置了 HF_TOKEN 环境变量。")