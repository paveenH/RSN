from huggingface_hub import snapshot_download
import os

# 1. Configure paths
model_id = "meta-llama/Llama-3.3-70B-Instruct"
local_dir = "/work/d12922004/models/Llama-3.3-70B-Instruct"
cache_dir = "/work/d12922004/hf_cache"

# 2. Ensure directories exist
os.makedirs(local_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

print(f"🚀 Starting download of {model_id}...")
print(f"📂 Target directory: {local_dir}")
print(f"📦 Cache directory: {cache_dir}")

try:
    # 3. Execute the download
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        cache_dir=cache_dir,
        local_dir_use_symlinks=True,  # use symlinks to save space
        token=True,                   # will read your logged-in token automatically
        max_workers=2,                # lower concurrency to reduce memory usage
        resume_download=True,         # enable resuming interrupted downloads
    )
    print("\n✅ Download complete! Model is ready.")
except Exception as e:
    print(f"\n❌ Download failed: {e}")
    print("💡 Tip: make sure you've run 'huggingface-cli login' or set the HF_TOKEN environment variable.")