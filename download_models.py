from huggingface_hub import snapshot_download
for repo in ["Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct"]:
    print("=== downloading", repo, flush=True)
    p = snapshot_download(repo, allow_patterns=["*.json","*.safetensors","*.txt","*.py"])
    print("=== done", repo, "->", p, flush=True)
print("ALL_DOWNLOADS_OK", flush=True)
