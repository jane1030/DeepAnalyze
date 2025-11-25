from huggingface_hub import snapshot_download
import os

repo_id = "RUC-DataLab/DeepAnalyze-8B"
local_dir = "models/DeepAnalyze-8B"

print(f"Starting download of {repo_id} to {local_dir}...")
try:
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    print("Download completed successfully!")
except Exception as e:
    print(f"Download failed: {e}")
