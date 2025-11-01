#!/usr/bin/env python3
import argparse
import json
import os
from huggingface_hub import create_repo, upload_folder
"""
Usage: 

python upload_sparse_model.py \
  --user quaternior \
  --local_dir $SPDP_HOME/out/llama2_7b/unstructured/wanda0.5 \
  --method wanda \
  --sparsity 0.5
"""
def main():
    parser = argparse.ArgumentParser(description="Upload sparse model folder to Hugging Face Hub")
    parser.add_argument("--user", type=str, required=True, help="Hugging Face username or organization")
    parser.add_argument("--local_dir", type=str, required=True, help="Local model directory to upload (target_dir)")
    parser.add_argument("--method", type=str, required=True, help="Sparsification method name (e.g., spdp, spmm)")
    parser.add_argument("--sparsity", type=str, required=True, help="Sparsity level (e.g., 0.5, 70%)")

    args = parser.parse_args()

    # 1. Load original model name from config.json
    config_path = os.path.join(args.local_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {args.local_dir}")

    with open(config_path, "r") as f:
        config = json.load(f)

    base_model = config.get("_name_or_path", None)
    if base_model is None:
        raise ValueError("config.json does not contain `_name_or_path`")

    # Extract only the last part (e.g., meta-llama/Llama-2-7b-hf → Llama-2-7b-hf)
    model_name = base_model.split("/")[-1]

    # 2. Construct new repo_id
    new_model_name = f"{model_name}-{args.method}-{args.sparsity}"
    repo_id = f"{args.user}/{new_model_name}"

    # 3. Create repo (skip if exists)
    create_repo(repo_id, exist_ok=True)

    # 4. Upload model folder
    upload_folder(
        repo_id=repo_id,
        folder_path=args.local_dir,
        commit_message=f"Upload sparse model ({args.method}, sparsity={args.sparsity})"
    )

    print(f"✅ Model uploaded to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
