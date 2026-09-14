"""Publish the selected VPTracker checkpoint to Hugging Face.

The model files stay outside Git. This command uploads only the files required
for Transformers loading plus the checked-in model card; evaluation logs are
excluded deliberately.
"""
from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-id", default="jcwang0602/VPTracker")
    parser.add_argument("--card", type=Path, default=Path("model_cards/VPTracker-Qwen3.5-2B.md"))
    args = parser.parse_args()

    from huggingface_hub import HfApi, upload_folder

    missing = [name for name in REQUIRED if not (args.checkpoint / name).is_file()]
    if missing:
        raise SystemExit(f"Checkpoint is missing required files: {', '.join(missing)}")
    if not args.card.is_file():
        raise SystemExit(f"Model card not found: {args.card}")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(args.checkpoint),
        allow_patterns=REQUIRED,
        commit_message="Update VPTracker to Qwen3.5-2B experiment 069",
    )
    api.upload_file(
        path_or_fileobj=str(args.card),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Update model card for Qwen3.5-2B checkpoint",
    )
    print(f"Uploaded {args.checkpoint} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
