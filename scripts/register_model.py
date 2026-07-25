#!/usr/bin/env python3
"""Push trained model checkpoint to HuggingFace Hub."""
import argparse
import os
from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--repo-id", default="cropos/cropos-gnn")
    parser.add_argument("--message", default="New training run checkpoint")
    args = parser.parse_args()

    api = HfApi(token=os.environ["HF_TOKEN"])
    url = api.upload_file(
        path_or_fileobj=args.checkpoint,
        path_in_repo="best_model.pt",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.message,
    )
    print(f"Model registered: {url}")


if __name__ == "__main__":
    main()
