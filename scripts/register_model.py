#!/usr/bin/env python3
"""Push trained model checkpoint to HuggingFace Hub."""
import argparse
import os

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument(
        "--repo-id",
        default=None,
        help="HuggingFace model repo (e.g. gjmck78/cropos-gnn). "
             "Defaults to {whoami}/cropos-gnn using HF_TOKEN.",
    )
    parser.add_argument("--message", default="New training run checkpoint")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("ERROR: HF_TOKEN env var not set")

    api = HfApi(token=token)

    repo_id = args.repo_id
    if not repo_id:
        # Auto-detect from HF_MODEL_REPO env var, then HF_USER, then whoami
        repo_id = os.environ.get("HF_MODEL_REPO")
    if not repo_id:
        username = os.environ.get("HF_USER") or api.whoami()["name"]
        repo_id = f"{username}/cropos-gnn"

    print(f"Uploading {args.checkpoint} → hf://{repo_id} ...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    url = api.upload_file(
        path_or_fileobj=args.checkpoint,
        path_in_repo="best_model.pt",
        repo_id=repo_id,
        repo_type="model",
        commit_message=args.message,
    )
    print(f"Model registered: {url}")


if __name__ == "__main__":
    main()
