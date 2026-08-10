#!/bin/bash
# RunPod pod startup script — clone repo, install deps, pull data, train, register model
set -e

echo "=== CropOS Training — RunPod ==="
apt-get install -y git -q

rm -rf /workspace/cropos
git clone https://github.com/ckgoh-axia/CropOS-Weather.git /workspace/cropos
cd /workspace/cropos

# The RunPod image (py3.10) is incompatible with pyproject.toml's python = "^3.11",
# so poetry install fails silently. Install deps directly via pip instead.
# torch 2.2.1+cu121 is already present in the base image — skip reinstalling it.
echo "=== Installing dependencies ==="
pip install wandb -q

# PyG wheels must come from the official index keyed to torch+cuda version
pip install torch-geometric -q
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html -q

# Remaining project runtime deps (mlflow required by train.py)
pip install \
    mlflow \
    openmeteo-requests requests-cache retry-requests \
    pandas numpy xarray scikit-learn \
    fastapi uvicorn pydantic httpx \
    python-dotenv "huggingface-hub>=0.21" pyarrow pyyaml tqdm \
    -q

echo "=== Dependencies installed ==="

# Pull training data from HuggingFace Datasets into data/raw/ so train.py can
# use --local data/raw instead of re-downloading inside the training process.
python - <<'PYEOF'
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

token = os.environ["HF_TOKEN"]
username = HfApi().whoami(token=token)["name"]
repo_id = f"{username}/cropos-data"

outdir = Path("data/raw")
outdir.mkdir(parents=True, exist_ok=True)

# Required files
for filename in ["era5_thailand.parquet", "metar_thai.parquet"]:
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=token,
        local_dir=str(outdir),
    )
    print(f"✓ {filename}")

# Optional top-up files — skip silently if not yet on HF
for filename in ["era5_north.parquet", "era5_recent.parquet"]:
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
            local_dir=str(outdir),
        )
        print(f"✓ {filename}")
    except Exception as e:
        print(f"  {filename} not found — skipping ({e})")

print("Data pull complete")
PYEOF

mkdir -p checkpoints
export HF_TOKEN="${HF_TOKEN}"
export MLFLOW_ALLOW_FILE_STORE=true

# -u disables Python output buffering so logs appear before the process is killed.
# 2>&1 merges stderr (logger output) into stdout so RunPod captures it.
PYTHONUNBUFFERED=1 python -u -m src.training.train --local data/raw 2>&1

echo "=== Registering model to HuggingFace Hub ==="
python scripts/register_model.py --checkpoint checkpoints/best_model.pt
echo "=== Done ==="
