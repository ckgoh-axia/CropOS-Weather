#!/bin/bash
# RunPod pod startup script — clone repo, install deps, pull data, train, register model
set -e

echo "=== CropOS Training — RunPod ==="
apt-get install -y git -q

rm -rf /workspace/cropos
git clone https://github.com/ckgoh-axia/CropOS-Weather.git /workspace/cropos
cd /workspace/cropos

pip install poetry wandb -q
poetry install --no-root -q

# Pull training data from HuggingFace Datasets
python - <<'PYEOF'
import os
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

token = os.environ["HF_TOKEN"]
username = HfApi().whoami(token=token)["name"]
repo_id = f"{username}/cropos-data"

outdir = Path("data/raw")
outdir.mkdir(parents=True, exist_ok=True)

for filename in ["era5_thailand.parquet", "metar_thai.parquet", "nwp_features.parquet"]:
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        token=token,
        local_dir=str(outdir),
    )
    print(f"✓ {filename}")

print("Data pull complete")
PYEOF

mkdir -p checkpoints
export HF_TOKEN="${HF_TOKEN}"

python -m src.training.train

echo "=== Registering model to HuggingFace Hub ==="
python scripts/register_model.py --checkpoint checkpoints/best_model.pt
echo "=== Done ==="
