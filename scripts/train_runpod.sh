#!/bin/bash
# RunPod pod startup script — clone repo, install, pull data, train
set -e

echo "=== CropOS Training — RunPod ==="
apt-get install -y git -q

git clone https://github.com/ckgoh-axia/CropOS-Weather.git /workspace/cropos
cd /workspace/cropos

pip install poetry -q
poetry install --no-root -q

# Pull preprocessed feature data from DVC remote
dvc pull data/raw/ || echo "DVC pull failed — will use local data if available"

mkdir -p checkpoints
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}"
export HF_TOKEN="${HF_TOKEN}"

python -m src.training.train

echo "=== Registering model to HuggingFace Hub ==="
python scripts/register_model.py --checkpoint checkpoints/best_model.pt
echo "=== Done ==="
