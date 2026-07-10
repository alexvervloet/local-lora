#!/usr/bin/env bash
# LoRA fine-tune with MLX. First run downloads the base model from Hugging Face
# (a few GB). Pass extra args through, e.g.:  ./train.sh --iters 10   (smoke run)
set -euo pipefail
cd "$(dirname "$0")"

# A 4-bit 3B instruct model is a good first base: small enough to train fast,
# big enough to follow the format. Swap via MODEL=... ./train.sh
MODEL="${MODEL:-mlx-community/Llama-3.2-3B-Instruct-4bit}"

# Where the adapter lands. Override to compare bases without clobbering the
# shipped one, e.g.  MODEL=...-1B... ADAPTER=adapters/model_1b ./train.sh
ADAPTER="${ADAPTER:-adapters}"

# NOTE: mlx-lm's CLI has changed names across versions (`python -m mlx_lm.lora`
# is deprecated in favor of `python -m mlx_lm lora` as of 0.31). If this errors
# after an upgrade, run `python -m mlx_lm lora --help` and adjust — PLAN.md
# phase 0 exists to catch exactly this.
python -m mlx_lm lora \
  --model "$MODEL" \
  --train \
  --data data \
  --iters "${ITERS:-200}" \
  --batch-size 1 \
  --adapter-path "$ADAPTER" \
  "$@"

echo
echo "Adapter written to $ADAPTER/ — now run:  python evaluate.py --adapter $ADAPTER"
