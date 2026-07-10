#!/usr/bin/env bash
# Close the loop: merge the LoRA adapter back into the base weights so the result
# is a single standalone model — no adapter flag needed to serve it. Keeps the
# base's 4-bit quantization (small, LM Studio / mlx_lm load it directly).
#
#   ./fuse.sh                       # fuse adapters/ into fused/
#   MODEL=... ADAPTER=... ./fuse.sh # fuse a different base/adapter
#
# Serve the result (all $0, local):
#   - mlx_lm.server:  python -m mlx_lm server --model fused   (OpenAI-compatible)
#   - LM Studio:      point it at this repo's fused/ folder (MLX runtime)
#   - Ollama:         needs GGUF — re-run with EXPORT_GGUF=1 (see README), then
#                     `ollama create` from a Modelfile pointing at the .gguf
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-mlx-community/Llama-3.2-3B-Instruct-4bit}"
ADAPTER="${ADAPTER:-adapters}"
SAVE="${SAVE:-fused}"

FUSE_ARGS=(--model "$MODEL" --adapter-path "$ADAPTER" --save-path "$SAVE")
# Ollama path: EXPORT_GGUF=1 also writes a dequantized f16 GGUF for `ollama create`.
if [[ "${EXPORT_GGUF:-0}" == "1" ]]; then
  FUSE_ARGS+=(--dequantize --export-gguf)
fi

python -m mlx_lm fuse "${FUSE_ARGS[@]}"

echo
echo "Fused model written to $SAVE/ — serve it standalone (no --adapter):"
echo "  python -m mlx_lm generate --model $SAVE --prompt '...'   # smoke test"
echo "  python -m mlx_lm server   --model $SAVE                  # local API"
