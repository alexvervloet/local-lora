# local-lora — fine-tune real weights on this Mac, and prove it helped

The [fine-tuning deep dive](../DeepDives/fine-tuning-deep-dive/) teaches the
whole discipline — dataset, validation, train, and the eval gate that only
ships a model that *provably* beats the baseline — but its trainer is a mock.
This project swaps the mock for **real weights**: a LoRA fine-tune of a small
open model, trained locally with **MLX** on Apple Silicon, evaluated with the
same held-out-set discipline.

**The claim being tested:** a LoRA-tuned 1.5–3B model follows the support
triage format (`category: <x> | reply: <one sentence>`) better than its base
model does — measured, not vibed.

**Cost: $0.** MLX trains on this machine's unified memory. No API key is used
anywhere in this repo.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # mlx-lm — Apple Silicon only
python check_setup.py

python convert_data.py               # deep-dive JSONL -> data/{train,valid}.jsonl
./train.sh                           # LoRA fine-tune (first run downloads the model)
python evaluate.py                   # base vs adapter on the held-out eval set
```

## The pieces

| Path | What it is |
|---|---|
| `convert_data.py` | Pulls the deep dive's chat-format datasets into `data/` for MLX |
| `train.sh` | The `mlx_lm.lora` invocation — model, iters, adapter output |
| `evaluate.py` | The gate: base vs adapter on `support_eval.jsonl`, same scorer idea as the dive |
| `PLAN.md` | Definition of done + results table |

## Honest scoping

- The dive's training set is **20 examples** — enough to see format adherence
  move on a small base model, probably not enough for a dramatic win. Phase 2
  of [PLAN.md](PLAN.md) grows it by distillation (the dive's example 08, but
  with a local teacher like `qwen3:8b` — still $0).
- MLX flags and model repos shift between releases. `check_setup.py` and a
  tiny `--iters 10` smoke run are the truth; the README is the intent.
- This is SFT/LoRA only. DPO and full fine-tunes are out of scope here (the
  dive covers the concepts; CUDA hardware would open Unsloth/QLoRA).
