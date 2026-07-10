# local-lora — fine-tune real weights on this Mac, and prove it helped

The [fine-tuning deep dive](https://github.com/alexvervloet/ai-engineering-deep-dive/tree/main/fine-tuning-deep-dive)
teaches the whole discipline — dataset, validation, train, and the eval gate that
only ships a model that *provably* beats the baseline — but its trainer is a mock.
This project swaps the mock for **real weights**: a LoRA fine-tune of a small
open model, trained locally with **MLX** on Apple Silicon, evaluated with the
same held-out-set discipline.

**The claim being tested:** a LoRA-tuned 1.5–3B model follows the support
triage format (`category: <x> | reply: <one sentence>`) better than its base
model does — measured, not vibed.

**Cost: $0.** MLX trains on this machine's unified memory. No API key is used
anywhere in this repo.

## Prerequisites

This repo reads its dataset from the deep dive at data-prep and eval time, so it
expects the [`ai-engineering-deep-dive`](https://github.com/alexvervloet/ai-engineering-deep-dive)
repo checked out **as a sibling folder** (`check_setup.py` verifies this):

```bash
git clone git@github.com:alexvervloet/ai-engineering-deep-dive.git DeepDives
git clone git@github.com:alexvervloet/local-lora.git
cd local-lora        # DeepDives/ now sits one level up, where the scripts expect it
```

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
| `train.sh` | The `mlx_lm.lora` invocation — model, iters, adapter output (`MODEL=`/`ADAPTER=` overridable) |
| `evaluate.py` | The gate: base vs adapter on `support_eval.jsonl`, same scorer idea as the dive |
| `sweep.py` | Phase 3: train at several `--iters` and score each on the held-out set |
| `fuse.sh` | Phase 3: merge the adapter into a standalone model for local serving |
| `PLAN.md` | Definition of done + results table |
| `LESSONS.md` | What the process taught — eval-gate discipline, overfitting signals, the 1B-vs-3B result |

## Serving the fused model (Phase 3)

`./fuse.sh` merges the adapter back into the base so the result is one standalone
model — no adapter flag needed. It lands in `fused/` (4-bit, ~1.8 GB). Then, all $0
and local:

```bash
python -m mlx_lm generate --model fused --prompt "..."   # smoke test
python -m mlx_lm server   --model fused                  # OpenAI-compatible API on :8080
```

- **LM Studio** — point it at this repo's `fused/` folder; it runs MLX models directly.
- **Ollama** — needs GGUF, not MLX. Re-run `EXPORT_GGUF=1 ./fuse.sh` to also write a
  dequantized f16 `.gguf`, then `ollama create <name> -f Modelfile` with a Modelfile
  whose `FROM` points at that file.

## Honest scoping

- The dive's training set is **20 examples** — enough to see format adherence
  move on a small base model, probably not enough for a dramatic win. Phase 2
  of [PLAN.md](PLAN.md) grows it by distillation (the dive's example 08, but
  with a local teacher like `qwen3:8b` — still $0).
- MLX flags and model repos shift between releases. `check_setup.py` and a
  tiny `--iters 10` smoke run are the truth; the README is the intent.
- This is SFT/LoRA only. DPO and full fine-tunes are out of scope here (the
  dive covers the concepts; CUDA hardware would open Unsloth/QLoRA).
