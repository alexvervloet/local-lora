# Plan — definition of done

The deliverable is the results table plus a short write-up of what moved and
why. A fine-tune that doesn't beat base **does not ship** — saying so honestly
is also a valid (and publishable) result.

## Phase 0 — plumbing

- [ ] `python check_setup.py` all green
- [ ] `python convert_data.py` writes `data/train.jsonl` + `data/valid.jsonl`
- [ ] `./train.sh --iters 10` smoke run completes and writes an adapter

## Phase 1 — the honest baseline

- [ ] `python evaluate.py --base-only` — score the *base* model on the eval set
      (format adherence + category accuracy). Record it before training anything.

## Phase 2 — train and gate

- [ ] Full `./train.sh`, then `python evaluate.py` (base vs adapter, same set)
- [ ] If the adapter doesn't clearly win: grow the training set by distillation
      (generate more labeled examples with a stronger local model, validate them,
      retrain). The dive's example 08 pattern, local teacher, $0.

## Results

| model | variant | format adherence | category accuracy | notes |
|---|---|---|---|---|
| (base id) | base | | | |
| (base id) | + LoRA | | | |

## Phase 3 — stretch (optional)

- [ ] Sweep one knob (iters or LoRA rank) and record the tradeoff
- [ ] Try a second base model size (1.5B vs 3B): does the smaller one gain more?
- [ ] Fuse the adapter and serve it via LM Studio/Ollama to close the loop

## Notes / gotchas discovered along the way

(keep a running log — MLX version quirks, token counts, memory use)

- **2026-07-07** — `mlx-lm` 0.31.3 crashes at import (`AttributeError:
  'str' object has no attribute '__module__'`) with `transformers` 5.x: its
  `NewlineTokenizer` registration uses the old `AutoTokenizer.register`
  signature. Pinned `transformers<5` (4.57.6 works).
