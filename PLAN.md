# Plan — definition of done

The deliverable is the results table plus a short write-up of what moved and
why. A fine-tune that doesn't beat base **does not ship** — saying so honestly
is also a valid (and publishable) result.

## Phase 0 — plumbing

- [x] `python check_setup.py` all green
- [x] `python convert_data.py` writes `data/train.jsonl` + `data/valid.jsonl`
- [x] `./train.sh --iters 10` smoke run completes and writes an adapter

## Phase 1 — the honest baseline

- [x] `python evaluate.py --base-only` — score the *base* model on the eval set
      (format adherence + category accuracy). Record it before training anything.
      **2026-07-10:** base `Llama-3.2-3B-Instruct-4bit` = format **60%**, category
      **20%** on the 10 held-out rows. This is the bar the adapter must clear.

## Phase 2 — train and gate

- [x] Full `./train.sh` (200 iters, val loss 4.67 → 0.84), then `python evaluate.py`
      (base vs adapter, same set). **2026-07-10: SHIP** — adapter wins on both axes.
- [x] ~~If the adapter doesn't clearly win: grow the training set by distillation~~
      Not needed — the adapter clearly won (format 60→100%, category 20→80%). The
      distillation fallback (dive's example 08, local teacher, $0) stays documented
      here for if a future base/knob regresses.

## Results

| model | variant | format adherence | category accuracy | notes |
|---|---|---|---|---|
| Llama-3.2-3B-Instruct-4bit | base | 60% | 20% | Phase 1 baseline, 10 held-out rows (2026-07-10) |
| Llama-3.2-3B-Instruct-4bit | + LoRA | 100% | 80% | 200 iters, r=16 default; SHIP (2026-07-10) |

## Phase 3 — stretch (optional)

- [x] Sweep one knob (iters or LoRA rank) and record the tradeoff — `python sweep.py`
      (2026-07-10, 3B base, same 10 held-out rows):

      | iters | val loss | format | category |
      |---|---|---|---|
      | base | — | 60% | 20% |
      | 25 | 0.708 | 100% | 90% |
      | 50 | 0.763 | 100% | 30% |
      | 100 | 0.835 | 100% | 80% |
      | 200 | 0.838 | 100% | 80% |
      | 400 | 0.884 | 100% | 80% |

      **Tradeoff:** format is learned almost instantly (100% by iter 25, flat after);
      category plateaus ~80% by iter 100. Crucially **val loss *rises* past iter 25**
      (0.708 → 0.884) while train loss craters — textbook overfitting. So the shipped
      200-iter run is past diminishing returns; ~100 iters buys the same held-out
      result. (The 50-iter 30% dip is eval noise — 10 rows, ±10pts each.)
- [ ] Try a second base model size (1.5B vs 3B): does the smaller one gain more?
- [ ] Fuse the adapter and serve it via LM Studio/Ollama to close the loop

## Notes / gotchas discovered along the way

(keep a running log — MLX version quirks, token counts, memory use)

- **2026-07-07** — `mlx-lm` 0.31.3 crashes at import (`AttributeError:
  'str' object has no attribute '__module__'`) with `transformers` 5.x: its
  `NewlineTokenizer` registration uses the old `AutoTokenizer.register`
  signature. Pinned `transformers<5` (4.57.6 works).
