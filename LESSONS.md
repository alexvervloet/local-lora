# LESSONS

Engineering lessons from swapping the [fine-tuning deep dive](https://github.com/alexvervloet/ai-engineering-deep-dive/tree/main/fine-tuning-deep-dive)'s
*mock* trainer for **real LoRA weights** on this Mac (MLX, Llama-3.2 1B/3B). The
dive teaches the discipline — dataset, validation, train, and an eval gate that
only ships a model that *provably* beats its base. Doing it for real, most of
these lessons are about the same thing: **not fooling yourself about whether the
fine-tune actually helped.** A loss curve is easy to produce; evidence is not.
Each is tied to the concrete run that taught it. Kept as a running log.

Sibling logs in the parent folder learned overlapping lessons from other angles —
`rag-at-scale` (don't fool yourself with your own benchmark numbers) and
`remote-jobs-digest` (local runners lie; observe the result, not the knob) —
cross-referenced by section below where they recur.

---

## 1. Train loss measures memorization; only the held-out gate measures learning

The full run drove train loss from 4.67 to **0.028** on a 16-example training
set — the adapter had all but memorized the data. Read alone, that number is
useless: it's equally consistent with "learned the task" and "overfit to noise,"
and you cannot tell which from the training curve. The only figure that settled it
was the **held-out gate**: base `Llama-3.2-3B` scored 60% format / 20% category,
the adapter 100% / 80%, on 10 rows the model never trained on. That gap is the
entire claim. The repo's rule — *a fine-tune that doesn't beat base does not
ship* — exists because a falling loss curve feels like progress while proving
nothing.

Takeaway: training loss is a convergence check, not a result. Never let it stand
in for the thing you actually care about — score on data the model didn't see, and
make that comparison the gate.

## 2. Split the metric by difficulty — an aggregate hides which part is hard

The task has two axes, and scoring them separately was what made the results
legible. **Format adherence** (does the reply match `category: <x> | reply: …`?)
hits 100% by iter 25 and never moves — the *shape* is trivially learned. **Category
accuracy** (is `<x>` right?) is the real work: it climbs slowly and plateaus around
80%. A single blended "accuracy" would have averaged a solved sub-task with an
unsolved one into a middling number that told you nothing about where to push.

Takeaway: when a task bundles a mechanical part and a semantic part, measure them
on separate axes. The average of "instant" and "hard" is a number that describes
neither. (Same shape as rag-at-scale's §1: name the subsystem your number actually
measures.)

## 3. More training is not more skill — watch train and val diverge for the stop signal

The whole point of the iters sweep was to find where the win saturates. It
saturates almost immediately: format is maxed by iter 25, category is at its
plateau by ~100. Past that, held-out score is flat while **val loss climbs** —
0.708 at iter 25, 0.838 at 200, 0.884 at 400 — even as train loss keeps falling.
That divergence *is* overfitting, made concrete: the extra iterations are fitting
the 16 training rows harder and buying nothing on held-out data. The shipped
200-iter run was already well past the useful point; ~25–100 iters would have
produced the same model for a quarter of the compute.

Takeaway: extra training tokens buy tighter fit to the training set, not more
generalization. When train loss falls while val loss rises, stop — the curve is
telling you the exact iteration where learning turned into memorizing. (Directly
parallel to remote-jobs-digest's §9: more reasoning is not more accuracy.)

## 4. On a tiny eval set, measure the noise before you rank configs

The held-out set is 10 rows, so each row is worth 10 percentage points and every
score is quantized to the nearest 10%. In the sweep, category accuracy read 90% at
25 iters, **30%** at 50, then 80% at 100/200/400. The 50-iter number looks like a
catastrophic regression and the 25-iter number like a winner — but both are inside
the noise of a 10-row sample. `sweep.py` dutifully prints "best held-out:
iters=25," and the honest reading is that you *cannot* distinguish 80% from 90%
here, nor trust a single low point. The real, robust signals were the ones that
held across every config (format pinned at 100%) or moved monotonically (val loss).

Takeaway: before declaring one configuration better than another, work out your
measurement's resolution. On a 10-row eval, a 10-point difference is a rounding
artifact, not a result — rank on signals that survive the noise, and never crown a
"best" from one run of one point. (Cousin of rag-at-scale's §2: don't draw the
curve from points too small to carry it.)

## 5. Fine-tuning's payoff is largest where the base is weakest

The size comparison gave the sharpest result of the project. Untuned `Llama-3.2-1B`
scores **0% / 0%** — it cannot even emit the required format, let alone the right
category. The *same* 200-iter LoRA recipe lifts it to **100% / 80%**: identical to
the tuned 3B. So the 3B gained +40/+60 and the 1B gained +100/+80, and on this
narrow task a tuned 1B fully erases the 3B's advantage at a third of the parameters.
The capacity a big base spends being generally capable is exactly what a small base
recovers through fine-tuning when the task is narrow.

Takeaway: don't reach for the bigger base by reflex. Fine-tuning returns the most
where the base is worst, and for a single well-specified task a tuned small model
can match a tuned large one — measure both before paying for size you may not need.

## 6. Make the train/eval boundary structural, not disciplined

The gate is only honest if the eval rows never leaked into training, and that's
enforced by *paths*, not by remembering to be careful. `convert_data.py` only ever
reads `support_train.jsonl` and writes the split into `data/`; `evaluate.py` reads
`support_eval.jsonl` **straight from the deep dive's folder**, which the training
code never touches. There is no step that *could* copy an eval row into the training
set, so leakage isn't something to vigilantly avoid — it's structurally impossible.

Takeaway: when a correctness property depends on two things never mixing, separate
them physically so mixing them isn't an available move. A boundary you have to
remember to honor is a boundary you'll eventually forget. (Same principle as
remote-jobs-digest's §15: draw boundaries with directories, not discipline.)

## 7. The smoke run is the truth about a shifting toolchain — and verify the fused artifact behaves

MLX and its model repos move fast, and the breakage is silent until you run it.
`mlx-lm` 0.31 renamed the CLI (`python -m mlx_lm.lora` → `python -m mlx_lm lora`);
0.31.3 crashes at *import* under `transformers` 5.x, forcing a `transformers<5`
pin. None of that is visible from reading the README — it surfaces only when
`check_setup.py` and the `--iters 10` smoke run actually execute. The same "observe,
don't assume" applied at the end: fusing the adapter into a standalone model can
produce weights that load fine but didn't absorb the adaptation, so I loaded
`fused/` with **no adapter flag** and pushed a real "charged twice" ticket through
it — got `category: billing | reply: …`, in-format and correct. That output is the
proof the merge worked; the exit code isn't.

Takeaway: a pinned, self-hosted ML stack deviates from the docs in undocumented
ways — treat the smoke run and a real end-to-end generation as the source of truth,
not the version numbers or a clean exit. (This is remote-jobs-digest's §5, "local runners have real,
undocumented quirks," in a training harness.)
