#!/usr/bin/env python3
"""Phase 3 sweep: how does the LoRA's held-out score move as we vary one knob?

Trains the adapter at several `--iters` values (the knob most tied to the
overfitting question — train loss craters on 16 examples long before iter 200)
and scores each on the *same* held-out eval set the gate uses. Prints a tradeoff
table so we can see where the held-out win saturates versus where we're just
memorizing.

    python sweep.py                       # default: iters 25,50,100,200,400 on the 3B
    python sweep.py --iters 50 200        # custom points
    MODEL=... python sweep.py             # sweep a different base (honored by evaluate too)

Adapters land under adapters/sweep/ (gitignored) so the shipped adapters/ stays put.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from evaluate import MODEL_DEFAULT, load_eval_rows, run_model

SWEEP_DIR = Path(__file__).parent / "adapters" / "sweep"
VAL_LOSS_RE = re.compile(r"Val loss ([\d.]+)")


def train(model: str, iters: int, out: Path) -> float | None:
    """Train one adapter; return the last reported val loss (None if not found)."""
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-m", "mlx_lm", "lora", "--model", model, "--train",
         "--data", "data", "--iters", str(iters), "--batch-size", "1",
         "--adapter-path", str(out)],
        capture_output=True, text=True, check=True,
    )
    hits = VAL_LOSS_RE.findall(proc.stdout + proc.stderr)
    return float(hits[-1]) if hits else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, nargs="+", default=[25, 50, 100, 200, 400])
    parser.add_argument("--model", default=os.environ.get("MODEL", MODEL_DEFAULT))
    args = parser.parse_args()

    rows = load_eval_rows()
    print(f"sweep base: {args.model}")
    print(f"eval set:   {len(rows)} held-out rows\n")

    base = run_model(rows, args.model, adapter=None)
    print(f"{'iters':>6}  {'val loss':>9}  {'format':>7}  {'category':>9}")
    print(f"{'base':>6}  {'—':>9}  {base['format']:>6.0%}  {base['category']:>8.0%}")

    results = []
    for it in args.iters:
        val = train(args.model, it, SWEEP_DIR / f"iters_{it}")
        scored = run_model(rows, args.model, adapter=str(SWEEP_DIR / f"iters_{it}"))
        results.append((it, val, scored))
        val_s = f"{val:.3f}" if val is not None else "?"
        print(f"{it:>6}  {val_s:>9}  {scored['format']:>6.0%}  {scored['category']:>8.0%}")

    best = max(results, key=lambda r: (r[2]["category"], r[2]["format"], -r[0]))
    print(f"\nbest held-out: iters={best[0]} "
          f"(format {best[2]['format']:.0%}, category {best[2]['category']:.0%})")


if __name__ == "__main__":
    main()
