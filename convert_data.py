#!/usr/bin/env python3
"""Pull the fine-tuning deep dive's datasets into data/ for MLX.

The dive's files are already chat-format JSONL ({"messages": [...]}), which
recent mlx-lm accepts directly — so "conversion" is mostly a copy plus a
train/valid split, kept as a script so the split is reproducible and the repo
never depends on the dive's paths at train time.
"""

import json
import sys
from pathlib import Path

DIVE = Path(__file__).parent.parent / "DeepDives" / "fine-tuning-deep-dive" / "datasets"
DATA = Path(__file__).parent / "data"
VALID_ROWS = 4  # of the 20 training rows, hold this many out for training-time validation


def main() -> None:
    src = DIVE / "support_train.jsonl"
    if not src.exists():
        sys.exit(f"Not found: {src} — is the DeepDives folder next to this repo?")

    rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    for row in rows:
        assert row["messages"][-1]["role"] == "assistant", "each row must end with the target"

    DATA.mkdir(exist_ok=True)
    # Deterministic split: every 5th row goes to valid (keeps categories mixed).
    valid_idx = set(list(range(0, len(rows), 5))[:VALID_ROWS])
    valid = [rows[i] for i in sorted(valid_idx)]
    train = [r for i, r in enumerate(rows) if i not in valid_idx]

    (DATA / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    (DATA / "valid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in valid))

    # The eval set stays where it is — evaluate.py reads it from the dive so the
    # held-out data can't drift from the source of truth.
    print(f"wrote {len(train)} train + {len(valid)} valid rows to {DATA}/")
    print(f"eval set stays at {DIVE / 'support_eval.jsonl'} (read by evaluate.py)")


if __name__ == "__main__":
    main()
