#!/usr/bin/env python3
"""The gate: does the LoRA adapter beat the base model on the held-out set?

    python evaluate.py                # base vs adapter, side by side
    python evaluate.py --base-only    # phase 1: record the baseline first

Scores two things per answer, same spirit as the dive's evaluate step:
  - format adherence: does the reply match 'category: <x> | reply: <text>'?
  - category accuracy: is <x> the expected category?

The eval rows are read straight from the deep dive's held-out set, so this
repo can't accidentally train on them.
"""

import argparse
import json
import re
from pathlib import Path

EVAL_SET = (
    Path(__file__).parent.parent
    / "DeepDives" / "fine-tuning-deep-dive" / "datasets" / "support_eval.jsonl"
)
MODEL_DEFAULT = "mlx-community/Llama-3.2-3B-Instruct-4bit"
FORMAT_RE = re.compile(r"^category:\s*(account|billing|technical|other)\s*\|\s*reply:\s*\S")


def load_eval_rows() -> list[dict]:
    rows = [json.loads(line) for line in EVAL_SET.read_text().splitlines() if line.strip()]
    out = []
    for row in rows:
        msgs = row["messages"]
        expected = msgs[-1]["content"]
        m = FORMAT_RE.match(expected.strip())
        out.append({
            "prompt_messages": msgs[:-1],          # system + user
            "expected_category": m.group(1) if m else None,
        })
    return out


def score(reply: str, expected_category: str | None) -> tuple[bool, bool]:
    m = FORMAT_RE.match(reply.strip())
    format_ok = m is not None
    category_ok = bool(m and expected_category and m.group(1) == expected_category)
    return format_ok, category_ok


def run_model(rows: list[dict], model: str, adapter: str | None) -> dict:
    # Imported here so --help works on machines without mlx installed.
    from mlx_lm import generate, load

    # load() is typed as a union with a 3-tuple (return_config=True) variant.
    llm, tokenizer, *_ = load(model, adapter_path=adapter)
    fmt_hits = cat_hits = 0
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            row["prompt_messages"], add_generation_prompt=True, tokenize=False
        )
        reply = generate(llm, tokenizer, prompt=prompt, max_tokens=80, verbose=False)
        f, c = score(reply, row["expected_category"])
        fmt_hits += f
        cat_hits += c
    n = len(rows)
    return {"format": fmt_hits / n, "category": cat_hits / n, "n": n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--adapter", default="adapters")
    parser.add_argument("--base-only", action="store_true")
    args = parser.parse_args()

    rows = load_eval_rows()
    print(f"eval set: {len(rows)} held-out rows from the deep dive\n")

    base = run_model(rows, args.model, adapter=None)
    print(f"BASE     format {base['format']:.0%}   category {base['category']:.0%}")

    if args.base_only:
        return
    if not Path(args.adapter).exists():
        raise SystemExit(f"no adapter at {args.adapter}/ — run ./train.sh first")

    tuned = run_model(rows, args.model, adapter=args.adapter)
    print(f"+ LoRA   format {tuned['format']:.0%}   category {tuned['category']:.0%}")

    verdict = (
        "SHIP: the adapter beats base on both axes"
        if tuned["format"] > base["format"] and tuned["category"] >= base["category"]
        else "DO NOT SHIP: no clear win — grow the training set (PLAN.md phase 2) and retrain"
    )
    print(f"\n{verdict}")


if __name__ == "__main__":
    main()
