#!/usr/bin/env python3
"""Verify the environment for local-lora. No network, no key, no downloads."""

import platform
import sys
from pathlib import Path

OK, BAD = "  \033[32m✓\033[0m", "  \033[31m✗\033[0m"
failures = 0


def check(label: str, ok: bool, fix: str = "") -> None:
    global failures
    print(f"{OK if ok else BAD} {label}" + ("" if ok else f"\n      fix: {fix}"))
    failures += 0 if ok else 1


print("local-lora setup check\n")

check(
    f"Python {sys.version_info.major}.{sys.version_info.minor} (need 3.11+)",
    sys.version_info >= (3, 11),
    "install a newer Python",
)
check(
    f"Apple Silicon macOS (got {platform.system()}/{platform.machine()})",
    platform.system() == "Darwin" and platform.machine() == "arm64",
    "MLX only runs on Apple Silicon — use Unsloth/QLoRA on a CUDA box instead",
)

try:
    import mlx.core

    check(f"package: mlx ({getattr(mlx.core, '__version__', '?')})", True)
except ImportError:
    check("package: mlx", False, "pip install -r requirements.txt")

try:
    import mlx_lm  # noqa: F401

    ver = getattr(mlx_lm, "__version__", "?")
    check(f"package: mlx-lm ({ver})", True)
except ImportError:
    check("package: mlx-lm", False, "pip install -r requirements.txt")

dive = Path(__file__).parent.parent / "DeepDives" / "fine-tuning-deep-dive" / "datasets"
check(
    "deep-dive datasets found (train + eval)",
    (dive / "support_train.jsonl").exists() and (dive / "support_eval.jsonl").exists(),
    f"expected {dive} — this repo assumes it sits next to DeepDives/",
)
check(
    "converted data present (data/train.jsonl)",
    (Path(__file__).parent / "data" / "train.jsonl").exists(),
    "python convert_data.py",
)

print()
if failures:
    sys.exit(f"{failures} check(s) failed — fix the items above and rerun.")
print("All good. Next:  ./train.sh --iters 10   (smoke run; downloads the base model)")
