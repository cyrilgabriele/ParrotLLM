"""Report token and window counts for a flat binary token file.

This inspects the file size only, so it is safe to run on very large `.bin`
files without loading them into RAM.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count tokens in a flat binary token file and report the window "
            "counts relevant to ParrotLLM training."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to the .bin token file, for example data/exp_c/train.bin",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=1024,
        help="Model context length used to derive window counts (default: 1024).",
    )
    parser.add_argument(
        "--dtype",
        choices=("uint16", "uint32"),
        default="uint16",
        help="Token storage dtype in the binary file (default: uint16).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = args.path

    if not path.exists():
        raise SystemExit(f"File does not exist: {path}")
    if not path.is_file():
        raise SystemExit(f"Path is not a file: {path}")
    if args.context_length <= 0:
        raise SystemExit("--context-length must be positive.")

    dtype_bytes = 2 if args.dtype == "uint16" else 4
    size_bytes = path.stat().st_size

    if size_bytes % dtype_bytes != 0:
        raise SystemExit(
            f"File size {size_bytes} is not divisible by {dtype_bytes} bytes "
            f"for dtype {args.dtype}."
        )

    n_tokens = size_bytes // dtype_bytes
    sliding_window_starts = max(0, n_tokens - args.context_length)
    stride_half = max(1, args.context_length // 2)
    eval_windows = 1 + max(0, sliding_window_starts - 1) // stride_half if sliding_window_starts > 0 else 0

    print(f"path: {path}")
    print(f"size_bytes: {size_bytes:,}")
    print(f"dtype: {args.dtype} ({dtype_bytes} bytes/token)")
    print(f"context_length: {args.context_length:,}")
    print(f"tokens: {n_tokens:,}")
    print(f"sliding_window_starts: {sliding_window_starts:,}")
    print(f"eval_windows (stride={stride_half}): {eval_windows:,}")


if __name__ == "__main__":
    main()
