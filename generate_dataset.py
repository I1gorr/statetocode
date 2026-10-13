"""Generate the synthetic state-diagram dataset (images, YOLO labels, ground-truth JSON).

Usage:
    python generate_dataset.py --output dataset_v2 --count 2000 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

from statemachine_pipeline.synthetic import generate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("dataset_v2"))
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()
    totals = generate_dataset(args.output, count=args.count, seed=args.seed, size=args.size)
    print(f"Wrote {sum(totals.values())} diagrams to {args.output}: {totals}")
    print(f"Train with: yolo detect train data={args.output / 'dataset.yaml'} model=yolo11n.pt imgsz={args.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
