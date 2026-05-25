"""Deterministically shuffle a JSONL file. Used to inject seed variance into
NTK-Mirror's fit (fit itself has no --seed knob — randomness enters via the
batch ordering during the gate-scoring pass and during training)."""

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    rows = [l for l in open(args.inp) if l.strip()]
    random.Random(args.seed).shuffle(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(rows), encoding="utf-8")
    print(f"shuffled {len(rows)} rows with seed={args.seed} → {out}")


if __name__ == "__main__":
    main()
