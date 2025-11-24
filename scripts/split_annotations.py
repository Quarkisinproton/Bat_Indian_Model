#!/usr/bin/env python3
"""Split a simple list-style annotations.json into stratified train/val/test files.

The input annotations file should be a JSON list with entries like:
  {"filename": "file.wav", "species": "Pip ceylonicus", "count": 1}

Example:
  python scripts/split_annotations.py --input data/annotations.json --out_dir data/splits --train 0.7 --val 0.15 --test 0.15 --seed 42

This will write `train.json`, `val.json`, `test.json` under `data/splits`.
"""
import argparse
import json
from pathlib import Path
import random
from collections import defaultdict


def stratified_split(entries, train_frac, val_frac, test_frac, seed=42):
    random.seed(seed)
    # Group by species
    groups = defaultdict(list)
    for e in entries:
        species = e.get('species', 'unknown')
        groups[species].append(e)

    train, val, test = [], [], []
    for species, items in groups.items():
        items_copy = items[:]
        random.shuffle(items_copy)
        n = len(items_copy)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        # remaining goes to test
        n_test = n - n_train - n_val

        train.extend(items_copy[:n_train])
        val.extend(items_copy[n_train:n_train + n_val])
        test.extend(items_copy[n_train + n_val: n_train + n_val + n_test])

    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to simple annotations.json (list of entries)')
    parser.add_argument('--out_dir', default='data/splits', help='Directory to write split files')
    parser.add_argument('--train', type=float, default=0.7, help='Train fraction (default 0.7)')
    parser.add_argument('--val', type=float, default=0.15, help='Validation fraction (default 0.15)')
    parser.add_argument('--test', type=float, default=0.15, help='Test fraction (default 0.15)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    inp = Path(args.input)
    assert inp.exists(), f"Input {inp} not found"
    with open(inp, 'r') as f:
        entries = json.load(f)

    # Normalize fractions
    total = args.train + args.val + args.test
    train_frac = args.train / total
    val_frac = args.val / total
    test_frac = args.test / total

    train, val, test = stratified_split(entries, train_frac, val_frac, test_frac, seed=args.seed)

    outp = Path(args.out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    with open(outp / 'train.json', 'w') as f:
        json.dump(train, f, indent=2)
    with open(outp / 'val.json', 'w') as f:
        json.dump(val, f, indent=2)
    with open(outp / 'test.json', 'w') as f:
        json.dump(test, f, indent=2)

    print(f"Wrote splits to {outp}: train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == '__main__':
    main()
