#!/usr/bin/env python3
"""Compare the tables re-created in analysis/ with the published versions in results/.

Usage, from the root of the repository, after running the analysis scripts (see docs/reproduce.md):

    python tools/compare_with_published.py                 # relative tolerance 1e-9
    python tools/compare_with_published.py --rtol 1e-6

For every CSV file in analysis/ whose name also exists in results/ (or in one of its subfolders), the script reports
whether the two tables are identical, equal within the tolerance (floating-point noise), or different, with the largest
difference. Two numbers count as equal when |new - published| <= atol + rtol * |published|. The exit code is 0 when no
table differs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def compare(new: Path, old: Path, rtol: float, atol: float) -> tuple[str, str]:
    a = pd.read_csv(new, low_memory=False)
    b = pd.read_csv(old, low_memory=False)
    if list(a.columns) != list(b.columns):
        return "DIFFERENT", "columns differ"
    if a.shape != b.shape:
        return "DIFFERENT", f"shape {a.shape} vs {b.shape}"
    if a.equals(b):
        return "identical", ""
    worst_abs, worst_rel, notes = 0.0, 0.0, []
    for col in a.columns:
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            if not np.array_equal(np.isnan(xv), np.isnan(yv)):
                notes.append(f"{col}: missing values differ")
                continue
            ok = ~np.isnan(yv)
            diff = np.abs(xv[ok] - yv[ok])
            if not np.all(diff <= atol + rtol * np.abs(yv[ok])):
                notes.append(f"{col}: max abs diff {diff.max():.3g}")
            worst_abs = max(worst_abs, float(diff.max(initial=0.0)))
            rel = diff / np.maximum(np.abs(yv[ok]), np.finfo(float).tiny)
            worst_rel = max(worst_rel, float(rel.max(initial=0.0)))
        elif not x.astype(str).equals(y.astype(str)):
            notes.append(f"{col}: text differs")
    if notes:
        return "DIFFERENT", "; ".join(notes[:3])
    return "equal", f"max abs diff {worst_abs:.1e}, max rel diff {worst_rel:.1e}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare analysis/*.csv with the published tables in results/.")
    parser.add_argument("--rtol", type=float, default=1e-9, help="accepted relative difference")
    parser.add_argument("--atol", type=float, default=1e-12, help="accepted absolute difference (for values near 0)")
    parser.add_argument("--new", type=Path, default=ROOT / "analysis", help="folder with the re-created tables")
    parser.add_argument("--published", type=Path, default=ROOT / "results", help="folder with the published tables")
    args = parser.parse_args()

    published = {p.name: p for p in sorted(args.published.rglob("*.csv"))}
    rows, n_diff = [], 0
    for new in sorted(args.new.glob("*.csv")):
        old = published.get(new.name)
        if old is None:
            continue
        status, note = compare(new, old, args.rtol, args.atol)
        n_diff += status == "DIFFERENT"
        rows.append((status, new.name, note))
    if not rows:
        print(f"No re-created tables found in {args.new}; run the analysis scripts first.")
        return 1
    width = max(len(name) for _, name, _ in rows)
    for status, name, note in rows:
        print(f"{status:<10} {name:<{width}}  {note}")
    print(f"\nTables compared: {len(rows)}, different: {n_diff}")
    return 1 if n_diff else 0


if __name__ == "__main__":
    sys.exit(main())
