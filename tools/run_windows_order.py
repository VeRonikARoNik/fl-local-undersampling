#!/usr/bin/env python3
"""Run a script with the file order of the original Windows run.

The split scripts concatenate the data files of a folder in the order returned by pathlib's glob, which is alphabetical
on Windows (where the experiment ran) but arbitrary on Linux and macOS. For the data sets that come as several files
(Statlog Vehicle Silhouettes, Crowdsourced Mapping) this changes the folds and client sets. This wrapper imports
analysis/win_order.py, which restores the Windows order, and then runs the given script with its arguments.

Usage, from the root of the repository (only needed on Linux and macOS):

    python tools/run_windows_order.py src/fixed/federated_data_split.py
    python tools/run_windows_order.py src/fixed/federated_data_split_path2.py
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import win_order  # noqa: E402,F401  (patches pathlib.Path.glob and rglob)


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script)] + sys.argv[2:]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
