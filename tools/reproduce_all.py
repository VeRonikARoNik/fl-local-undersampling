#!/usr/bin/env python3
"""Regenerate every table and figure of the paper from the stored results, then compare with the published tables.

Runs the scripts in analysis/ in the order given in docs/reproduce.md and finally tools/compare_with_published.py.
Works the same on Windows, macOS and Linux. Takes about 10 minutes, most of it in rescore_vanished.py.

Usage, from the root of the repository:

    python tools/reproduce_all.py                  # full sequence
    python tools/reproduce_all.py --skip-rescore   # reuse the stored re-scored runs instead of re-training them

Tables are written to analysis/ and figures to analysis/figures/; the published versions in results/ and figures/ are
not changed. Two steps need the raw data sets: data_summary.py (all nine, including UNSW-NB15) and partition_car.py (Car
Evaluation). When they are missing, the script uses the stored tables from results/ instead and says so.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
RESULTS = ROOT / "results"
DATA = ROOT / "datasets_original"

NINE_DATA_SETS = ["unsw-nb15", "rt-iot2022", "letter+recognition", "dry+bean+dataset", "crowdsourced+mapping",
                  "statlog+landsat+satellite", "car+evaluation", "statlog+vehicle+silhouettes",
                  "aviaere_influenza_fallliste"]

AUDIT_INPUTS = ["orig_audit_all.csv", "us_audit_RT_UNSW_nm.csv"]  # plus results/us_audit_classes_*.csv

STEPS = [
    # client audit used in the paper
    "build_audit_orig.py", "make_table3.py", "enn_examples.py",
    # correction of the 360 affected runs and the analysed data set
    "rescore_vanished.py", "build_clean.py",
    # statistics
    "anova.py", "nonparam.py", "tables.py", "stats_extra.py", "ci_sensitivity.py", "sensitivity_without_unsw.py",
    # data sets
    "data_summary.py", "partition_car.py",
    # figures
    "make_figs_v3.py", "make_fig_algorithm.py", "make_fig1_stages.py", "make_figs_extra.py",
    "make_figG_retention.py", "make_figH_kept_delta.py",
    # Section 8: interactions and the three paths
    "interactions.py", "paths_compare.py", "make_figs_ix.py",
]


def copy_stored(names, reason):
    for name in names:
        shutil.copy2(RESULTS / name, ANALYSIS / name)
    print(f"  stored  {', '.join(names)} copied from results/ ({reason})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate the tables and figures of the paper.")
    parser.add_argument("--skip-rescore", action="store_true",
                        help="copy results/rescored_vanished_*.csv instead of running rescore_vanished.py")
    args = parser.parse_args()

    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    audit = AUDIT_INPUTS + sorted(p.name for p in RESULTS.glob("us_audit_classes_*.csv"))
    copy_stored(audit, "inputs of the client audit")

    have_all_data = all((DATA / d).is_dir() and any((DATA / d).iterdir()) for d in NINE_DATA_SETS)
    have_car = (DATA / "car+evaluation").is_dir()
    start = time.time()
    for script in STEPS:
        if script == "rescore_vanished.py" and args.skip_rescore:
            copy_stored(["rescored_vanished_runs.csv", "rescored_vanished_clients.csv"], "--skip-rescore")
            continue
        if script == "data_summary.py" and not have_all_data:
            copy_stored(["data_summary.csv", "class_counts.csv"],
                        "not all nine raw data sets in datasets_original/, see datasets/README.md")
            continue
        if script == "partition_car.py" and not have_car:
            copy_stored(["partition_car_f1.csv"], "Car Evaluation not unpacked, run tools/unpack_datasets.py")
            continue
        t0 = time.time()
        run = subprocess.run([sys.executable, script], cwd=ANALYSIS, env=env, capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
        if run.returncode != 0:
            print(f"  FAILED  {script}\n")
            print("\n".join((run.stdout + run.stderr).splitlines()[-25:]))
            return 1
        print(f"  ok      {script} ({time.time() - t0:.0f} s)")

    print(f"\nAll steps done in {(time.time() - start) / 60:.1f} min. "
          "Tables: analysis/, figures: analysis/figures/.\n")
    print("Comparison with the published tables:")
    return subprocess.run([sys.executable, str(ROOT / "tools" / "compare_with_published.py")], env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
