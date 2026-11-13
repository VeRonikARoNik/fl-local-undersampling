# Reproducing the results

[Back to the documentation index](README.md)

There are three levels, from minutes to days:

1. [Regenerate the tables and figures](#regenerate-the-tables-and-figures) of the paper from the stored results (about
   10 minutes).
2. [Re-run the federated experiment](#re-run-the-federated-experiment) from the raw data (several days on a laptop).
3. [Check the reproduction](#reproduction-check) of the original run on your system (about two hours each step).

## Install

Python 3.11 or newer. The experiment ran on Python 3.11 under Windows; the reproduction check and the test of this
repository used Python 3.13 under Linux, with the same package versions.

```bash
git clone https://github.com/VeRonikARoNik/fl-local-undersampling.git
cd fl-local-undersampling
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-experiment.txt -r requirements-analysis.txt
python tools/unpack_datasets.py    # optional for level 1, needed for levels 2 and 3
```

`requirements-experiment.txt` pins the packages of the experiment (scikit-learn 1.8.0, imbalanced-learn 0.14.1,
xgboost 3.2.0, scipy 1.17.0, numpy 2.4.2, pandas 3.0.2); `requirements-analysis.txt` adds statsmodels,
scikit-posthocs and matplotlib.

## Regenerate the tables and figures

### One command

```bash
python tools/reproduce_all.py
```

The script runs every analysis script in the right order, then compares each re-created table with the published one
(`tools/compare_with_published.py`). Tables are written to `analysis/` and figures to `analysis/figures/`; the published
versions in `results/` and `figures/` are not changed. Expected end of the output:

```
Tables compared: 59, different: 0
```

On the test system (Linux, two CPU cores) the whole sequence took 6.4 minutes, of which 5.1 minutes were spent in
`rescore_vanished.py`, which re-trains the models of the 15 affected federations. `--skip-rescore` reuses the stored
re-scored runs instead. 54 of the 59 tables were identical byte for byte, including `fl_results_clean.csv` and
`table_us_methods.csv`; the other five (the ANOVA tables and the residual standard deviations) differed only by
floating-point noise, at most 5e-11 relative. All 30 figures were re-drawn.

Two steps need raw data. `data_summary.py` reads all nine data sets, including UNSW-NB15, which is not distributed
here; `partition_car.py` reads Car Evaluation. When the data are missing, `reproduce_all.py` uses the stored tables
from `results/` for these two steps and says so.

### Step by step

The same sequence by hand, run from `analysis/` (bash; for PowerShell see the note below):

```bash
cd analysis
# 1. client audit used in the paper (its inputs are stored in ../results/)
cp ../results/orig_audit_all.csv ../results/us_audit_classes_*.csv ../results/us_audit_RT_UNSW_nm.csv .
python build_audit_orig.py
python make_table3.py
python enn_examples.py
# 2. correction of the 360 affected runs and the analysed data set
python rescore_vanished.py       # reads ../results/vanished_federations/ and ../results/fl_results_raw.csv
python build_clean.py            # writes fl_results_clean.csv (86 400 runs)
# 3. statistics
python anova.py
python nonparam.py
python tables.py
python stats_extra.py
python ci_sensitivity.py
python sensitivity_without_unsw.py
# 4. data sets (need ../datasets_original/; without UNSW-NB15 copy the stored tables instead:
#    cp ../results/data_summary.csv ../results/class_counts.csv .)
python data_summary.py
python partition_car.py
# 5. figures (make_figs_v3.py reads the re-run of fold 1 with K = 5 from ../results/rerun/)
python make_figs_v3.py
python make_fig_algorithm.py
python make_fig1_stages.py
python make_figs_extra.py
python make_figG_retention.py
python make_figH_kept_delta.py
# 6. Section 8: interactions and the three paths
python interactions.py
python paths_compare.py
python make_figs_ix.py
cd ..
python tools/compare_with_published.py
```

**PowerShell.** Every line works as written except the copy commands, which take a comma-separated list:

```powershell
Copy-Item ..\results\orig_audit_all.csv, ..\results\us_audit_classes_*.csv, ..\results\us_audit_RT_UNSW_nm.csv -Destination .
Copy-Item ..\results\data_summary.csv, ..\results\class_counts.csv -Destination .
```

What every script produces is listed in [analysis-scripts.md](analysis-scripts.md), and which script draws which
figure in [figures.md](figures.md).

## Re-run the federated experiment

Re-running all conditions of the paper takes several days on a laptop. The scripts in `src/fixed/` contain the
corrections described in [known issues](known-issues.md); `src/original/` is the code exactly as it ran.

1. **Data.** `python tools/unpack_datasets.py`, then add UNSW-NB15 to `datasets_original/unsw-nb15/` (see
   [datasets/README.md](../datasets/README.md)).
2. **Base path.** Set `BASE_PATH` at the top of `src/fixed/federated_data_split.py`,
   `src/fixed/federated_data_split_path2.py` and `src/fixed/fl_training.py` to the folder of this repository (the folder
   that contains `datasets_original/`). The split scripts write `data_splits/` there.
3. **Splits.** Baseline splits (5-fold CV, validation and test halves, K = 3 to 10 clients), then per-client
   undersampling (path 2); the second step takes one to three hours:

   ```bash
   python src/fixed/federated_data_split.py
   python src/fixed/federated_data_split_path2.py
   ```

   On Linux and macOS, run both through `python tools/run_windows_order.py src/fixed/<script>.py` to read the data
   files in the order of the original Windows run; otherwise the folds of Statlog (Vehicle Silhouettes) and Crowdsourced
   Mapping differ from the original ones (see [known issues](known-issues.md)).
4. **Training.** One-shot federated training of the analysed conditions, written to a separate folder:

   ```bash
   python src/fixed/fl_training.py --paths baseline us_after --results results_rerun \
       --datasets UNSW RT LR DRY CM SLS CAR SVS AIH \
       --us RUS RUS25 RUS50 RUS75 NearMiss NM3 ENN AllKNN TomekLinks \
       --fl FedAvg WeightedFedAvg FedMedian FedTrimmedMean FedProx FedBoost
   ```

   The run is resumable: after an interruption, the same command skips the runs already in
   `results_rerun/fl_results.csv`. To work in batches, restrict it further, for example `--datasets AIH CAR SVS` (small
   and fast), `--models DT` (the fastest learner), `--folds 1` or `--clients 3 5 10`. Without these options the script
   also runs the conditions that the paper excludes: path 1 (`us_before`, part of the default `--paths`), the
   Neighbourhood Cleaning Rule, Soybean (Large) and FedBagging.

Results are written as one row per run in the format of `results/fl_results_raw.csv` (see [results.md](results.md)).
Neighbourhood-based undersamplers and gradient boosting resolve ties in a way that depends on the compiled code, so
single runs on another system can differ from the stored ones; the size of these differences is reported in the paper.

## Reproduction check

These steps re-create the inputs of the client audit and repeat the comparison of the paper between the original run and
a second system. They need the raw data sets in `datasets_original/` and, for `audit_orig.py`, the split folders of the
original run. Run from `analysis/`:

```bash
# class counts of the original client files, on the machine that holds the split folders
SPLITS=/path/to/data_splits python audit_orig.py AIH        # one call per data set (methods and folds can be split)
python compare_audit_orig.py /folder/with/orig_audit_files
# re-created client sets (fallback flags and cross-check; about two hours)
python us_audit_classes.py AIH CAR SVS SLS CM DRY LR
FOLDS=1 TAG=f1 python us_audit_classes.py RT UNSW
METHODS=NearMiss,NM3 TAG=nm python us_audit_v2.py RT UNSW
python tie_check.py
# test rows with an exact copy in the training part of their fold, on the machine that holds the split folders
SPLITS=/path/to/data_splits python duplicates_split.py
# re-run of fold 1 with K = 5 with the original code (about two hours; writes rerun/*.csv)
mkdir -p rerun && cd rerun && python ../rerun_check.py && cd ..
```

Scripts that re-create client sets import the original split code and read the raw data. Their locations default to
`../src/original/` and `../datasets_original/` and can be changed with environment variables (see
[analysis-scripts.md](analysis-scripts.md#environment-variables)).
