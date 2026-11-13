# Analysis scripts

[Back to the documentation index](README.md)

All scripts are in [`analysis/`](../analysis) and are run from that folder. They write their tables to the current
folder (`analysis/`) and their figures to `analysis/figures/`; the published versions are in `results/` and `figures/`.
The order in which to run them is given in [reproduce.md](reproduce.md).

| Script | Produces |
| --- | --- |
| `rescore_vanished.py` | `rescored_vanished_runs.csv`, `rescored_vanished_clients.csv`: the 360 runs in which a class was absent from every client, re-scored over all classes from the original client files in `results/vanished_federations/` (see [known issues](known-issues.md)) |
| `build_clean.py` | `fl_results_clean.csv` (86 400 runs; documented exclusions; the 360 re-scored runs replaced) from `../results/fl_results_raw.csv` |
| `anova.py` | `anova_typeII.csv` (factorial ANOVA table) |
| `nonparam.py` | Friedman and Nemenyi (`meanranks_*.csv`, `nemenyi_*.csv`), Wilcoxon with Holm (`wilcoxon_vs_baseline_holm.csv`), `per_dataset_delta_pp.csv`, `time_ratio_vs_baseline.csv` |
| `tables.py` | `table_us_methods.csv` (methods versus the baseline, with cluster-bootstrap intervals) |
| `stats_extra.py` | residual diagnostics, ANOVA on rank-transformed F1 (`anova_raw_vs_rank.csv`), Friedman tests with the Iman-Davenport correction for all factors including K (`friedman_summary.csv`, `*_K_baseline.csv`), Wilcoxon statistics (`wilcoxon_vs_baseline_holm_full.csv`), aggregation rules versus FedAvg (`rules_vs_fedavg.csv`), `stats_extra.json` |
| `ci_sensitivity.py` | `ci_sensitivity.csv`: percentile, BCa, t and Bonferroni intervals and the Wilcoxon test over the nine data-set means |
| `sensitivity_without_unsw.py` | the same intervals and tests over the eight data sets without UNSW-NB15 (`sensitivity_without_unsw.csv`) and the factorial ANOVA without it (`anova_typeII_without_unsw.csv`) |
| `duplicates_split.py` | `duplicates_split.csv`: share of the test rows of every fold that also occur in its training part, with the same label and by features only (run on the machine that holds the split folders) |
| `data_summary.py` | `data_summary.csv`, `class_counts.csv` (data set table, class distributions); needs all nine raw data sets |
| `partition_car.py` | `partition_car_f1.csv` (stages 2 and 3 on Car Evaluation, fold 1); needs the raw Car Evaluation data |
| `audit_orig.py` | `orig_audit_<code>_<tag>.csv`: class counts of every client training file written by the original run (run on the machine that holds the split folders) |
| `compare_audit_orig.py` | `orig_audit_all.csv` (all original files) and `audit_orig_vs_recreated*.csv` (original against re-created client sets) |
| `us_audit_classes.py`, `us_audit_v2.py` | re-created client sets: class counts and sampler fallback (`us_audit_classes_*.csv`, `us_audit_RT_UNSW_nm.csv`) |
| `build_audit_orig.py` | `audit_classes_orig.csv`, `audit_sizes_orig.csv`: the client audit used in the paper (original files; fallback flags from the re-created sets) |
| `make_table3.py` | `us_audit_per_dataset.csv`, `table3_us_audit.csv` (data kept and fallback per method and data set) |
| `enn_examples.py` | per-class effect of ENN on one RT and one SVS client (`enn_examples.csv`) |
| `tie_check.py` | `tie_check_f1_K5.csv`: share of instances whose 3rd and 4th nearest neighbours are equally distant |
| `rerun_check.py` | re-runs fold 1 with K = 5 with the original code and compares with the stored results (`results/rerun/`) |
| `make_figs_v3.py` | most figures (see [figures](figures.md)) and the derived tables `class_retention_*.csv`, `local_vs_federated_f1_K5.csv`, `delta_prec_recall_f1.csv`, `client_*.csv` |
| `make_fig_algorithm.py` | the flowcharts of Algorithms 1 and 2 |
| `make_fig1_stages.py` | the stages of the experiment |
| `make_figs_extra.py` | one Car Evaluation client under every method (`example_client_CAR.csv`) and the learner by data set table |
| `make_figG_retention.py`, `make_figH_kept_delta.py` | data kept per method and data set; difference to the baseline against data kept (`delta_vs_kept.csv`) |
| `interactions.py` | how the elements of the study affect the result and each other (Section 8): `main_effects.csv`, `anova_all2way.csv` (ANOVA with all ten two-way interactions), `ix_method_*.csv` (effect of each method per rule, K and learner), `ix_rule_*.csv` (each rule against FedAvg per data set, K and condition), `ix_dataset_properties.csv` (data-set properties and the loss of the harmful methods) |
| `audit_path1.py` | `path1_audit_<code>.csv`: class counts of the test halves and training parts of path 1 and of the baseline split files (run on the machine that holds the split folders) |
| `paths_compare.py` | the three paths of the original experiment (Section 8.5): `paths_by_method.csv`, `paths_by_dataset_method.csv`, `path1_testsets.csv` (reads `../results/fl_results_raw.csv` and `../results/path1_audit/`) |
| `make_figs_ix.py` | the figures of Section 8 (`figIx*`, `figPaths*`) |
| `win_order.py`, `figstyle.py` | helpers: the file order of the original Windows run (see [known issues](known-issues.md)); common figure style |

## Environment variables

Scripts that re-create client sets import the original split code and read the raw data. Their locations default to the
folders of this repository and can be changed with environment variables:

| Variable | Default | Used by |
| --- | --- | --- |
| `DATA_DIR` | `../datasets_original/` | `data_summary.py`, `partition_car.py`, `us_audit_classes.py`, `us_audit_v2.py`, `tie_check.py`, `rerun_check.py` |
| `SPLIT_SCRIPT` | `../src/original/federated_data_split_path2.py` | `data_summary.py`, `us_audit_classes.py`, `us_audit_v2.py`, `tie_check.py` |
| `SPLIT_BASE` | `../src/original/federated_data_split.py` | `partition_car.py` |
| `SRC_DIR` | `../src/original` | `rescore_vanished.py`, `rerun_check.py` |
| `RAW` | `../results/fl_results_raw.csv` | `rescore_vanished.py`, `build_clean.py`, `paths_compare.py`, `rerun_check.py` |
| `FED_DIR` | `../results/vanished_federations` | `rescore_vanished.py` |
| `SPLITS` | `~/data_splits` (`audit_orig.py`), `data_splits` (`audit_path1.py`, `duplicates_split.py`) | folder with the split files written by the original run |
| `OUT` | `.` | `audit_orig.py`, `audit_path1.py`, `duplicates_split.py` |
| `FOLDS`, `METHODS`, `TAG` | all folds and methods, tag `all` | `us_audit_classes.py`, `us_audit_v2.py` |
| `FOLD`, `K`, `DATASETS` | fold 1, K = 5, all data sets | `rerun_check.py` |
