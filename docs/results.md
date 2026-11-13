# Result files

[Back to the documentation index](README.md)

All results are in [`results/`](../results). Metrics are stored as fractions; the paper reports them multiplied by 100.

## Runs

| File | Content |
| --- | --- |
| `results/fl_results_raw.csv` | all 235 200 runs as written by `src/original/fl_training.py`, one row per run, including the excluded conditions |
| `results/fl_results_clean.csv` | the 86 400 runs analysed in the paper (written by `analysis/build_clean.py`; the 360 affected runs re-scored, see [known issues](known-issues.md)) |

Columns of both files:

| Column | Values |
| --- | --- |
| `path_type` | `baseline`, `us_after` (path 2) and, in the raw file only, `us_before` (path 1) |
| `us_method` | `none`, `RUS`, `RUS25`, `RUS50`, `RUS75`, `NearMiss` (NearMiss-1), `NM3` (NearMiss-3), `ENN`, `AllKNN`, `TomekLinks`; raw file also `NCL` |
| `dataset` | `UNSW`, `RT`, `LR`, `DRY`, `CM`, `SLS`, `CAR`, `SVS`, `AIH`; raw file also `SOY` |
| `fold`, `n_clients` | 1 to 5; K = 3 to 10 |
| `fl_algorithm` | `FedAvg`, `WeightedFedAvg`, `FedMedian`, `FedTrimmedMean`, `FedProx`, `FedBoost`; raw file also `FedBagging` (identical to FedAvg) |
| `local_model` | `DT`, `RF`, `GB`, `XGB` |
| `train_time_sec` | time to train the K local models of the federation and compute their probability vectors, plus the aggregation, in seconds; the undersampling itself is done beforehand by the split script and is not included |
| `val_*`, `test_*` | accuracy, F1, precision and recall (macro and weighted) and AUC-ROC on the validation and test halves |

Load the analysed runs, for example:

```python
import pandas as pd

runs = pd.read_csv("results/fl_results_clean.csv")
print(runs.groupby("us_method").test_f1_macro.mean().mul(100).round(2))
```

## Derived results

```
results/
    fl_results_raw.csv           235 200 runs as produced by fl_training.py (one row per run)
    fl_results_clean.csv         86 400 runs used in the paper (360 runs re-scored, see known issues)
    rescored_vanished_runs.csv   the 360 re-scored runs: stored value, value re-computed with the original scoring
                                 (equal to the stored one in all 360 runs) and corrected value, for every metric
    rescored_vanished_clients.csv  local models of these runs on their own
    vanished_federations/        original client, validation and test files of the 15 affected federations
    anova_*.csv, stats_extra.json, resid_sd_*.csv, friedman_summary.csv, meanranks_*.csv, nemenyi_*.csv,
    wilcoxon_*.csv, rules_vs_fedavg.csv, table_us_methods.csv, ci_sensitivity.csv, per_dataset_delta_pp.csv,
    time_ratio_vs_baseline.csv, delta_prec_recall_f1.csv                        statistics
    duplicates_split.csv, sensitivity_without_unsw.csv, anova_typeII_without_unsw.csv
                                 test rows with an exact copy in the training part of their fold, and the
                                 analysis repeated without UNSW-NB15
    data_summary.csv, class_counts.csv, partition_car_f1.csv, client_*.csv    data sets and client sets
    orig_audit_all.csv, audit_classes_orig.csv, audit_sizes_orig.csv, table3_us_audit.csv, us_audit_per_dataset.csv,
    class_retention_*.csv, delta_vs_kept.csv, example_client_CAR.csv, enn_examples.csv        audit of the client data
    us_audit_classes_*.csv, us_audit_RT_UNSW_nm.csv, audit_orig_vs_recreated*.csv, tie_check_f1_K5.csv
                                 re-created client sets and their comparison with the original files
    rerun/rerun_runs_f1_K5.csv   stored and re-computed metrics of the 2 160 runs of fold 1 with K = 5
    rerun/rerun_clients_f1_K5.csv, rerun/rerun_agreement_f1_K5.csv, local_vs_federated_f1_K5.csv
                                 local models on their own and agreement of the aggregation rules
    main_effects.csv, anova_all2way.csv, ix_*.csv                 interactions of all elements (Section 8)
    paths_by_method.csv, paths_by_dataset_method.csv, path1_testsets.csv, path1_audit/path1_audit_*.csv
                                 path 1 (undersampling before the CV split) against path 2 and the baseline
```

The most used summary is `results/table_us_methods.csv`: for every undersampling method, the mean test F1-macro, the
mean rank over the data sets, the mean difference to the baseline over matched runs (`delta`) with its 95%
cluster-bootstrap confidence interval (`ci_lo`, `ci_hi`), the Holm-adjusted p-value of the Wilcoxon signed-rank test
(`p_holm`), the rank-biserial correlation (`rbc`), the share of matched runs in which the method is worse than the
baseline (`share_worse`, in %) and the reduction of the local training time (`time_saving`, in %).

## Client audit

The client audit (`audit_*_orig.csv`) counts the classes in all 23 400 client training files of the original run (9
data sets, 5 folds, 52 client sets per fold, 10 conditions). In `audit_sizes_orig.csv`, `min_class` is the size of the
smallest class on the client before undersampling and `fallback` marks clients on which the method could not be applied
(the client then keeps its data); the fallback flag comes from the re-created client sets, and every client set marked
as a fallback is unchanged in the original files. The re-created per-class audit covers all folds of seven data sets and
fold 1 of RT and UNSW.

## Reproduction check

To verify that the released code reproduces the stored results, the client data were re-created and a sample of the
runs was re-run on a second system (Linux, Python 3.13) and compared with the original run:

- `audit_orig_vs_recreated.csv` and `audit_orig_vs_recreated_summary.csv`: re-created client sets against the original
  client files;
- `tie_check_f1_K5.csv`: share of instances whose 3rd and 4th nearest neighbours are equally distant, which explains
  where neighbourhood-based undersamplers can choose differently on another system;
- `rerun/`: the re-run of fold 1 with K = 5 (2 160 runs) with the original code, compared with the stored results.

The size of the differences is reported in the section "Implementation and reproducibility" of the paper and summarised
in [known issues](known-issues.md).
