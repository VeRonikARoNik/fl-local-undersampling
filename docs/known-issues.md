# Known issues in the original code

[Back to the documentation index](README.md)

`src/original/` holds the exact code that produced `results/fl_results_raw.csv`. The paper reports the issues below and
how each was handled; [`src/fixed/`](../src/fixed) and [`patches/`](../patches) contain the corrections. Use
`src/fixed/` for new experiments.

| Issue | Effect | In the paper | Fix |
| --- | --- | --- | --- |
| `encode_y` in `fl_training.py` fitted the label encoder on the clients' training labels only and mapped every unseen validation or test label to code 0 | when undersampling removed a class from every client (15 of the 360 ENN federations: 14 on AIH, one on RT, 360 runs), the test instances of that class were scored as the first class and the class dropped out of the macro averages, which inflated F1-macro (mean of the 360 runs 84.7% instead of 63.7%) | the 360 runs were re-scored over all classes from the original client files (`rescore_vanished.py`); with the original scoring all 360 re-computed values equal the stored ones, so the models are identical; `build_clean.py` uses the corrected values | labels are coded over all classes of the training, validation and test data (`src/fixed/`); models and predictions are unchanged |
| `federated_data_split_path1.py` undersampled the whole data set before the CV split, so validation and test sets were undersampled too | test sets of the "undersampling before split" condition are not comparable | condition excluded from the main analysis; shown in Section 8.5 as an illustration (`paths_compare.py`) | undersampling is applied to the training part of each fold only (`src/fixed/`) |
| `NeighbourhoodCleaningRule` raised an error with string labels under SciPy >= 1.15 and the scripts silently kept the original data | NCL rows equal the baseline | NCL excluded | labels are integer-coded before resampling (`src/fixed/`) |
| `federated_data_split.py` read the Soybean description files (`fisher-order`, `stepp-order`, ...) as data rows | baseline Soybean split contained 94 non-data rows and 26 "classes" | Soybean excluded | description files are ignored (`src/fixed/`) |
| FedBagging is soft voting with equal weights | numerically identical to FedAvg | merged with FedAvg | not changed |
| The aggregation rules carry FL algorithm names (`FedProx`, `FedBoost`, ...) and an unused constant `FL_ROUNDS = 3` suggests rounds | could be read as iterative FL; the protocol is one-shot (each client trains once, the server combines probability vectors) | rules described by what they compute: `FedProx` = distance-weighted mean, `FedBoost` = rank-weighted mean | docstring documents the protocol and the name mapping; `FL_ROUNDS` removed (`src/fixed/`) |
| The split scripts concatenate the data files of a folder in the order returned by `pathlib.glob`, which is alphabetical on Windows but arbitrary on Linux | UNSW, CM and SVS read several files of one type at once; on a system that lists them in another order their folds change (on the Linux system used for the check: SVS and CM) | the analysis scripts import `win_order.py`, which restores the Windows order on any system | import `win_order.py` before the split code (as `analysis/*.py` do) |
| Neighbourhood-based undersamplers choose among equally distant neighbours, and gradient boosting among nearly equal splits, in a way that depends on the compiled code | re-created client sets of ENN, AllKNN and Tomek Links differ from the original files in 28% to 31% of the client sets and those of NearMiss-3 in 2.3% (only on CAR, LR, UNSW, SLS and RT, which have many exact distance ties), by a median of 1.1% of their instances counted class by class; in the re-run of fold 1 with K = 5, 20.2% of the results differ, single runs by up to 13.8% | the audit uses the original client files; the differences are quantified by `compare_audit_orig.py`, `tie_check.py` and `rerun_check.py` (Section "Implementation and reproducibility" of the paper) | not a code error |

## Excluded conditions

The raw results contain conditions that are not part of the analysis: path 1 (undersampling before the
cross-validation split), the Neighbourhood Cleaning Rule (NCL) and Soybean (Large), for the reasons above. FedBagging is
merged with FedAvg. `analysis/build_clean.py` applies exactly these exclusions.

## Other documented limitations

See the section on limitations of the paper: categorical encoding is fitted on the union of the clients' training data;
distance-based undersamplers work on unscaled features; exact duplicate rows are not removed (40.2% of UNSW-NB15); the CM
training labels are noisy according to the source; random forest and XGBoost use all CPU cores during timing; the
recorded training time does not include the undersampling step itself.

Because duplicates were kept, 45.1% of the test rows of UNSW-NB15, 9.2% of Letter Recognition and 6.6% of RT-IoT2022
have an exact copy (same features and label) in the training part of their fold, and at most 1.4% on the other data
sets ([`results/duplicates_split.csv`](../results/duplicates_split.csv), written by `analysis/duplicates_split.py`).
Absolute scores on these data sets may therefore be optimistic. Repeated without UNSW-NB15, the analysis reaches the
same conclusions ([`results/sensitivity_without_unsw.csv`](../results/sensitivity_without_unsw.csv) and
[`results/anova_typeII_without_unsw.csv`](../results/anova_typeII_without_unsw.csv), written by
`analysis/sensitivity_without_unsw.py`).

## Language of the code

Comments, docstrings and log messages in `src/` are in Polish; identifiers, command-line options, file names and the
columns of the result files are in English. The analysis scripts are commented in English.
