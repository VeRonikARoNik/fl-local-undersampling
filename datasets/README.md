# Data sets

All nine data sets of the study are public. This folder holds eight of them as obtained from their sources, one zip
archive per data set. Each archive contains one folder with the name that the split scripts expect.

UNSW-NB15 is **not** included: its terms grant free use for academic research but do not cover redistribution.
Download it from its source (see below).

## Unpack

From the root of the repository:

```bash
python tools/unpack_datasets.py
```

This creates `datasets_original/<folder>/` for every archive (the folder is ignored by git) and reminds you about
UNSW-NB15. Then put `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` from
https://research.unsw.edu.au/projects/unsw-nb15-dataset into `datasets_original/unsw-nb15/`.

You can also unzip the archives by hand into `datasets_original/`, or download every data set from its source into
`datasets_original/<folder>`.

## The nine data sets

The values describe the data sets after preparation (stage 1: merge the source files, remove identifier columns,
columns that reveal the label and rows with missing values). IR is the imbalance ratio, the size of the largest class
divided by that of the smallest.

| Code | Data set | Instances | Features | Classes | IR | Label | Folder | Source (licence) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UNSW | UNSW-NB15 (training and testing CSV) | 257 673 | 42 | 2 | 1.8 | `label` | `unsw-nb15/` | UNSW Canberra, https://research.unsw.edu.au/projects/unsw-nb15-dataset (free for academic research; not redistributed here) |
| RT | RT-IoT2022 | 123 117 | 83 | 12 | 3381 | `Attack_type` | `rt-iot2022/` | UCI, [doi:10.24432/C5P338](https://doi.org/10.24432/C5P338) (CC BY 4.0) |
| LR | Letter Recognition | 20 000 | 16 | 26 | 1.1 | `lettr` | `letter+recognition/` | UCI, [doi:10.24432/C5ZP40](https://doi.org/10.24432/C5ZP40) (CC BY 4.0) |
| DRY | Dry Bean | 13 611 | 16 | 7 | 6.8 | `Class` | `dry+bean+dataset/` | UCI, [doi:10.24432/C50S4B](https://doi.org/10.24432/C50S4B) (CC BY 4.0) |
| CM | Crowdsourced Mapping | 10 845 | 28 | 6 | 75.1 | `class` | `crowdsourced+mapping/` | UCI, [doi:10.24432/C56315](https://doi.org/10.24432/C56315) (CC BY 4.0) |
| SLS | Statlog (Landsat Satellite) | 6 435 | 36 | 6 | 2.4 | column 36 | `statlog+landsat+satellite/` | UCI, [doi:10.24432/C55887](https://doi.org/10.24432/C55887) (CC BY 4.0) |
| CAR | Car Evaluation | 1 728 | 6 | 4 | 18.6 | `class` | `car+evaluation/` | UCI, [doi:10.24432/C5JP48](https://doi.org/10.24432/C5JP48) (CC BY 4.0) |
| SVS | Statlog (Vehicle Silhouettes) | 846 | 18 | 4 | 1.1 | column 18 | `statlog+vehicle+silhouettes/` | UCI, [doi:10.24432/C5HG6N](https://doi.org/10.24432/C5HG6N); the original documentation asks users to acknowledge the Turing Institute, Glasgow |
| AIH | Avian influenza A(H5N1) in humans, line list | 294 | 15 | 4 | 9.2 | `outcome` | `aviaere_influenza_fallliste/` | Robert Koch Institute, [doi:10.25646/7661](https://doi.org/10.25646/7661) (CC BY 3.0 DE) |

Further details of every data set after preparation (numeric and categorical features, size of the largest and the
smallest class, share of duplicate rows) are in [`results/data_summary.csv`](../results/data_summary.csv), and the class
counts in [`results/class_counts.csv`](../results/class_counts.csv).

Soybean (Large) was also processed in the original run but is excluded from the analysis (see
[known issues](../docs/known-issues.md)); it is not included here.

## Licences and citation

The archives are redistributed under the licences of their sources listed above; the MIT License of this repository
covers the code only. When you reuse a data set, please cite its original source (the references are listed in the
paper and on the source pages).
