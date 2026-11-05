# BrainTab-Benchmark

Code for the paper **"Frozen Tabular Foundation Models for Functional-Connectivity Phenotypes: A Leakage-Controlled
Benchmark"** (Qoseen Zahra, Zhonghua Wan, Yifei He, Ye Wu; School of Computer Science and Engineering,
Nanjing University of Science and Technology; submitted to *Computers in Biology and Medicine*).

The benchmark asks whether a **frozen** tabular foundation model (TabICL; TabICLv2 as a sensitivity model) can
extract disorder-related information from resting-state functional-connectivity (FC) phenotypes when labels are limited and
acquisition sites differ. It compares TabICL with logistic regression, random forest, XGBoost and TabM on ABIDE-I
(autism; N = 873, 17 sites) and ADHD-200 (N = 716, 7 sites), with every data-dependent step fitted inside the
training fold, identical folds and context draws for all models, leave-one-site-out testing, few-shot learning,
permutation tests and demographic/head-motion controls. Everything runs on a CPU; no GPU is needed.

## Repository layout

| Path | Purpose |
|---|---|
| `braintab/data.py` | ABIDE-I (PCP, C-PAC, CC200) loader and quality-control rule |
| `braintab/features.py` | Pearson / Fisher-z connectivity vectors |
| `braintab/compress.py` | PCA-k, PLS-k, **PLS-auto** (nested choice of k), cross-fitted PLS |
| `braintab/harmonize.py` | Training-fold-only ComBat; label-free per-site standardization |
| `braintab/models.py` | LR, LR-full, RF, XGBoost, TabM, TabICL (v1, v2) wrappers |
| `braintab/groups.py` | CC200 → anatomical groups (AAL maximum overlap) |
| `braintab/metrics.py` | accuracy, balanced accuracy, AUC, Brier score, ECE |
| `scripts/01_build_features.py` | ABIDE-I ROI time series → `data/abide_features.npz` |
| `scripts/08_build_adhd_full.py` | ADHD-200 Athena CC200 time series → `data/adhd_full_features.npz` |
| `scripts/11_build_aal_features.py` | AAL-atlas sensitivity features for ABIDE-I |
| `scripts/02_run_benchmark.py` | All experiments (E1–E5 and supporting analyses); resumable |
| `analysis/rev_analyze.py` | Statistics, tables and figures of the paper |
| `analysis/pooled_loso.py` | Pooled out-of-site AUC / balanced accuracy with bootstrap CIs |
| `scripts/03_…10_*.py` | Additional analyses and figure scripts |

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The data are public and de-identified, but they are **not redistributed here**; obtain them from the original
distributors under their data-use terms:

* ABIDE-I Preprocessed Connectomes Project derivatives (C-PAC, `nofilt_noglobal`, `rois_cc200`; optional `rois_aal`) and
  the phenotype file `Phenotypic_V1_0b_preprocessed1.csv`.
* ADHD-200 Neuro Bureau Athena derivatives (`ADHD200_CC200_TCs_filtfix.tar.gz`) and phenotypes from NITRC.

## Reproducing the paper

```bash
python scripts/01_build_features.py <ABIDE rois_cc200 dir> <Phenotypic_V1_0b_preprocessed1.csv> data
python scripts/08_build_adhd_full.py <extracted ADHD-200 dir> --pheno adhd200_preprocessed_phenotypics.tsv

python scripts/02_run_benchmark.py main        # E1 main benchmark (30 identical splits)
python scripts/02_run_benchmark.py fewshot     # E2 (supervised PLS); fewshot_u = semi-supervised PCA-100
python scripts/02_run_benchmark.py loso        # E3 inductive LOSO; label-free adaptation included
python scripts/02_run_benchmark.py groups      # E4 feature groups
python scripts/02_run_benchmark.py adhdmixed   # E5a (five mixed-diagnosis sites); also adhdfull, adhdfull_loso,
                                               # adhdfull_fewshot, adhdfull_perm, adhdmixed_perm
python scripts/02_run_benchmark.py perm        # permutation tests; also demo / adhddemo (age + sex control)

# AAL-atlas sensitivity analysis
python scripts/11_build_aal_features.py <download dir>
BT_ABIDE_FEATURES=abide_features_aal.npz BT_REPS=plsauto,full,pca100 python scripts/02_run_benchmark.py main
```

Folds and context draws are generated from fixed seeds (E1: `RepeatedStratifiedKFold(10 folds, 3 repeats,
seed 2026)` on site × diagnosis; E2: `StratifiedKFold(5, seed 7)` with draw seeds that depend only on fold, n and
draw), so every model sees exactly the same training and test participants.

Useful environment variables: `BT_RESULTS` (output folder), `BT_THREADS` (CPU threads), `BT_TABICL_NEST`
(TabICL ensemble size, default 32), `BT_SPLITS` (e.g. `0-9`, to run splits in parallel workers), `BT_REPS`
(representations), `BT_NOBUDGET=1` (all models on all splits), `BT_NPERM`, `BT_NPERM_ADHD`, `BT_PERM_START`,
`BT_FOLDS`, `BT_ABIDE_FEATURES`.

## Leakage control

Standardization, ComBat, PCA/PLS and the choice of the number of PLS components are fitted **inside each training
fold only**. Diagnosis is never a ComBat covariate. Per-site standardization uses only unlabeled features and is
reported separately as label-free target-site adaptation.

## Citation

If you use this code, please cite the paper (citation will be added after publication).

## License

MIT License (see `LICENSE`).
