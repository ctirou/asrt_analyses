# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research analysis pipeline for an MEG (magnetoencephalography) neuroscience study on learning regularities in noise. Companion code for the manuscript: "Learning regularities in noise engages both neural predictive activity and representational changes" (Tirou et al., [doi:10.1101/2025.08.18.670891](https://doi.org/10.1101/2025.08.18.670891)).

Analyzes MEG data from 15 subjects (`sub01`-`sub15`) performing a visuomotor alternating serial reaction time (ASRT) task across 5 epochs (practice + 4 experimental blocks).

## Environment Setup

```bash
conda env create -f environment.yml
conda activate asrt_analysis
```

Key dependencies: MNE-Python, scikit-learn, autoreject, mne-rsa, nilearn, pingouin, bambi/pymc, joblib.

Python scripts are designed to be run interactively (Jupyter) or as modules. Quarto `.qmd` files (in `05_gam/`) require RStudio.

## Running Scripts

**Locally (interactive):** Run Python scripts in Jupyter Notebook/Lab.

**On HPC cluster:** Submit via SLURM scripts in `06_bash_files/`:
```bash
sbatch 06_bash_files/timeg_htc.sh
```
SLURM scripts use `--array=0-14` for the 15 subjects, with `--partition=htc`, 20 CPUs, 90GB memory. They run scripts as modules, e.g.:
```bash
python -m 04_source.time_gen.timeg_htc
```

**Quarto reports:**
```bash
# Render from RStudio, or:
quarto render 05_gam/rsa_tables.qmd
quarto render 05_gam/timeg_tables.qmd
```

## Architecture

### Data Flow

```
Raw MEG → 02_preprocessing/save_epochs.py → Preprocessed Epochs (.fif)
                                              ├── Sensor-space (03_sensors/)
                                              │     ├── RSA (rsa_rdms.py → rsa_figure.py)
                                              │     └── Time generalization (timeg_*.py)
                                              └── Source-space (04_source/)
                                                    ├── source_recon.py → Forward models
                                                    ├── RSA (rsa/rsa_blocks_*.py)
                                                    ├── Time generalization (time_gen/timeg_*.py)
                                                    └── Decoding (time_gen/decode_*.py)

Results → 05_gam/get_tables_*.py → Statistical tables
        → 05_gam/*.qmd → GAM analysis & formatted reports
        → 01_behavior/ → Behavioral RT analysis
```

### Key Modules

- **`config.py`** - All paths, subject lists, epoch definitions, network labels, color palettes. Paths auto-switch via environment variables: `MB_ENV` (MacBook + external drive), `CLUSTER_ENV` (HPC), or default (local desktop).
- **`base.py`** - Shared utility library (~870 lines). Core functions:
  - `decod_stats()` / `gat_stats()` / `gat_t1samp()` - Cluster permutation statistics for decoding and temporal generalization
  - `cv_mahalanobis_parallel()` / `train_test_mahalanobis_fast()` - Cross-validated Mahalanobis distance for RDM computation
  - `get_sequence()` / `get_rdm()` / `get_in_out_seq()` - Behavioral sequence extraction and RDM pair classification
  - `get_volume_estimate_time_course()` / `get_labels_from_vol_src()` - Source-space time course extraction
  - `do_pca()` - PCA dimensionality reduction on MEG epochs
  - `fisher_z_and_ttest()` - Fisher Z-transform and t-test for RDM comparisons

### Directory Layout

| Directory | Purpose |
|---|---|
| `01_behavior/` | Behavioral analysis: RT ANOVAs, plots, sequence extraction |
| `02_preprocessing/` | MEG preprocessing: ICA, autoreject, epoch extraction |
| `03_sensors/` | Sensor-space RSA and temporal generalization |
| `04_source/` | Source reconstruction, source-space RSA, time generalization, decoding |
| `05_gam/` | GAM modeling, statistical tables, Quarto reports |
| `06_bash_files/` | SLURM job submission scripts for HPC |

### Naming Conventions

- **`htc`** = hippocampus, thalamus, and cerebellum (subcortical regions)
- **`net`** = network-based (analysis per brain network from `config.NETWORKS`)
- **`lobo`** = leave-one-block-out cross-validation
- **`_supp`** suffix = supplementary analysis variants
- **`_wblock`** suffix = with block-level information

### Brain Networks Analyzed

Defined in `config.NETWORKS`: Visual, Sensorimotor, Dorsal Attention, Salience/Ventral Attention, Limbic, Central Executive (Control), Default Mode, Hippocampus, Thalamus, Cerebellum.

## Important Conventions

- Scripts read subject index from `SLURM_ARRAY_TASK_ID` environment variable when running on the cluster; set it manually for local runs.
- Statistical testing uses MNE's `permutation_cluster_1samp_test` with 2^12 permutations.
- Cross-validation: `StratifiedKFold` or block-wise leave-one-out schemes.
- Decoding pipeline: `StandardScaler` + `LogisticRegressionCV` via MNE's `SlidingEstimator`.
- Source reconstruction uses dSPM with FreeSurfer anatomical data.
- No test suite exists; this is research code validated through analysis and peer review.
