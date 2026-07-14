# Discrete State-Learning in Multivariate Time Series for CPS and HAR

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](../../actions/workflows/tests.yml/badge.svg)](../../actions/workflows/tests.yml)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)
![Datasets](https://img.shields.io/badge/Datasets-5%20%2B%20custom-2E8B57)
![Status](https://img.shields.io/badge/Status-Research%20Code-8A2BE2)

This repository contains the experiment code used for an ETFA paper. It compares five approaches for learning
discrete states from multivariate time series:

- vector-quantised VAE (VQ-VAE)
- categorical VAE (CatVAE)
- self-organising-map VAE (SOM-VAE)
- hidden Markov model (HMM)
- k-means

### Post-paper extension: RBM

A Gaussian–Bernoulli restricted Boltzmann machine (RBM) is available as an additional baseline. **The RBM was added
after the paper and is not part of the experiments, comparisons, or claims reported in the ETFA publication.** It is
kept separate in `metrics/<dataset>/rbm/`, and every generated RBM metric file contains `paper_model: false`.

The Gaussian visible units support the standardised continuous sensor values produced by the shared preprocessing
pipeline. The most active hidden unit defines the discrete state, so `--states K` creates `K` hidden units and at most
`K` discrete states.

```bash
python main.py --model rbm --dataset har --mode train --states 4 --seeds 42
python main.py --model rbm --dataset har --mode evaluate --states 4 --seeds 42
```

The same command-line interface runs training, evaluation, TEP anomaly-detection experiments, and hyperparameter
searches across five datasets.

## Quick start

Python 3.12 is required. CUDA is optional; PyTorch uses it automatically when available.

```bash
git clone <repository-url>
cd ETFA-Paper
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download and prepare at least one dataset as described below. A minimal run then looks like:

```bash
python main.py --model kmeans --dataset har --mode train --states 4 --seeds 42
python main.py --model kmeans --dataset har --mode evaluate --states 4 --seeds 42
```

Run `python main.py --help` for all options. Defaults intentionally execute only one state-space size and one seed,
so an accidental command does not launch the full paper experiment.

## Data preparation

Datasets are not committed to this repository. This keeps the clone small and respects the providers' distribution
terms. Download and extract a dataset, then point the preparation command at the extracted directory:

```bash
python prepare_data.py --dataset har --raw-dir ~/Downloads/UCI-HAR
python prepare_data.py --dataset pamap2 --raw-dir ~/Downloads/PAMAP2_Dataset
python prepare_data.py --dataset mhealth --raw-dir ~/Downloads/MHEALTHDATASET
python prepare_data.py --dataset tep --raw-dir ~/Downloads/TEP
python prepare_data.py --dataset wadi --raw-dir ~/Downloads/WADI
```

The script converts the source files where necessary and writes the NumPy windows, scaler, and metadata below
`data/<dataset>/processed/`.

| CLI name | Source | Raw files used |
| --- | --- | --- |
| `har` | [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones) | `X_train.txt`, `X_test.txt`, labels, subjects, and `features.txt` |
| `pamap2` | [UCI PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2%2Bphysical%2Bactivity%2Bmonitoring) | `subject*.dat` |
| `mhealth` | [UCI MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth%2Bdataset) | `mHealth_subject*.log` |
| `tep` | Tennessee Eastman Process data used in the paper | four `TEP_*.RData` files |
| `wadi` | [SUTD iTrust WADI request](https://www.sutd.edu.sg/itrust/request-for-datasets/) | `WADI_normal.csv` |

WADI requires a dataset request and acceptance of SUTD's terms. The TEP preparation currently targets the four
RData files named in `tep.py`; document the exact archive/DOI used by the paper before publishing the repository.

### Using another dataset

Custom CSV and Parquet datasets use the same leakage-safe preprocessing pipeline as the built-in loaders:

1. sort observations within each run (when `time_column` is configured),
2. handle missing values,
3. split complete runs into train, validation, and test,
4. fit the standard or min-max scaler on training observations only,
5. transform every split with that scaler, and
6. create sliding windows without crossing run or split boundaries.

Copy [the example specification](configs/datasets/example.json), adjust the column names, and prepare the data:

```bash
cp configs/datasets/example.json configs/datasets/my_dataset.json
# Edit my_dataset.json
python prepare_data.py \
  --spec configs/datasets/my_dataset.json \
  --raw-dir /path/to/extracted/files
```

The important specification fields are:

| Field | Meaning |
| --- | --- |
| `name` | Dataset identifier used by `main.py` and in output paths |
| `file_pattern` | Glob relative to `--raw-dir`, such as `*.csv` or `**/*.parquet` |
| `format` | `csv` or `parquet` |
| `label_column` | Discrete ground-truth state/activity; string labels are encoded automatically |
| `group_column` | Subject, machine, experiment, or run identifier |
| `time_column` | Optional column used to sort observations within a group |
| `feature_columns` | Optional explicit list; otherwise all eligible numeric columns are used |
| `exclude_columns` | Numeric metadata columns that must not become model inputs |
| `split_strategy` | `group` (recommended) or `chronological` for one continuous series |
| `scaler` | `standard`, `minmax`, or `none` |
| `fillna` | `ffill`, `interpolate`, `zero`, or `drop` |
| `window_length`, `stride` | Sliding-window parameters |

With `split_strategy: "group"`, at least three groups are required. If `group_column` is omitted, every input file is
treated as one run. Use `chronological` for a single continuous recording; each resulting segment receives a hard
window boundary.

The prepared dataset can then be used with every model. Generic baseline model configurations are selected
automatically:

```bash
python main.py --model kmeans --dataset my_dataset --mode train --states 4 --seeds 42
python main.py --model kmeans --dataset my_dataset --mode evaluate --states 4 --seeds 42
```

By default, custom data is read from `data/custom/<name>/processed/`. To keep it elsewhere, pass the dataset root
(the directory containing `processed/`):

```bash
python main.py --model vqvae --dataset my_dataset --data-dir /data/my_dataset \
  --mode train --states 4 --seeds 42
```

The files must contain a discrete label because the paper metrics compare learned states with ground-truth states.
Unlabelled data or continuous regression targets require a separate evaluation protocol.

## Running experiments

The general command is:

```bash
python main.py \
  --model {vqvae,catvae,somvae,hmm,kmeans,rbm} \
  --dataset {har,mhealth,pamap2,tep,wadi} \
  --mode {train,evaluate,tune,anomaly} \
  --states 4 \
  --seeds 42
```

Multiple values run a small grid in one invocation:

```bash
python main.py --model vqvae --dataset har --mode train \
  --states 4 6 9 12 16 20 25 \
  --seeds 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40
```

SOM-VAE converts each requested state count into the most balanced integer grid (for example, `6` becomes `2 x 3`).
Configuration defaults live in `configs/`. Use a custom configuration without editing tracked files:

```bash
python main.py --model catvae --dataset har --mode train \
  --config path/to/config.json --states 6 --seeds 42
```

Optuna tuning is available for VQ-VAE, CatVAE, SOM-VAE, HMM, and the post-paper RBM extension:

```bash
python main.py --model vqvae --dataset har --mode tune \
  --states 4 --seeds 40 --trials 30
```

Anomaly mode currently supports TEP only. Its normal reference splits are generated together with the standard TEP
splits by `prepare_data.py`.

## Outputs

- `checkpoints/<model>/<dataset>/<states>/`: trained models
- `anomaly_checkpoints/`: anomaly-detection models
- `tuning_results/`: Optuna studies and best configurations
- `metrics/`: per-run and aggregated paper metrics
- `paper_results/` and `statistics_results/`: generated tables and statistical analyses

Generated checkpoints, tuning results, and datasets are ignored by Git. The existing compact metric JSON files can be
kept as published experiment results.

## Reproducing tables and figures

After evaluation runs have produced metric JSON files:

```bash
python get_all_metrics.py
python paper_summary.py
python compute_statistics.py
python figures.py
```

These scripts read from `metrics/` and write derived summaries/figures to their configured output directories.

## Development

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
python -m compileall -q -x 'data|.git|.venv' .
ruff check .
pytest
```

## Citation


## License

This project is licensed under the [MIT License](LICENSE).
