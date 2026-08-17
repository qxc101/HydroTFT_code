# HydroTFT: A Cross-Basin Attention Model for Multi-Horizon Rainfall–Runoff Prediction

This repository contains the code used to produce every result in the manuscript
*HydroTFT: A Cross-Basin Attention Model for Multi-Horizon Rainfall–Runoff Prediction*
: the HydroTFT model, the four baselines
(EA-LSTM, LSTM, PatchTST, iTransformer), and the training / evaluation pipeline for the
1-, 7- and 14-day tasks on the 531 CAMELS basins.

The code is a modification of the CAMELS benchmark code base of Kratzert et al. (2019),
<https://github.com/kratzert/ealstm_regional_modeling> (Apache-2.0). Only the files we
added or changed are distributed here; the unchanged upstream files (the EA-LSTM and
LSTM model definitions, metrics, etc.) are obtained from the upstream repository as
described in *Setup*. Data (CAMELS) and trained weights are not included because of their
size; the data are public and instructions for obtaining them are below.

---

## 1. Contents

```
HydroTFT_code/
├── main.py                       entry point: train / evaluate           [modified upstream file]
├── papercode/
│   ├── tft.py                    HydroTFT                                           [new]
│   ├── patchtst.py               PatchTST baseline                                  [new]
│   ├── itransformer.py           iTransformer baseline                              [new]
│   ├── datasets.py               CAMELS dataset classes                  [modified upstream file]
│   ├── datautils.py              I/O, normalisation, engineered features [modified upstream file]
│   ├── nseloss.py                basin-normalised NSE loss               [modified upstream file]
│   └── utils.py                  HDF5 caching                            [modified upstream file]
├── data/
│   ├── basin_list.txt            the 531 CAMELS basins used throughout the paper
│   └── basin_list_quick50.txt    the 50-basin development subset (ablation, sweep, interpretability)
├── LICENSE                       Apache-2.0 (inherited from upstream)
└── README.md
```

What each modified upstream file changes:

| File | Modification |
|---|---|
| `main.py` | adds `--model_type {tft, itransformer, patchtst}`; multi-horizon prediction (`--pred_days`); engineered features (`--use_starter_features`); ablation switches (`--no_attention`, `--no_feature_selection`, `--no_static`); transformer hyperparameters (`--d_model`, `--n_heads`, `--e_layers`, `--d_ff`, `--patch_head`, `--patch_channels`); `--hidden_size`, `--batch_size`, `--weight_decay`, `--basin_file`; best-of-last-*N* checkpoint selection at evaluation (`--eval_last_n`, `--eval_epoch`); per-lead-time evaluation output. |
| `papercode/datasets.py` | multi-day targets (a `pred_days`-vector per sample) and the five engineered dynamic features. |
| `papercode/datautils.py` | computation of the five engineered features (`doy_sin`, `doy_cos`, `prcp_sum_90`, `degday_7`, `wetdays_7`); `N_STARTER_FEATURES`. |
| `papercode/nseloss.py` | basin-normalised NSE loss over a multi-day output; optional linear lead-time weighting (`--horizon_alpha`, 0 in all reported runs). |
| `papercode/utils.py` | HDF5 caching generalised to a variable number of input channels and output length. |

---

## 2. Setup

### 2.1 Code

The upstream repository supplies the unchanged files (`papercode/ealstm.py`, `papercode/lstm.py`,
`papercode/metrics.py`, `papercode/__init__.py`, ...). Clone it, then copy this repository over it:

```bash
git clone https://github.com/kratzert/ealstm_regional_modeling.git
cd ealstm_regional_modeling
git checkout d118158            # the upstream commit we started from

# copy HydroTFT_code on top: adds the new files and REPLACES main.py and the four
# modified papercode/ files
cp -r /path/to/HydroTFT_code/main.py        ./main.py
cp -r /path/to/HydroTFT_code/papercode/*.py ./papercode/
cp -r /path/to/HydroTFT_code/data/*.txt     ./data/
```

After this step `papercode/` contains both the untouched upstream models (EA-LSTM, LSTM) and
ours (HydroTFT, PatchTST, iTransformer), and `main.py` can train any of them.

### 2.2 Environment

The upstream `environment_gpu.yml` is out of date; the runs in the paper used

```
python 3.10 · pytorch 2.7.0 (CUDA 12.8) · numpy 2.0 · pandas 2.2 · scipy 1.15
h5py 3.14 · numba 0.61 · tqdm 4.67
```

```bash
conda create -n hydrotft python=3.10 -y && conda activate hydrotft
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==2.0.2 pandas==2.2.3 scipy==1.15.3 h5py==3.14.0 numba==0.61.2 tqdm==4.67.1
```

Hardware: a single NVIDIA RTX 5090 (32 GB). One epoch on 531 basins takes about 3 min for
HydroTFT / PatchTST and about 1 min for the LSTMs.

### 2.3 Data

Download the CAMELS-US dataset (Newman et al., 2015; Addor et al., 2017) from
<https://ral.ucar.edu/solutions/products/camels> and arrange it exactly as the upstream
README describes:

```
CAMELS_US/
├── basin_mean_forcing/maurer_extended/   Maurer forcings (used in the paper)
├── usgs_streamflow/
└── camels_attributes_v2.0/               static attributes
```

The catchment attributes are loaded from `camels_attributes_v2.0` into a SQLite file the first
time `main.py train` runs (upstream behaviour). Pass the folder as `--camels_root`.

---

## 3. Reproducing the paper

Every command below is the exact configuration of a run reported in the manuscript
(the multi-seed results use seeds 456, 111 and 222 with all other
flags identical). Training writes to `runs/run_<DDMM>_<HHMM>_seed<seed>/`; evaluation writes
per-lead-time, per-basin predictions and a `summary_stats.json` under
`runs/<run>/eval_results/`.

Common flags: `CAMELS=/path/to/CAMELS_US`.

### 3.1 HydroTFT (Tables 3, 5, 6)

```bash
# 1-day (nowcast variant: 5 forcings + 27 static attributes, 270-day window)
python main.py train --camels_root $CAMELS --model_type tft --use_starter_features --pred_days 0 --seq_length 270 \
    --learning_rate 1e-3 --dropout 0.4 --epochs 30 --batch_size 1024 --seed 456 --cache_data True

# 7-day  (5 forcings + 5 engineered features + 27 static attributes, 365-day window)
python main.py train --camels_root $CAMELS --model_type tft --use_starter_features --pred_days 7 \
    --seq_length 365 --learning_rate 5e-4 --dropout 0.3 --epochs 40 --batch_size 512 \
    --weight_decay 1e-5 --seed 456 --cache_data True

# 14-day
python main.py train --camels_root $CAMELS --model_type tft --use_starter_features --pred_days 14 \
    --seq_length 365 --learning_rate 5e-4 --dropout 0.3 --epochs 50 --batch_size 512 \
    --weight_decay 1e-5 --seed 456 --cache_data True
```


### 3.2 Baselines (Tables 3, 5, 6)

All baselines receive the same inputs as each other .
The commands below are the 7-day runs. For the other horizons: the LSTMs use `--pred_days 0`
(1-day nowcast) or `--pred_days 14` with `--epochs 30` at every horizon; the transformers use
`--pred_days 1 --epochs 30` (1-day) and `--pred_days 14 --epochs 50` (14-day).

```bash
# EA-LSTM
python main.py train --camels_root $CAMELS --model_type ealstm --pred_days 7 --seq_length 270 \
    --learning_rate 1e-3 --dropout 0.4 --epochs 30 --seed 456 --cache_data True

# LSTM (static attributes concatenated to the dynamic input)
python main.py train --camels_root $CAMELS --model_type lstm --concat_static True --pred_days 7 \
    --seq_length 270 --learning_rate 1e-3 --dropout 0.4 --epochs 30 --seed 456 --cache_data True

# iTransformer  (paper defaults: d_model 512, 8 heads, 3 layers, d_ff 512, dropout 0.1)
python main.py train --camels_root $CAMELS --model_type itransformer --pred_days 7 --seq_length 365 \
    --learning_rate 5e-4 --dropout 0.1 --epochs 40 --weight_decay 1e-5 --seed 456 --cache_data True

# PatchTST      (paper defaults: patch 16 / stride 8, d_model 128, 16 heads, 3 layers, d_ff 256, dropout 0.2)
python main.py train --camels_root $CAMELS --model_type patchtst --pred_days 7 --seq_length 365 \
    --learning_rate 5e-4 --dropout 0.2 --epochs 40 --weight_decay 1e-5 --seed 456 --cache_data True
```

To run the transformers with the HydroTFT-matched configuration (reported in the response
letter) add `--d_model 256 --n_heads 4 --e_layers 3 --d_ff 256 --dropout 0.3`.

PatchTST fairness variants: `--no_static True` removes the static-attribute
encoder; `--patch_head channel` uses PatchTST's own per-channel head (shared across channels
and averaged); `--patch_channels 0` restricts the input to precipitation only.

### 3.3 Evaluation

For every model, including all baselines, the reported checkpoint is the one with the highest
mean basin NSE among the final ten training epochs:

```bash
python main.py evaluate --camels_root $CAMELS --run_dir runs/<run> --eval_last_n 10
```

This evaluates the last ten checkpoints on the test period (water years 1990–1999), keeps
the best, and writes `eval_results/<name>/step_<k>/basin_<id>.npz` (per-basin observed and
predicted streamflow at lead time *k*) plus `summary_stats.json` (mean and median basin NSE
per lead time). All metrics, significance tests and figures in the paper are computed from
these per-basin files.

For the 14-day model the reported checkpoint is the best of epochs 21–30 (the model is trained
for 30 epochs, see 3.1); `--eval_last_n 10` on that run applies exactly this rule.

### 3.4 Ablation and architecture sweep (Table 7, Section 3.3)

Run on the 50-basin development subset at the 7-day horizon, all with
`--basin_file data/basin_list_quick50.txt --epochs 30` and evaluated with
`--eval_last_n 10 --basin_file data/basin_list_quick50.txt`:

```bash
BASE="--camels_root $CAMELS --model_type tft --pred_days 7 --seq_length 365 --learning_rate 5e-4 \
      --dropout 0.3 --epochs 30 --weight_decay 1e-5 --seed 456 --cache_data True \
      --basin_file data/basin_list_quick50.txt"

python main.py train $BASE --use_starter_features                          # full model
python main.py train $BASE                                                  # - engineered features
python main.py train $BASE --use_starter_features --no_attention True       # - attention
python main.py train $BASE --use_starter_features --no_feature_selection True   # - variable selection
python main.py train $BASE --use_starter_features --no_static True          # - static attributes

# one-factor sweep around the full model
python main.py train $BASE --use_starter_features --seq_length 180         # (and 270, 540)
python main.py train $BASE --use_starter_features --hidden_size 64         # (and 128)
python main.py train $BASE --use_starter_features --n_heads 1              # (and 2, 8)
```

The 1-day and 14-day ablation rows use `--pred_days 1` and `--pred_days 14` with the same
flags.

### 3.5 Flags not used in the paper

`main.py` also accepts `--pretrained_run_dir`, `--encoder_lr_scale`, `--use_mse`,
`--horizon_alpha` and the upstream `eval_robustness` mode. None of these was active in any
reported run (`--horizon_alpha 0`, `--use_mse False`, no pretraining); they are left in place
so that the file matches the one that produced the released checkpoints exactly.

### 3.6 Interpretability (Section 3.3, Figures 4–5)

The permutation feature importance and the attention profiles are computed at inference time
from a trained 7-day checkpoint on the 50-basin subset. Load the model with the `Model` class in
`main.py` (`model_type='tft'`, `pred_days=7`, `input_size_dyn=10`, `input_size_stat=27`,
`hidden_size=256`, `dropout=0.3`, `seq_length=365`); the forward pass returns
`(prediction, hidden, attention)` where `attention` has shape `(batch, 4 heads, 365, 365)`
(softmax over keys). Feature importance permutes one input channel across samples and
recomputes the mean basin NSE; the attention profile averages `attention` over heads, queries
and samples.

---

## 4. Licence and attribution

This code is distributed under the Apache License 2.0 (`LICENSE`), the licence of the upstream
repository from which it is derived. The citation will soon be avaliable.
