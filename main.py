"""HydroTFT -- training and evaluation entry point.

Modified from main.py of https://github.com/kratzert/ealstm_regional_modeling (Apache-2.0):
adds the HydroTFT (TFT / VanillaTFT), iTransformer and PatchTST models, multi-horizon
prediction (--pred_days), engineered input features (--use_starter_features), the ablation
switches, and best-of-last-N checkpoint selection at evaluation. Distributed under the Apache
License 2.0; see the LICENSE file.
"""
import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from papercode.datasets import CamelsH5, CamelsTXT
from papercode.datautils import (add_camels_attributes, load_attributes,
                                 rescale_features)
from papercode.ealstm import EALSTM
from papercode.lstm import LSTM
from papercode.tft import TFT, VanillaTFT
from papercode.itransformer import iTransformer
from papercode.patchtst import PatchTST
from papercode.metrics import calc_nse
from papercode.nseloss import NSELoss
from papercode.utils import create_h5_files, get_basin_list

###########
# Globals #
###########

# fixed settings for all experiments
def _parse_channels(v):
    """'0,2' -> [0, 2]; None/'' -> None (all channels)."""
    if v is None or v == '' or v == 'None':
        return None
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    return [int(x) for x in str(v).split(',') if x.strip() != '']


GLOBAL_SETTINGS = {
    'batch_size': 512,
    'clip_norm': True,
    'clip_value': 1,
    'dropout': 0.4,
    'epochs': 30,
    'hidden_size': 256,
    'initial_forget_gate_bias': 5,
    'log_interval': 50,
    'learning_rate': 1e-3,
    'seq_length': 270,
    'pred_days': 0,
    'train_start': pd.to_datetime('01101999', format='%d%m%Y'),
    'train_end': pd.to_datetime('30092008', format='%d%m%Y'),
    'val_start': pd.to_datetime('01101989', format='%d%m%Y'),
    'val_end': pd.to_datetime('30091999', format='%d%m%Y')
}

# check if GPU is available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###############
# Prepare run #
###############


def get_args() -> Dict:
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate", "eval_robustness"])
    parser.add_argument('--camels_root', type=str, help="Root directory of CAMELS data set")
    parser.add_argument('--seed', type=int, required=False, help="Random seed")
    parser.add_argument('--run_dir', type=str, help="For evaluation mode. Path to run directory.")
    parser.add_argument('--cache_data',
                        type=bool,
                        default=False,
                        help="If True, loads all data into memory")
    parser.add_argument('--num_workers',
                        type=int,
                        default=12,
                        help="Number of parallel threads for data loading")
    parser.add_argument('--no_static',
                        type=bool,
                        default=False,
                        help="If True, trains LSTM without static features")
    parser.add_argument('--concat_static',
                        type=bool,
                        default=False,
                        help="If True, train LSTM with static feats concatenated at each time step")
    parser.add_argument('--use_mse',
                        type=bool,
                        default=False,
                        help="If True, uses mean squared error as loss function.")
    parser.add_argument('--model_type',
                        type=str,
                        default='lstm',
                        choices=['lstm', 'ealstm', 'tft', 'itransformer', 'patchtst'],
                        help="Model type to use. Options: 'lstm', 'ealstm', 'tft', 'itransformer', 'patchtst'")
    parser.add_argument('--use_starter_features',
                        action='store_true',
                        help='If set, adds engineered starter dynamic features (DoY sin/cos, precip/temperature aggregates, etc.)')
    parser.add_argument('--pred_days',
                        type=int,
                        default=None,
                        help='Prediction horizon: 0=nowcast (day 270), 1+=forecast (next N days)')
    parser.add_argument('--seq_length',
                        type=int,
                        default=None,
                        help='Input sequence length in days. Overrides default of 270.')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=None,
                        help='Initial learning rate. Overrides default of 1e-3.')
    parser.add_argument('--dropout',
                        type=float,
                        default=None,
                        help='Dropout probability. Overrides default of 0.4.')
    parser.add_argument('--eval_last_n',
                        type=int,
                        default=10,
                        help='During evaluation, test last N epoch checkpoints and pick best by mean basin NSE.')
    parser.add_argument('--eval_epoch',
                        type=int,
                        default=None,
                        help='End epoch for evaluation window. E.g. --eval_epoch 20 --eval_last_n 1 evaluates only epoch 20.')
    parser.add_argument('--epochs',
                        type=int,
                        default=None,
                        help='Number of training epochs. Overrides default of 30.')
    parser.add_argument('--horizon_alpha',
                        type=float,
                        default=0.0,
                        help='Horizon weighting alpha for NSELoss. 0=equal weighting, >0=upweight later days.')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0,
                        help='Weight decay (L2 regularization) for Adam optimizer.')
    parser.add_argument('--basin_file',
                        type=str,
                        default=None,
                        help='Path to custom basin list file (one ID per line). Overrides default 531-basin list.')
    parser.add_argument('--pretrained_run_dir',
                        type=str,
                        default=None,
                        help='Path to a previous run dir. Loads encoder weights, skips output head.')
    parser.add_argument('--encoder_lr_scale',
                        type=float,
                        default=0.1,
                        help='LR multiplier for encoder params when using --pretrained_run_dir (default: 0.1)')
    parser.add_argument('--no_attention',
                        action='store_true',
                        help='Ablation: disable self-attention in TFT')
    parser.add_argument('--no_feature_selection',
                        action='store_true',
                        help='Ablation: disable temporal variable selection in TFT')
    parser.add_argument('--d_model', type=int, default=None,
                        help='Transformer model dim for itransformer/patchtst. Default: each paper value.')
    parser.add_argument('--n_heads', type=int, default=None,
                        help='Transformer attention heads for itransformer/patchtst. Default: each paper value.')
    parser.add_argument('--e_layers', type=int, default=None,
                        help='Transformer encoder layers for itransformer/patchtst. Default: each paper value.')
    parser.add_argument('--d_ff', type=int, default=None,
                        help='Transformer feed-forward dim for itransformer/patchtst. Default: each paper value.')
    parser.add_argument('--patch_head', type=str, default='joint', choices=['joint', 'channel'],
                        help="PatchTST head: 'joint' flatten over all channels (paper setup) or 'channel' "
                             "= PatchTST's own per-channel head, shared and averaged over channels.")
    parser.add_argument('--patch_channels', type=str, default=None,
                        help='PatchTST only: comma-separated dynamic channel indices to keep '
                             '(0=prcp,1=srad,2=tmax,3=tmin,4=vp). E.g. "0" = univariate precipitation.')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Training batch size. Overrides the GLOBAL_SETTINGS default of 512.')
    parser.add_argument('--hidden_size', type=int, default=None,
                        help='Hidden size (d_model for TFT). Overrides the GLOBAL_SETTINGS default of 256.')
    cfg = vars(parser.parse_args())

    # Validation checks
    if (cfg["mode"] == "train") and (cfg["seed"] is None):
        # generate random seed for this run
        cfg["seed"] = int(np.random.uniform(low=0, high=1e6))

    if (cfg["mode"] in ["evaluate", "eval_robustness"]) and (cfg["run_dir"] is None):
        raise ValueError("In evaluation mode a run directory (--run_dir) has to be specified")

    # Save user-specified overrides before GLOBAL_SETTINGS overwrite
    user_pred_days = cfg["pred_days"]
    user_seq_length = cfg["seq_length"]
    user_learning_rate = cfg["learning_rate"]
    user_dropout = cfg["dropout"]
    user_epochs = cfg["epochs"]
    user_basin_file = cfg["basin_file"]
    user_pretrained_run_dir = cfg["pretrained_run_dir"]
    user_encoder_lr_scale = cfg["encoder_lr_scale"]
    user_batch_size = cfg["batch_size"]
    user_hidden_size = cfg["hidden_size"]

    # combine global settings with user config
    cfg.update(GLOBAL_SETTINGS)

    # Restore user-specified overrides (if provided)
    if user_pred_days is not None:
        cfg["pred_days"] = user_pred_days
    if user_seq_length is not None:
        cfg["seq_length"] = user_seq_length
    if user_learning_rate is not None:
        cfg["learning_rate"] = user_learning_rate
    if user_dropout is not None:
        cfg["dropout"] = user_dropout
    if user_epochs is not None:
        cfg["epochs"] = user_epochs
    if user_batch_size is not None:
        cfg["batch_size"] = user_batch_size
    if user_hidden_size is not None:
        cfg["hidden_size"] = user_hidden_size
    cfg["basin_file"] = user_basin_file
    cfg["pretrained_run_dir"] = user_pretrained_run_dir
    cfg["encoder_lr_scale"] = user_encoder_lr_scale

    if cfg["mode"] == "train":
        # print config to terminal
        for key, val in cfg.items():
            print(f"{key}: {val}")

    # convert path to PosixPath object
    cfg["camels_root"] = Path(cfg["camels_root"])
    if cfg["run_dir"] is not None:
        cfg["run_dir"] = Path(cfg["run_dir"])
    return cfg


def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    now = datetime.now()
    day = f"{now.day}".zfill(2)
    month = f"{now.month}".zfill(2)
    hour = f"{now.hour}".zfill(2)
    minute = f"{now.minute}".zfill(2)
    run_name = f'run_{day}{month}_{hour}{minute}_seed{cfg["seed"]}'
    cfg['run_dir'] = Path(__file__).absolute().parent / "runs" / run_name
    if not cfg["run_dir"].is_dir():
        cfg["train_dir"] = cfg["run_dir"] / 'data' / 'train'
        cfg["train_dir"].mkdir(parents=True)
        cfg["val_dir"] = cfg["run_dir"] / 'data' / 'val'
        cfg["val_dir"].mkdir(parents=True)
    else:
        raise RuntimeError(f"There is already a folder at {cfg['run_dir']}")

    # dump a copy of cfg to run directory
    with (cfg["run_dir"] / 'cfg.json').open('w') as fp:
        temp_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, PosixPath):
                temp_cfg[key] = str(val)
            elif isinstance(val, pd.Timestamp):
                temp_cfg[key] = val.strftime(format="%d%m%Y")
            else:
                temp_cfg[key] = val
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)

    return cfg


def _prepare_data(cfg: Dict, basins: List) -> Dict:
    """Preprocess training data.

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config
    basins : List
        List containing the 8-digit USGS gauge id

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    # create database file containing the static basin attributes
    cfg["db_path"] = str(cfg["run_dir"] / "attributes.db")
    add_camels_attributes(cfg["camels_root"], db_path=cfg["db_path"])

    # create .h5 files for train and validation data
    cfg["train_file"] = cfg["train_dir"] / 'train_data.h5'
    create_h5_files(camels_root=cfg["camels_root"],
                    out_file=cfg["train_file"],
                    basins=basins,
                    dates=[cfg["train_start"], cfg["train_end"]],
                    with_basin_str=True,
                    seq_length=cfg["seq_length"],
                    use_starter_features=cfg.get("use_starter_features", False),
                    pred_days=cfg.get("pred_days", 0))

    return cfg


################
# Define Model #
################


class Model(nn.Module):
    """Wrapper class that connects LSTM/EA-LSTM/TFT with fully connected layer"""

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 initial_forget_bias: int = 5,
                 dropout: float = 0.0,
                 concat_static: bool = False,
                 no_static: bool = False,
                 model_type: str = 'lstm',
                 pred_days: int = 0,
                 no_attention: bool = False,
                 no_feature_selection: bool = False,
                 seq_length: int = 365,
                 d_model: int = None,
                 n_heads: int = None,
                 e_layers: int = None,
                 d_ff: int = None,
                 patch_head: str = 'joint',
                 patch_channels=None):
        """Initialize model.

        Parameters
        ----------
        input_size_dyn: int
            Number of dynamic input features.
        input_size_stat: int
            Number of static input features (used in the EA-LSTM input gate).
        hidden_size: int
            Number of LSTM cells/hidden units.
        initial_forget_bias: int
            Value of the initial forget gate bias. (default: 5)
        dropout: float
            Dropout probability in range(0,1). (default: 0.0)
        concat_static: bool
            If True, uses standard LSTM otherwise uses EA-LSTM
        no_static: bool
            If True, runs standard LSTM
        model_type: str
            Type of model to use ('lstm', 'ealstm', 'tft')
        pred_days: int
            0=nowcast (output size 1), >0=forecast (output size pred_days)
        no_attention: bool
            Ablation: disable self-attention in TFT
        no_feature_selection: bool
            Ablation: disable temporal VSN in TFT
        """
        super(Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout
        self.concat_static = concat_static
        self.no_static = no_static
        self.model_type = model_type
        self.out_size = max(pred_days, 1)

        print(f"-> Using model type: {self.model_type}, out_size: {self.out_size}")
        if model_type == 'tft':
                # Full TFT (v3f): VSN, GateAddNorm, static enrichment for forecasting
                self.model = TFT(
                    input_size_dyn=input_size_dyn,
                    input_size_stat=input_size_stat,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    concat_static=concat_static,
                    no_static=no_static,
                    initial_forget_bias=initial_forget_bias,
                    pred_days=pred_days,
                    no_attention=no_attention,
                    no_feature_selection=no_feature_selection,
                    n_heads=(n_heads if n_heads is not None else 4))
            # Both TFT variants have built-in output projection
            self.fc = None
        elif model_type == 'itransformer':
            # iTransformer baseline. Consumes the same dynamic (x_d) + static (x_s)
            # inputs as TFT and outputs pred_days values. Architecture params default
            # to the paper values unless overridden (e.g. matched to HydroTFT).
            arch = {k: v for k, v in
                    dict(d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff).items()
                    if v is not None}
            self.model = iTransformer(
                input_size_dyn=input_size_dyn,
                input_size_stat=input_size_stat,
                pred_days=pred_days,
                seq_length=seq_length,
                dropout=dropout,
                **arch)
            self.fc = None
        elif model_type == 'patchtst':
            # PatchTST baseline. Architecture params default to paper values unless
            # overridden (e.g. matched to HydroTFT).
            arch = {k: v for k, v in
                    dict(d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff).items()
                    if v is not None}
            self.model = PatchTST(
                input_size_dyn=input_size_dyn,
                input_size_stat=input_size_stat,
                pred_days=pred_days,
                seq_length=seq_length,
                dropout=dropout,
                head_mode=patch_head,
                channels=patch_channels,
                **arch)
            self.fc = None
        elif self.concat_static or self.no_static or model_type == 'lstm':
            self.model = LSTM(input_size=input_size_dyn,
                             hidden_size=hidden_size,
                             initial_forget_bias=initial_forget_bias)
            self.fc = nn.Linear(hidden_size, self.out_size)
        else:  # EA-LSTM
            self.model = EALSTM(input_size_dyn=input_size_dyn,
                               input_size_stat=input_size_stat,
                               hidden_size=hidden_size,
                               initial_forget_bias=initial_forget_bias)
            self.fc = nn.Linear(hidden_size, self.out_size)

        if self.fc is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass through the model.

        Parameters
        ----------
        x_d : torch.Tensor
            Tensor containing the dynamic input features of shape [batch, seq_length, n_features]
        x_s : torch.Tensor, optional
            Tensor containing the static catchment characteristics, by default None

        Returns
        -------
        out : torch.Tensor
            Tensor containing the network predictions
        h_n : torch.Tensor
            Tensor containing the hidden states of each time step
        c_n : torch,Tensor
            Tensor containing the cell states of each time step (or attention weights for TFT)
        """
        if self.model_type in ('tft', 'itransformer', 'patchtst'):
            out, h_n, attention_weights = self.model(x_d, x_s)
            return out, h_n, attention_weights
        elif self.concat_static or self.no_static or self.model_type == 'lstm':
            h_n, c_n = self.model(x_d)
        else:  # EA-LSTM
            h_n, c_n = self.model(x_d, x_s)
            
        if self.fc is not None:
            last_h = self.dropout(h_n[:, -1, :])
            out = self.fc(last_h)
        else:
            out = h_n[:, -1, :]  # This shouldn't happen with current logic
            
        return out, h_n, c_n


###########################
# Train or evaluate model #
###########################


def train(cfg):
    """Train model.

    Parameters
    ----------
    cfg : Dict
        Dictionary containing the run config
    """
    # fix random seeds
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    if cfg.get("basin_file"):
        with open(cfg["basin_file"]) as f:
            basins = [line.strip() for line in f if line.strip()]
        print(f"Using custom basin list: {cfg['basin_file']} ({len(basins)} basins)")
    else:
        basins = get_basin_list()

    # create folder structure for this run
    cfg = _setup_run(cfg)

    # prepare data for training
    cfg = _prepare_data(cfg=cfg, basins=basins)

    # prepare PyTorch DataLoader
    ds = CamelsH5(h5_file=cfg["train_file"],
                  basins=basins,
                  db_path=cfg["db_path"],
                  concat_static=cfg["concat_static"],
                  cache=cfg["cache_data"],
                  no_static=cfg["no_static"])
    loader = DataLoader(ds,
                        batch_size=cfg["batch_size"],
                        shuffle=True,
                        num_workers=cfg["num_workers"])

    # create model and optimizer
    input_size_stat = 0 if cfg["no_static"] else 27
    # Determine input dynamic feature size: base 5 + extra if starter features enabled
    extra = 0
    if cfg.get("use_starter_features", False):
        from papercode.datautils import N_STARTER_FEATURES
        extra = N_STARTER_FEATURES
    input_size_dyn = (5 + extra) if (cfg["no_static"] or not cfg["concat_static"]) else (32 + extra)
    model = Model(input_size_dyn=input_size_dyn,
                  input_size_stat=input_size_stat,
                  hidden_size=cfg["hidden_size"],
                  initial_forget_bias=cfg["initial_forget_gate_bias"],
                  dropout=cfg["dropout"],
                  concat_static=cfg["concat_static"],
                  no_static=cfg["no_static"],
                  model_type=cfg["model_type"],
                  pred_days=cfg["pred_days"],
                  no_attention=cfg.get("no_attention", False),
                  no_feature_selection=cfg.get("no_feature_selection", False),
                  seq_length=cfg["seq_length"],
                  d_model=cfg.get("d_model"),
                  n_heads=cfg.get("n_heads"),
                  e_layers=cfg.get("e_layers"),
                  d_ff=cfg.get("d_ff"),
                  patch_head=cfg.get("patch_head", "joint"),
                  patch_channels=_parse_channels(cfg.get("patch_channels"))).to(DEVICE)
    # Load pretrained encoder weights (transfer learning)
    if cfg.get("pretrained_run_dir"):
        pretrained_dir = Path(cfg["pretrained_run_dir"])
        ckpts = list(pretrained_dir.glob("model_epoch*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {pretrained_dir}")
        ckpt_path = max(ckpts, key=lambda p: int(p.stem.replace("model_epoch", "")))
        print(f"Transfer learning: loading weights from {ckpt_path}")
        pretrained_state = torch.load(str(ckpt_path), map_location=DEVICE)
        model_state = model.state_dict()
        filtered = {k: v for k, v in pretrained_state.items()
                    if k in model_state and v.shape == model_state[k].shape}
        skipped = [k for k in pretrained_state if k not in filtered]
        print(f"  Loaded {len(filtered)}/{len(pretrained_state)} params, skipped: {skipped}")
        model.load_state_dict(filtered, strict=False)

    # Create optimizer (with differential LR for transfer learning)
    if cfg.get("pretrained_run_dir"):
        enc_scale = cfg.get("encoder_lr_scale", 0.1)
        encoder_params, head_params = [], []
        for name, param in model.named_parameters():
            if 'output_fc' in name:
                head_params.append(param)
            else:
                encoder_params.append(param)
        optimizer = torch.optim.Adam([
            {'params': encoder_params, 'lr': cfg["learning_rate"] * enc_scale, 'lr_scale': enc_scale},
            {'params': head_params, 'lr': cfg["learning_rate"], 'lr_scale': 1.0},
        ], weight_decay=cfg.get("weight_decay", 0.0))
        print(f"  Differential LR: encoder={cfg['learning_rate']*enc_scale:.1e}, head={cfg['learning_rate']:.1e}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                                      weight_decay=cfg.get("weight_decay", 0.0))

    # define loss function
    if cfg["use_mse"]:
        loss_func = nn.MSELoss()
    else:
        loss_func = NSELoss(horizon_alpha=cfg.get("horizon_alpha", 0.0))

    # reduce learning rates at 33% and 66% of total epochs (relative to initial LR)
    lr = cfg["learning_rate"]
    total_epochs = cfg["epochs"]
    learning_rates = {
        int(total_epochs * 0.33) + 1: lr * 0.5,
        int(total_epochs * 0.66) + 1: lr * 0.1,
    }

    for epoch in range(1, cfg["epochs"] + 1):
        # set new learning rate (respects per-group lr_scale for transfer learning)
        if epoch in learning_rates.keys():
            for param_group in optimizer.param_groups:
                scale = param_group.get('lr_scale', 1.0)
                param_group["lr"] = learning_rates[epoch] * scale

        train_epoch(model, optimizer, loss_func, loader, cfg, epoch, cfg["use_mse"])

        model_path = cfg["run_dir"] / f"model_epoch{epoch}.pt"
        torch.save(model.state_dict(), str(model_path))


def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, loss_func: nn.Module,
                loader: DataLoader, cfg: Dict, epoch: int, use_mse: bool):
    """Train model for a single epoch.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    optimizer : torch.optim.Optimizer
        Optimizer used for weight updating
    loss_func : nn.Module
        The loss function, implemented as a PyTorch Module
    loader : DataLoader
        PyTorch DataLoader containing the training data in batches.
    cfg : Dict
        Dictionary containing the run config
    epoch : int
        Current Number of epoch
    use_mse : bool
        If True, loss_func is nn.MSELoss(), else NSELoss() which expects addtional std of discharge
        vector

    """
    model.train()

    # process bar handle
    pbar = tqdm(loader, file=sys.stdout)
    pbar.set_description(f'# Epoch {epoch}')

    # Iterate in batches over training set
    for data in pbar:
        # delete old gradients
        optimizer.zero_grad()

        # forward pass through LSTM
        if len(data) == 3:
            x, y, q_stds = data
            x, y, q_stds = x.to(DEVICE), y.to(DEVICE), q_stds.to(DEVICE)
            predictions = model(x)[0]

        # forward pass through EALSTM
        elif len(data) == 4:
            x_d, x_s, y, q_stds = data
            x_d, x_s, y = x_d.to(DEVICE), x_s.to(DEVICE), y.to(DEVICE)
            predictions = model(x_d, x_s[:, 0, :])[0]

        # MSELoss
        if use_mse:
            loss = loss_func(predictions, y)

        # NSELoss needs std of each basin for each sample
        else:
            q_stds = q_stds.to(DEVICE)
            loss = loss_func(predictions, y, q_stds)

        # calculate gradients
        loss.backward()

        if cfg["clip_norm"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_value"])

        # perform parameter update
        optimizer.step()

        pbar.set_postfix_str(f"Loss: {loss.item():5f}")


def _evaluate_single_epoch(model: nn.Module, weight_file: Path, basins: List,
                           user_cfg: Dict, run_cfg: Dict, means: pd.Series,
                           stds: pd.Series, db_path: str,
                           pred_days: int) -> Tuple[Dict, float]:
    """Evaluate a single checkpoint across all basins.

    Returns
    -------
    results : Dict
        basin_id -> {'preds': np.ndarray, 'obs': np.ndarray, 'date_range': DatetimeIndex}
    mean_basin_nse : float
        Mean NSE across basins (step 0). Returns -inf if no valid basins.
    """
    model.load_state_dict(torch.load(weight_file, map_location=DEVICE))

    date_range = pd.date_range(start=GLOBAL_SETTINGS["val_start"], end=GLOBAL_SETTINGS["val_end"])
    results = {}
    for basin in tqdm(basins, desc=f"  {weight_file.name}"):
        ds_test = CamelsTXT(camels_root=user_cfg["camels_root"],
                            basin=basin,
                            dates=[GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]],
                            is_train=False,
                            seq_length=run_cfg["seq_length"],
                            with_attributes=True,
                            attribute_means=means,
                            attribute_stds=stds,
                            concat_static=run_cfg["concat_static"],
                            use_starter_features=run_cfg.get("use_starter_features", False),
                            pred_days=pred_days,
                            db_path=db_path)
        loader = DataLoader(ds_test, batch_size=1024, shuffle=False, num_workers=12)
        preds, obs = evaluate_basin(model, loader)
        results[basin] = {'preds': preds, 'obs': obs, 'date_range': date_range}

    # Compute mean basin NSE on step 0 for ranking
    basin_nses = []
    for basin, data in results.items():
        obs_step = data['obs'][:, 0]
        pred_step = data['preds'][:, 0]
        valid_mask = obs_step >= 0
        if valid_mask.sum() > 0:
            try:
                nse = calc_nse(obs_step[valid_mask], pred_step[valid_mask])
                basin_nses.append(nse)
            except RuntimeError:
                pass  # skip basins where all obs are equal

    mean_nse = np.mean(basin_nses) if basin_nses else float('-inf')
    return results, mean_nse


def evaluate(user_cfg: Dict):
    """Evaluate model, selecting the best checkpoint from the last N epochs.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
    """
    with open(user_cfg["run_dir"] / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)

    if user_cfg.get("basin_file"):
        with open(user_cfg["basin_file"]) as f:
            basins = [line.strip() for line in f if line.strip()]
        print(f"Using custom basin list: {user_cfg['basin_file']} ({len(basins)} basins)")
    elif run_cfg.get("basin_file"):
        with open(run_cfg["basin_file"]) as f:
            basins = [line.strip() for line in f if line.strip()]
        print(f"Using basin list from training config: {run_cfg['basin_file']} ({len(basins)} basins)")
    else:
        basins = get_basin_list()

    # get attribute means/stds
    db_path = str(user_cfg["run_dir"] / "attributes.db")
    attributes = load_attributes(db_path=db_path,
                                 basins=basins,
                                 drop_lat_lon=True)
    means = attributes.mean()
    stds = attributes.std()

    # create model (architecture only, weights loaded per-epoch)
    pred_days = run_cfg.get("pred_days", 0)
    out_size = max(pred_days, 1)
    input_size_stat = 0 if run_cfg["no_static"] else 27
    if run_cfg.get("use_starter_features", False):
        from papercode.datautils import N_STARTER_FEATURES
        extra = N_STARTER_FEATURES
    else:
        extra = 0
    input_size_dyn = (5 + extra) if (run_cfg["no_static"] or not run_cfg["concat_static"]) else (32 + extra)
    model = Model(input_size_dyn=input_size_dyn,
                  input_size_stat=input_size_stat,
                  hidden_size=run_cfg["hidden_size"],
                  dropout=run_cfg["dropout"],
                  concat_static=run_cfg["concat_static"],
                  no_static=run_cfg["no_static"],
                  model_type=run_cfg.get("model_type", "lstm"),
                  pred_days=pred_days,
                  no_attention=run_cfg.get("no_attention", False),
                  no_feature_selection=run_cfg.get("no_feature_selection", False),
                  seq_length=run_cfg["seq_length"],
                  d_model=run_cfg.get("d_model"),
                  n_heads=run_cfg.get("n_heads"),
                  e_layers=run_cfg.get("e_layers"),
                  d_ff=run_cfg.get("d_ff"),
                  patch_head=run_cfg.get("patch_head", "joint"),
                  patch_channels=_parse_channels(run_cfg.get("patch_channels"))).to(DEVICE)

    # Find available checkpoints in the last N epochs
    eval_last_n = user_cfg.get("eval_last_n", 10)
    end_epoch = user_cfg.get("eval_epoch") or run_cfg.get("epochs", 30)
    candidate_epochs = []
    for epoch in range(max(1, end_epoch - eval_last_n + 1), end_epoch + 1):
        weight_file = user_cfg["run_dir"] / f'model_epoch{epoch}.pt'
        if weight_file.is_file():
            candidate_epochs.append(epoch)

    if not candidate_epochs:
        raise FileNotFoundError(
            f"No checkpoints found in {user_cfg['run_dir']} for epochs "
            f"{max(1, total_epochs - eval_last_n + 1)}-{total_epochs}")

    print(f"\n=== Best Checkpoint Selection (eval_last_n={eval_last_n}) ===")
    print(f"Evaluating {len(candidate_epochs)} checkpoints: epochs {candidate_epochs}")

    # Evaluate each candidate checkpoint
    best_epoch = None
    best_nse = float('-inf')
    best_results = None

    for epoch in candidate_epochs:
        weight_file = user_cfg["run_dir"] / f'model_epoch{epoch}.pt'
        results, mean_nse = _evaluate_single_epoch(
            model=model, weight_file=weight_file, basins=basins,
            user_cfg=user_cfg, run_cfg=run_cfg, means=means,
            stds=stds, db_path=db_path, pred_days=pred_days)
        print(f"  Epoch {epoch}: Mean Basin NSE = {mean_nse:.4f}")

        if mean_nse > best_nse:
            best_nse = mean_nse
            best_epoch = epoch
            best_results = results

    print(f"\n>>> Best Checkpoint: epoch {best_epoch} (Mean Basin NSE = {best_nse:.4f})")

    # Print detailed per-step metrics for the best epoch
    print(f"\n=== Detailed Results (pred_days={pred_days}, best_epoch={best_epoch}) ===")
    for step in range(out_size):
        basin_nses = []
        all_obs = []
        all_preds = []
        for basin, data in best_results.items():
            obs_step = data['obs'][:, step]
            pred_step = data['preds'][:, step]
            valid_mask = obs_step >= 0
            if valid_mask.sum() > 0:
                basin_obs = obs_step[valid_mask]
                basin_preds = pred_step[valid_mask]
                try:
                    basin_nse = calc_nse(basin_obs, basin_preds)
                    basin_nses.append(basin_nse)
                except RuntimeError:
                    pass
                all_obs.extend(basin_obs)
                all_preds.extend(basin_preds)

        if pred_days == 0:
            step_label = "Nowcast (day 270)"
        else:
            step_label = f"Day +{step + 1}"

        if len(all_obs) > 0:
            overall_nse = calc_nse(np.array(all_obs), np.array(all_preds))
            mean_nse_step = np.mean(basin_nses) if basin_nses else 0.0
            print(f"  {step_label}: Overall NSE={overall_nse:.4f}, Mean Basin NSE={mean_nse_step:.4f} ({len(basin_nses)} basins)")

    _store_results(user_cfg, run_cfg, best_results, pred_days)


def evaluate_basin(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model on a single basin

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    loader : DataLoader
        PyTorch DataLoader containing the basin data in batches.

    Returns
    -------
    preds : np.ndarray
        Array containing the (rescaled) network prediction for the entire data period
    obs : np.ndarray
        Array containing the observed discharge for the entire data period

    """
    model.eval()

    preds, obs = None, None

    with torch.no_grad():
        for data in loader:
            if len(data) == 2:
                x, y = data
                x, y = x.to(DEVICE), y.to(DEVICE)
                p = model(x)[0]
            elif len(data) == 3:
                x_d, x_s, y = data
                x_d, x_s, y = x_d.to(DEVICE), x_s.to(DEVICE), y.to(DEVICE)
                p = model(x_d, x_s[:, 0, :])[0]

            if preds is None:
                preds = p.detach().cpu()
                obs = y.detach().cpu()
            else:
                preds = torch.cat((preds, p.detach().cpu()), 0)
                obs = torch.cat((obs, y.detach().cpu()), 0)

        preds = rescale_features(preds.numpy(), variable='output')
        obs = obs.numpy()
        # set discharges < 0 to zero
        preds[preds < 0] = 0

    return preds, obs


def eval_robustness(user_cfg: Dict):
    """Evaluate model robustness of EA-LSTM

    In this experiment, gaussian noise with increasing scale is added to the static features to
    evaluate the model robustness against pertubations of the static catchment characteristics.
    For each scale, 50 noise vectors are drawn.
    
    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
    
    Raises
    ------
    NotImplementedError
        If the run_dir specified points not to a EA-LSTM model folder.
    """
    random.seed(user_cfg["seed"])
    np.random.seed(user_cfg["seed"])

    # fixed settings for this analysis
    n_repetitions = 50
    scales = [0.1 * i for i in range(11)]

    with open(user_cfg["run_dir"] / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)

    if run_cfg["concat_static"] or run_cfg["no_static"]:
        raise NotImplementedError("This function is only implemented for EA-LSTM models")

    basins = get_basin_list()

    # get attribute means/stds
    db_path = str(user_cfg["run_dir"] / "attributes.db")
    attributes = load_attributes(db_path=db_path, 
                                 basins=basins,
                                 drop_lat_lon=True)
    means = attributes.mean()
    stds = attributes.std()

    # initialize Model (respect dynamic feature expansion if used in training)
    if run_cfg.get("use_starter_features", False):
        from papercode.datautils import N_STARTER_FEATURES
        extra = N_STARTER_FEATURES
    else:
        extra = 0
    model = Model(input_size_dyn=5 + extra,
                  input_size_stat=27,
                  hidden_size=run_cfg["hidden_size"],
                  dropout=run_cfg["dropout"],
                  model_type=run_cfg.get("model_type", "ealstm"),
                  pred_days=run_cfg.get("pred_days", 0),
                  no_attention=run_cfg.get("no_attention", False),
                  no_feature_selection=run_cfg.get("no_feature_selection", False)).to(DEVICE)
    weight_file = user_cfg["run_dir"] / "model_epoch30.pt"
    model.load_state_dict(torch.load(weight_file, map_location=DEVICE))

    overall_results = {}
    # process bar handle
    pbar = tqdm(basins, file=sys.stdout)
    for basin in pbar:
        ds_test = CamelsTXT(camels_root=user_cfg["camels_root"],
                            basin=basin,
                            dates=[GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]],
                            is_train=False,
                            with_attributes=True,
                            attribute_means=means,
                            attribute_stds=stds,
                            use_starter_features=run_cfg.get("use_starter_features", False),
                            pred_days=run_cfg.get("pred_days", 0),
                            db_path=db_path)
        loader = DataLoader(ds_test, batch_size=len(ds_test), shuffle=False, num_workers=12)
        basin_results = defaultdict(list)
        step = 1
        for scale in scales:
            for _ in range(1 if scale == 0.0 else n_repetitions):
                noise = np.random.normal(loc=0, scale=scale, size=27).astype(np.float32)
                noise = torch.from_numpy(noise).to(DEVICE)
                nse = eval_with_added_noise(model, loader, noise)
                basin_results[scale].append(nse)
                pbar.set_postfix_str(f"Basin progress: {step}/{(len(scales)-1)*n_repetitions+1}")
                step += 1

        overall_results[basin] = basin_results
    out_file = (Path(__file__).absolute().parent /
                f'results/{user_cfg["run_dir"].name}_model_robustness.p')
    if not out_file.parent.is_dir():
        out_file.parent.mkdir(parents=True)
    with out_file.open("wb") as fp:
        pickle.dump(overall_results, fp)


def eval_with_added_noise(model: torch.nn.Module, loader: DataLoader, noise: torch.Tensor) -> float:
    """Evaluate model on a single basin with added noise

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    loader : DataLoader
        PyTorch DataLoader containing the basin data in batches.
    noise : torch.Tensor
        Tensor containing the noise for this evaluation run.
    
    Returns
    -------
    float
        Nash-Sutcliff-Efficiency of the simulations with added noise.
    """
    model.eval()
    preds, obs = None, None
    with torch.no_grad():
        for x_d, x_s, y in loader:
            x_d, x_s, y = x_d.to(DEVICE), x_s.to(DEVICE), y.to(DEVICE)
            batch_noise = noise.repeat(*x_s.size()[:2], 1)
            x_s = x_s.add(batch_noise)
            y_hat = model(x_d, x_s[:, 0, :])[0]

            if preds is None:
                preds = y_hat.detach().cpu()
                obs = y.detach().cpu()
            else:
                preds = torch.cat((preds, y_hat.detach().cpu()), 0)
                obs = torch.cat((obs, y.detach().cpu()), 0)

        obs = obs.numpy()
        preds = rescale_features(preds.numpy(), variable='output')

        # set discharges < 0 to zero
        preds[preds < 0] = 0

        nse = calc_nse(obs[obs >= 0], preds[obs >= 0])
        return nse


def _store_results(user_cfg: Dict, run_cfg: Dict, results: Dict, pred_days: int = 0):
    """Store results with per-step directories for multi-day forecasts.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
    run_cfg : Dict
        Dictionary containing the run config loaded from the cfg.json file
    results : Dict
        Dictionary mapping basin_id -> {'preds': np.ndarray, 'obs': np.ndarray, 'date_range': DatetimeIndex}
    pred_days : int
        0=nowcast, >0=forecast horizon
    """
    out_size = max(pred_days, 1)

    # Create results directory structure
    model_type = run_cfg.get("model_type", "lstm")
    if run_cfg.get("no_static", False):
        model_name = f"{model_type}-no_static"
    elif run_cfg.get("concat_static", False):
        model_name = f"{model_type}-concat_static"
    else:
        if model_type in ("tft", "itransformer", "patchtst"):
            model_name = f"{model_type}-default"
        else:
            model_name = f"{model_type}-ealstm"

    checkpoint_name = f"{model_name}-pred{pred_days}-seed{run_cfg['seed']}"
    results_dir = user_cfg["run_dir"] / "eval_results" / checkpoint_name
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving evaluation results to {results_dir}...")

    # Per-step directories and metrics
    overall_nse_by_step = []
    mean_basin_nse_by_step = []
    all_basins_data = {}

    for step in range(out_size):
        step_dir = results_dir / f"step_{step + 1}"
        step_dir.mkdir(exist_ok=True)

        basin_nses = []
        all_obs_step = []
        all_preds_step = []

        for basin_id, data in results.items():
            obs_step = data['obs'][:, step]
            pred_step = data['preds'][:, step]
            valid_mask = obs_step >= 0
            if valid_mask.sum() == 0:
                continue

            predictions = pred_step[valid_mask].astype(np.float32)
            targets = obs_step[valid_mask].astype(np.float32)
            basin_nse = calc_nse(targets, predictions)
            basin_nses.append(basin_nse)
            all_obs_step.extend(targets)
            all_preds_step.extend(predictions)

            # Save per-basin npz
            basin_file = step_dir / f"basin_{basin_id}.npz"
            np.savez(basin_file,
                     basin_id=str(basin_id),
                     predictions=predictions,
                     targets=targets,
                     nse_score=np.float32(basin_nse),
                     n_samples=np.int64(len(predictions)))

            # Build comprehensive data (accumulate across steps)
            if str(basin_id) not in all_basins_data:
                all_basins_data[str(basin_id)] = {
                    'basin_id': str(basin_id),
                    'predictions': {},
                    'targets': {},
                    'nse_scores': {}
                }
            all_basins_data[str(basin_id)]['predictions'][f'step_{step + 1}'] = predictions
            all_basins_data[str(basin_id)]['targets'][f'step_{step + 1}'] = targets
            all_basins_data[str(basin_id)]['nse_scores'][f'step_{step + 1}'] = np.float32(basin_nse)

        if len(all_obs_step) > 0:
            step_overall_nse = calc_nse(np.array(all_obs_step), np.array(all_preds_step))
        else:
            step_overall_nse = float('nan')
        step_mean_nse = np.mean(basin_nses) if basin_nses else float('nan')

        overall_nse_by_step.append(float(step_overall_nse))
        mean_basin_nse_by_step.append(float(step_mean_nse))

    # Save comprehensive pickle
    comprehensive_file = results_dir / "all_basins_comprehensive.pkl"
    with open(comprehensive_file, 'wb') as f:
        pickle.dump(all_basins_data, f)

    # Save summary statistics JSON
    summary_stats = {
        'checkpoint_name': checkpoint_name,
        'pred_days': pred_days,
        'out_size': out_size,
        'total_basins': len(results),
        'overall_nse_by_step': overall_nse_by_step,
        'mean_basin_nse_by_step': mean_basin_nse_by_step,
    }
    summary_file = results_dir / "summary_stats.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)

    print(f"Successfully stored results:")
    print(f"  - Step directories: step_1 .. step_{out_size}")
    print(f"  - Comprehensive data: {comprehensive_file}")
    print(f"  - Summary statistics: {summary_file}")
    for s in range(out_size):
        print(f"  - Step {s+1}: Overall NSE={overall_nse_by_step[s]:.4f}, Mean Basin NSE={mean_basin_nse_by_step[s]:.4f}")


if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)
