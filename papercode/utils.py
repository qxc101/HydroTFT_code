"""HydroTFT -- HDF5 caching of training sequences for multi-horizon targets.

Modified from the file of the same name in
    https://github.com/kratzert/ealstm_regional_modeling  (Apache-2.0)
for the HydroTFT experiments. Modifications are listed in the accompanying README.
Distributed under the Apache License 2.0; see the LICENSE file.
"""

import sys
from pathlib import Path, PosixPath
from typing import List

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from .datasets import CamelsTXT


def create_h5_files(camels_root: PosixPath,
                    out_file: PosixPath,
                    basins: List,
                    dates: List,
                    with_basin_str: bool = True,
                    seq_length: int = 270,
                    use_starter_features: bool = False,
                    pred_days: int = 0):
    """[summary]
    
    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    out_file : PosixPath
        Path of the location, where the hdf5 file should be stored
    basins : List
        List containing the 8-digit USGS gauge id
    dates : List
        List of start and end date of the discharge period to use, when combining the data.
    with_basin_str : bool, optional
        If True, stores for each sample the corresponding USGS gauged id, by default True
    seq_length : int, optional
        Length of the requested input sequences., by default 270
    
    Raises
    ------
    FileExistsError
        If file at this location already exists.
    """
    if out_file.is_file():
        raise FileExistsError(f"File already exists at {out_file}")

    # Probe first basin to determine number of input features and output size
    probe_basin = basins[0]
    probe_ds = CamelsTXT(camels_root=camels_root,
                         basin=probe_basin,
                         is_train=True,
                         seq_length=seq_length,
                         dates=dates,
                         use_starter_features=use_starter_features,
                         pred_days=pred_days)
    n_features = probe_ds.x.shape[-1]
    out_size = probe_ds.y.shape[-1]  # 1 for nowcast, pred_days for forecast

    with h5py.File(out_file, 'w') as out_f:
        input_data = out_f.create_dataset('input_data',
                                          shape=(0, seq_length, n_features),
                                          maxshape=(None, seq_length, n_features),
                                          chunks=True,
                                          dtype=np.float32,
                                          compression='gzip')
        target_data = out_f.create_dataset('target_data',
                                           shape=(0, out_size),
                                           maxshape=(None, out_size),
                                           chunks=True,
                                           dtype=np.float32,
                                           compression='gzip')

        q_stds = out_f.create_dataset('q_stds',
                                      shape=(0, 1),
                                      maxshape=(None, 1),
                                      dtype=np.float32,
                                      compression='gzip',
                                      chunks=True)

        if with_basin_str:
            sample_2_basin = out_f.create_dataset('sample_2_basin',
                                                  shape=(0, ),
                                                  maxshape=(None, ),
                                                  dtype="S10",
                                                  compression='gzip',
                                                  chunks=True)

        for basin in tqdm(basins, file=sys.stdout):

            dataset = CamelsTXT(camels_root=camels_root,
                                basin=basin,
                                is_train=True,
                                seq_length=seq_length,
                                dates=dates,
                                use_starter_features=use_starter_features,
                                pred_days=pred_days)

            num_samples = len(dataset)
            total_samples = input_data.shape[0] + num_samples

            # store input and output samples
            input_data.resize((total_samples, seq_length, n_features))
            target_data.resize((total_samples, out_size))
            input_data[-num_samples:, :, :] = dataset.x
            target_data[-num_samples:, :] = dataset.y

            # additionally store std of discharge of this basin for each sample
            q_stds.resize((total_samples, 1))
            q_std_array = np.array([dataset.q_std] * num_samples, dtype=np.float32).reshape(-1, 1)
            q_stds[-num_samples:, :] = q_std_array

            if with_basin_str:
                sample_2_basin.resize((total_samples, ))
                str_arr = np.array([basin.encode("ascii", "ignore")] * num_samples)
                sample_2_basin[-num_samples:] = str_arr

            out_f.flush()


def get_basin_list() -> List:
    """Read list of basins from text file.
    
    Returns
    -------
    List
        List containing the 8-digit basin code of all basins
    """
    basin_file = Path(__file__).absolute().parent.parent / "data/basin_list.txt"
    with basin_file.open('r') as fp:
        basins = fp.readlines()
    basins = [basin.strip() for basin in basins]
    return basins
