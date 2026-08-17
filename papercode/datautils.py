"""HydroTFT -- CAMELS I/O, normalisation and the five engineered features.

Modified from the file of the same name in
    https://github.com/kratzert/ealstm_regional_modeling  (Apache-2.0)
for the HydroTFT experiments. Modifications are listed in the accompanying README.
Distributed under the Apache License 2.0; see the LICENSE file.
"""

import sqlite3
from pathlib import Path, PosixPath
from typing import List, Tuple

import numpy as np
import pandas as pd
from numba import njit
from math import pi

# CAMELS catchment characteristics ignored in this study
INVALID_ATTR = [
    'gauge_name', 'area_geospa_fabric', 'geol_1st_class', 'glim_1st_class_frac', 'geol_2nd_class',
    'glim_2nd_class_frac', 'dom_land_cover_frac', 'dom_land_cover', 'high_prec_timing',
    'low_prec_timing', 'huc', 'q_mean', 'runoff_ratio', 'stream_elas', 'slope_fdc',
    'baseflow_index', 'hfd_mean', 'q5', 'q95', 'high_q_freq', 'high_q_dur', 'low_q_freq',
    'low_q_dur', 'zero_q_freq', 'geol_porostiy', 'root_depth_50', 'root_depth_99', 'organic_frac',
    'water_frac', 'other_frac'
]

# Maurer mean/std calculated over all basins in period 01.10.1999 until 30.09.2008
SCALER = {
    'input_means': np.array([3.17563234, 372.01003929, 17.31934062, 3.97393362, 924.98004197]),
    'input_stds': np.array([6.94344737, 131.63560881, 10.86689718, 10.3940032, 629.44576432]),
    'output_mean': np.array([1.49996196]),
    'output_std': np.array([3.62443672])
}

# Approximate mean/std for the 5 starter features, derived from SCALER.
# Order: doy_sin, doy_cos, prcp_sum_90, degday_7, wetdays_7
# - doy_sin/cos: exact for sin/cos of uniformly distributed day-of-year
# - prcp_sum_90: 90-day rolling sum of daily precipitation
# - degday_7: 7-day sum of max(0.5*(tmax+tmin), 0)
# - wetdays_7: count of days with prcp > 1mm in last 7 days
STARTER_SCALER = {
    'means': np.array([0.0, 0.0, 286.0, 80.0, 2.1]),
    'stds': np.array([0.7071, 0.7071, 130.0, 55.0, 1.7])
}

# Number of starter features (kept in sync with compute_starter_features)
N_STARTER_FEATURES = 5


def add_camels_attributes(camels_root: PosixPath, db_path: str = None):
    """Load catchment characteristics from txt files and store them in a sqlite3 table
    
    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    db_path : str, optional
        Path to where the database file should be saved. If None, stores the database in the 
        `data` directory in the main folder of this repository., by default None
    
    Raises
    ------
    RuntimeError
        If CAMELS attributes folder could not be found.
    """
    attributes_path = Path(camels_root) / 'camels_attributes_v2.0'

    if not attributes_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob('camels_*.txt')

    # Read-in attributes into one big dataframe
    df = None
    for f in txt_files:
        df_temp = pd.read_csv(f, sep=';', header=0, dtype={'gauge_id': str})
        df_temp = df_temp.set_index('gauge_id')

        if df is None:
            df = df_temp.copy()
        else:
            df = pd.concat([df, df_temp], axis=1)

    # convert huc column to double digit strings
    df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
    df = df.drop('huc_02', axis=1)

    if db_path is None:
        db_path = str(Path(__file__).absolute().parent.parent / 'data' / 'attributes.db')

    with sqlite3.connect(db_path) as conn:
        # insert into databse
        df.to_sql('basin_attributes', conn)

    print(f"Sucessfully stored basin attributes in {db_path}.")


def load_attributes(db_path: str,
                    basins: List,
                    drop_lat_lon: bool = True,
                    keep_features: List = None) -> pd.DataFrame:
    """Load attributes from database file into DataFrame

    Parameters
    ----------
    db_path : str
        Path to sqlite3 database file
    basins : List
        List containing the 8-digit USGS gauge id
    drop_lat_lon : bool
        If True, drops latitude and longitude column from final data frame, by default True
    keep_features : List
        If a list is passed, a pd.DataFrame containing these features will be returned. By default,
        returns a pd.DataFrame containing the features used for training.

    Returns
    -------
    pd.DataFrame
        Attributes in a pandas DataFrame. Index is USGS gauge id. Latitude and Longitude are
        transformed to x, y, z on a unit sphere.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM 'basin_attributes'", conn, index_col='gauge_id')

    # drop rows of basins not contained in data set
    drop_basins = [b for b in df.index if b not in basins]
    df = df.drop(drop_basins, axis=0)

    # drop lat/lon col
    if drop_lat_lon:
        df = df.drop(['gauge_lat', 'gauge_lon'], axis=1)

    # drop invalid attributes
    if keep_features is not None:
        drop_names = [c for c in df.columns if c not in keep_features]
    else:
        drop_names = [c for c in df.columns if c in INVALID_ATTR]

    df = df.drop(drop_names, axis=1)

    return df


def normalize_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """Normalize features using global pre-computed statistics.

    Parameters
    ----------
    feature : np.ndarray
        Data to normalize
    variable : str
        One of ['inputs', 'output'], where `inputs` mean, that the `feature` input are the model
        inputs (meteorological forcing data) and `output` that the `feature` input are discharge
        values.

    Returns
    -------
    np.ndarray
        Normalized features

    Raises
    ------
    RuntimeError
        If `variable` is neither 'inputs' nor 'output'
    """

    if variable == 'inputs':
        feature = (feature - SCALER["input_means"]) / SCALER["input_stds"]
    elif variable == 'output':
        feature = (feature - SCALER["output_mean"]) / SCALER["output_std"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")

    return feature


def rescale_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """Rescale features using global pre-computed statistics.

    Parameters
    ----------
    feature : np.ndarray
        Data to rescale
    variable : str
        One of ['inputs', 'output'], where `inputs` mean, that the `feature` input are the model
        inputs (meteorological forcing data) and `output` that the `feature` input are discharge
        values.

    Returns
    -------
    np.ndarray
        Rescaled features

    Raises
    ------
    RuntimeError
        If `variable` is neither 'inputs' nor 'output'
    """
    if variable == 'inputs':
        feature = feature * SCALER["input_stds"] + SCALER["input_means"]
    elif variable == 'output':
        feature = feature * SCALER["output_std"] + SCALER["output_mean"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")

    return feature


@njit
def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int,
                 pred_days: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape data into LSTM many-to-one/many input samples.

    Parameters
    ----------
    x : np.ndarray
        Input features of shape [num_samples, num_features]
    y : np.ndarray
        Output feature of shape [num_samples, 1]
    seq_length : int
        Length of the requested input sequences.
    pred_days : int
        0 = nowcast (target is last day of input window, output size 1).
        >0 = forecast (targets are the next pred_days days after input window).

    Returns
    -------
    x_new: np.ndarray
        Reshaped input features of shape [num_new, seq_length, num_features]
    y_new: np.ndarray
        Target values of shape [num_new, out_size]
    """
    num_samples, num_features = x.shape

    if pred_days == 0:
        # Nowcast: target is the last day of the input window
        out_size = 1
        target_start = seq_length - 1
    else:
        # Forecast: targets are days after the input window
        out_size = pred_days
        target_start = seq_length

    num_new = num_samples - target_start - out_size + 1
    x_new = np.zeros((num_new, seq_length, num_features))
    y_new = np.zeros((num_new, out_size))

    for i in range(num_new):
        x_new[i, :, :] = x[i:i + seq_length, :]
        for d in range(out_size):
            y_new[i, d] = y[i + target_start + d, 0]

    return x_new, y_new


def load_forcing(camels_root: PosixPath, basin: str) -> Tuple[pd.DataFrame, int]:
    """Load Maurer forcing data from text files.

    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    basin : str
        8-digit USGS gauge id

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the Maurer forcing
    area: int
        Catchment area (read-out from the header of the forcing file)

    Raises
    ------
    RuntimeError
        If not forcing file was found.
    """
    forcing_path = camels_root / 'basin_mean_forcing' / 'maurer_extended'
    files = list(forcing_path.glob('**/*_forcing_leap.txt'))
    file_path = [f for f in files if f.name[:8] == basin]
    if len(file_path) == 0:
        raise RuntimeError(f'No file for Basin {basin} at {file_path}')
    else:
        file_path = file_path[0]

    df = pd.read_csv(file_path, sep='\s+', header=3)
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    # load area from header
    with open(file_path, 'r') as fp:
        content = fp.readlines()
        area = int(content[2])

    return df, area


def load_discharge(camels_root: PosixPath, basin: str, area: int) -> pd.Series:
    """[summary]

    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    basin : str
        8-digit USGS gauge id
    area : int
        Catchment area, used to normalize the discharge to mm/day

    Returns
    -------
    pd.Series
        A Series containing the discharge values.

    Raises
    ------
    RuntimeError
        If no discharge file was found.
    """
    discharge_path = camels_root / 'usgs_streamflow'
    files = list(discharge_path.glob('**/*_streamflow_qc.txt'))
    file_path = [f for f in files if f.name[:8] == basin]
    if len(file_path) == 0:
        raise RuntimeError(f'No file for Basin {basin} at {file_path}')
    else:
        file_path = file_path[0]

    col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    # normalize discharge from cubic feed per second to mm per day
    df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10**6)

    return df.QObs


#############################
# Engineered feature helpers #
#############################

def _compute_time_cyclical_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute cyclical encodings for day-of-year.

    Returns a DataFrame with columns: ['doy_sin', 'doy_cos'] aligned to index.
    """
    # Use 365.25 to be robust to leap years
    doy = index.dayofyear.values.astype(np.float32)
    theta = 2.0 * pi * (doy / 365.25)
    sin_doy = np.sin(theta).astype(np.float32)
    cos_doy = np.cos(theta).astype(np.float32)
    return pd.DataFrame({'doy_sin': sin_doy, 'doy_cos': cos_doy}, index=index)


def _rolling_sum(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).sum()


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    # Exponential moving average with span ~ memory window
    return series.ewm(span=span, adjust=False).mean()


def compute_starter_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Compute 5 engineered dynamic features for TFT.

    Features provide information the LSTM cannot easily learn on its own:
    - doy_sin/cos: seasonal encoding (date not available in raw features)
    - prcp_sum_90: long-term antecedent wetness (beyond LSTM effective memory)
    - degday_7: degree-day melt proxy (non-linear transform of temperature)
    - wetdays_7: precipitation frequency (discrete count, different from amount)

    Should be called on the FULL forcing DataFrame (before date slicing) to
    avoid incomplete rolling windows at sequence boundaries.

    Parameters
    ----------
    df : pd.DataFrame
        Full forcing DataFrame with columns: 'prcp(mm/day)', 'tmax(C)', 'tmin(C)'

    Returns
    -------
    features_df : pd.DataFrame
        DataFrame with 5 features aligned to df.index.
    names : List[str]
        Ordered list of feature column names.
    """
    prcp = df['prcp(mm/day)']
    tmax = df['tmax(C)']
    tmin = df['tmin(C)']

    # Cyclical day-of-year encoding
    time_feats = _compute_time_cyclical_features(df.index)

    # 90-day antecedent precipitation (use min_periods=90 to avoid artifacts)
    p_sum_90 = prcp.rolling(window=90, min_periods=90).sum().fillna(0.0)

    # Degree-day melt proxy: 7-day sum of max(tmean, 0)
    t_mean = 0.5 * (tmax + tmin)
    dd = np.maximum(t_mean, 0.0)
    dd_7 = pd.Series(dd, index=df.index).rolling(window=7, min_periods=7).sum().fillna(0.0)

    # Wet-day count: days with prcp > 1mm in last 7 days
    wet_indicator = (prcp > 1.0).astype(float)
    wet_days_7 = wet_indicator.rolling(window=7, min_periods=7).sum().fillna(0.0)

    feat_dict = {
        'doy_sin': time_feats['doy_sin'],
        'doy_cos': time_feats['doy_cos'],
        'prcp_sum_90': p_sum_90,
        'degday_7': dd_7,
        'wetdays_7': wet_days_7,
    }

    names = list(feat_dict.keys())
    features_df = pd.DataFrame(
        {k: v.astype(np.float32).values for k, v in feat_dict.items()},
        index=df.index
    )
    return features_df, names


def normalize_starter_features(features: np.ndarray) -> np.ndarray:
    """Normalize starter features using STARTER_SCALER.

    Parameters
    ----------
    features : np.ndarray
        Raw starter features of shape [num_timesteps, N_STARTER_FEATURES].

    Returns
    -------
    np.ndarray
        Normalized features with approximately zero mean and unit variance.
    """
    return (features - STARTER_SCALER['means']) / STARTER_SCALER['stds']

