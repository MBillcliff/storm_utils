# Functions for organising data

import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
import fastparquet
import os
import glob

from storm_utils.config_paths import get_project_paths
import storm_utils.huxt_utils as HU
from storm_utils.data_loader import load_omni_data, load_hp30_data


def process_hp30_data(data_dir=None, save=True, save_format='parquet'):
    """
    Process downloaded hp30 data from https://kp.gfz.de/en/hp30-hp60/data

    Args:
        data_dir (Path or str, optional): Directory where 'hpodata.txt' is stored.
                                          If None, resolves using get_project_paths().
        save (bool): Whether to save the processed DataFrame.
        save_format (str): 'parquet' or 'csv'

    Returns:
        times (np.ndarray): Array of datetime objects for each hp30 point.
        hpo (np.ndarray): Corresponding Hpo values.
    """
    if data_dir is None:
        paths = get_project_paths()
        data_dir = paths['data_shared']

    filename = data_dir / 'hpodata.txt'

    if not filename.exists():
        raise FileNotFoundError(f"Could not find hp30 data at {filename}")

    headers = ('YYYY', 'MM', 'DD', 'hh.h', 'hh._m', 'days', 'days_m', 'Hpo', 'apo', 'D')

    df = pd.read_csv(filename, delimiter='\s+', names=headers)

    df['datetime_str'] = (df['YYYY'].astype(str) + '-' + 
                      df['MM'].astype(str).str.zfill(2) + '-' + 
                      df['DD'].astype(str).str.zfill(2) + ' ' + 
                      df['hh.h'].astype(int).astype(str) + ':' + 
                      (df['hh.h'] % 1 * 60).astype(int).astype(str).str.zfill(2))
    
    # Convert the combined string to datetime
    df['datetime'] = pd.to_datetime(df['datetime_str'], format='%Y-%m-%d %H:%M')
    
    # Drop the temporary 'datetime_str' column if you don't need it
    df.drop(columns=['datetime_str'], inplace=True)
    times = df['datetime'].to_numpy()
    hp30 = df['Hpo'].to_numpy()

    # Setup the DataFrame with just Hp30 data and datetime index
    df = pd.DataFrame(data=hp30, columns=['hp30'], index=times)
    df.index.name = None

    if save:
        out_path = data_dir / f'hp30.{save_format}'
        if save_format == 'parquet':
            df.to_parquet(out_path)
        elif save_format == 'csv':
            df.to_csv(out_path)
        else:
            raise ValueError("save_format must be 'parquet' or 'csv'")
        print(f"Saved processed Hp30 data to: {out_path}")

    return df


def process_omni_data(data_dir=None, save=True, interpolate=True, save_format='parquet'):
    """
    Loads and processes OMNI solar wind data and optionally saves it.

    Args:
        data_dir (Path or str, optional): Directory where 'omni.txt' is stored.
                                          If None, resolves using get_project_paths().
        save (bool): Whether to save the processed DataFrame.
        save_format (str): 'parquet' or 'csv'

    Returns:
        pd.DataFrame: Processed data (regardless of save flag)
    """
    if data_dir is None:
        paths = get_project_paths()
        data_dir = paths['data_shared']

    file_path = data_dir / 'omni.txt'
    if not file_path.exists():
        raise FileNotFoundError(f"OMNI data not found at {file_path}")

    df = pd.read_csv(
        file_path,
        sep=r'\s+',
        names=['YEAR', 'DOY', 'HR', 'omni_flow_speed'],
        header=None
    )

    datetime_index = pd.to_datetime(df['YEAR'] * 1000 + df['DOY'], format='%Y%j') + pd.to_timedelta(df['HR'], unit='h')
    df.index = datetime_index
    df.drop(columns=['YEAR', 'DOY', 'HR'], inplace=True)

    if save:
        out_path = data_dir / f'omni_solar_wind.{save_format}'
        if save_format == 'parquet':
            df.to_parquet(out_path)
        elif save_format == 'csv':
            df.to_csv(out_path)
        else:
            raise ValueError("save_format must be 'parquet' or 'csv'")
        print(f"Saved processed OMNI data to: {out_path}")

    return df


def process_huxt_to_modified_df(huxt_run_id, extra_columns=None, rotation_number=None, input_path=None, overwrite=False, save=False):
    """
    Function to convert the HUXt ensemble output to a usable dataframe for machine learning purposes
    Must provide either rotation_number or input_path
    
    args:
    - rotation_number : number of the carrinton rotation
    - extra_columns   : array of column names that we may wish to add
                      + options
                         - 'velocity gradient' or 'density'
                         - 'hp30' 
                         - 'omni_flow_speed'
    - huxt_data_dir     : name of the folder where HUXt data is stored

    kwargs:
    - overwrite       : bool, whether to override current modified data (default=False)
    - save            : boolean value - set to "True" to save dataframe to parquet file (default=False)
                        
    returns:
    df   : pandas.DataFrame object with all specified variables
    """
    paths = get_project_paths()
    huxt_data_path = paths['huxt_data_shared']
    save_path = huxt_data_path / f'HUXt{huxt_run_id}_modified'
    save_path.mkdir(parents=True, exist_ok=True)

    # Determine load path
    if input_path is not None:
        load_path = Path(input_path)
        cr = int(re.search(r'HUXt_rotation_(\d+)', load_path.name).group(1))
    elif rotation_number is not None:
        cr = rotation_number
        load_path = huxt_data_path / f'HUXt{huxt_run_id}' / f'HUXt_rotation_{cr}.parquet'
    else:
        raise ValueError("Must provide either 'rotation_number' or 'input_path'")

    out_path = save_path / f'HUXt_rotation_{cr}.parquet'
    
    if out_path.exists() and not overwrite:
        return

    huxt_df = pd.read_parquet(load_path)
    n_ensembles = len(huxt_df.columns)

    df = huxt_df.drop(huxt_df.index[:6*24])

    if extra_columns is None:
        extra_columns = []

    if 'velocity gradient' in extra_columns or 'density' in extra_columns:
        gradient_cols = {
            f'vgrad_{i}': np.gradient(df[f'v_{i}'].values)
            for i in range(n_ensembles)
        }
        df = pd.concat([df, pd.DataFrame(gradient_cols, index=df.index)], axis=1).copy()

    if 'omni_flow_speed' in extra_columns:
        omnidf = load_omni_data()
        omnidf = omnidf.reindex(df.index, method='ffill')
        df = pd.merge(df, omnidf, left_index=True, right_index=True, how='inner')

    df_cadence = df.index[1] - df.index[0]
    if 'Hp30' in extra_columns and df_cadence == pd.Timedelta(minutes=30):
        hpodf = load_hp30_data()
        df = pd.merge(df, hpodf, left_index=True, right_index=True, how='inner')

    if save:
        df.to_parquet(out_path)

    return df


def process_huxt_chunk(dfs, last_index, huxt_run_id, omni, Nens):
    """
    Function to process and save a batch of HUXt dataframes, appending them to a combined Parquet file.
    
    Handles deduplication, OMNI merging, feature engineering, and column reordering before saving.
    
    args:
    - dfs          : list of pandas.DataFrame objects for consecutive Carrington rotations
    - last_index   : latest datetime index from previous chunk (used to avoid overlap)
    - huxt_run_id  : ID of the HUXt run, used to determine output directory
    - omni         : pandas.DataFrame of OMNI solar wind data, used for alignment with ensemble velocities
    - n_ensembles  : int - number of ensembles in the provided id's HUXt data

    Process:
    1. Concatenates input DataFrames and removes duplicate time indices
    2. Removes overlap with previous chunk based on `last_index`
    3. Aligns each timestep with corresponding OMNI data (via forward-fill)
    4. Computes velocity error: ensemble velocity minus OMNI velocity (`v_minus_omni`)
    5. Interleaves ensemble velocity, velocity gradient, and error columns for model readiness
    6. Appends result to a cumulative Parquet file using fastparquet (compressed, append mode)

    returns:
    - last_index : datetime index of the final row in the current chunk, for use in future calls
    """
    if not dfs:  # If there's no data, skip saving
        return

    last_indices = [df.index[-1] for df in dfs]

    # Concatenate and remove duplicates
    chunk_df = pd.concat(dfs, ignore_index=False)

    # Drop duplicated indices while keeping the last occurrence
    chunk_df = chunk_df[~chunk_df.index.duplicated(keep='last')]

    # Remove overlap with the last processed chunk
    if last_index is not None:
        chunk_df = chunk_df.loc[chunk_df.index > last_index]

    # Update last index for next batch
    last_index = chunk_df.index[-1]

    v_columns = chunk_df.columns[:Nens]
    v_grad_columns = chunk_df.columns[Nens:2*Nens]
    remainder = chunk_df.columns[2*Nens:]

    omni = load_omni_data()

    omni_v = omni['omni_flow_speed']
    x = np.arange(len(omni_v))
    non_zero_mask = omni_v != 0
    omni_v = np.interp(x, x[non_zero_mask], omni_v[non_zero_mask])
    omni['omni_flow_speed'] = omni_v

    chunk_df['omni_flow_speed'] = omni.reindex(chunk_df.index, method='ffill')
    
    v_minus_omni = pd.DataFrame(
        chunk_df[v_columns].values - chunk_df['omni_flow_speed'].values[:, None],
        columns=[f'vomni_{i}' for i in range(Nens)],
        index=chunk_df.index
    )

    v_minus_omni_columns = v_minus_omni.columns
    
    chunk_df = pd.concat((chunk_df, v_minus_omni), axis=1)

    # Interleave the columns
    interleaved_columns = []
    for a_col, b_col, c_col in zip(v_columns, v_grad_columns, v_minus_omni_columns):
        interleaved_columns.extend([a_col, b_col, c_col])
    interleaved_columns.extend(remainder)
    
    # Reorder the DataFrame using the new column order
    chunk_df_rearranged = chunk_df[interleaved_columns]

    # Append to output file using fastparquet
    paths = get_project_paths()
    out_path = paths['huxt_data_shared'] / f'HUXt{huxt_run_id}_modified' / 'full_df.parquet'
    fastparquet.write(out_path, chunk_df_rearranged, compression="snappy", append=out_path.exists())

    dfs.clear()

    return last_indices


def process_huxt(huxt_run_id, additional_cols, chunk_size=20, save_discontinuities=False, delete_intermediates=True):
    """
    Function to process and compile modified HUXt ensemble outputs across multiple Carrington rotations 
    into a single DataFrame for machine learning use.

    Automatically detects available rotations from file names in the raw HUXt output folder.

    args:
    - huxt_run_id     : ID of the HUXt run, used to locate input/output directories
    - additional_cols : array of column names to include in the processed data
                      + options:
                         - 'velocity gradient' or 'density'
                         - 'Hp30'
                         - 'omni_flow_speed'

    Process:
    1. Converts each HUXt rotation file to a machine learning-ready format
    2. Optionally adds extra derived features (velocity gradients, hp30)
    3. Collects all rotations into a unified dataset
    4. Aligns HUXt data with OMNI velocity measurements
    5. Saves final DataFrame to a compressed Parquet file

    Notes:
    - If output file already exists, function will skip reprocessing
    - Intermediate and final files are stored in the shared HUXt data directory
    """
    
    paths = get_project_paths()
    huxt_shared_data_path = paths['huxt_data_shared'] / f'HUXt{huxt_run_id}'
    save_path = paths['huxt_data_shared'] / f'HUXt{huxt_run_id}_modified'

    n_ensembles = HU.get_huxt_ensemble_number(huxt_shared_data_path)

    print(f'Processing saved HUXt data at {huxt_shared_data_path}')

    rotation_numbers = HU.get_rotation_numbers(huxt_shared_data_path)

    for cr in rotation_numbers:
        df = process_huxt_to_modified_df(huxt_run_id, additional_cols, rotation_number=cr, save=True, overwrite=False)

    discontinuities = [None]
    dfs = []
    OMNI = load_omni_data()
    output_file = save_path / 'full_df.parquet'
    last_index = None
    if output_file.exists():
        output_file.unlink()
    for cr in rotation_numbers:
        out_path = save_path / f'HUXt_rotation_{cr}.parquet'
        try:
            df = pd.read_parquet(out_path)
            dfs.append(df)
        except Exception:
            print(f'File for CR {cr} not created')
        if cr % chunk_size == 0:
            print('Appending CRs', cr, '->', cr + chunk_size)
            last_indices = process_huxt_chunk(dfs, discontinuities[-1], huxt_run_id, omni=OMNI, Nens=n_ensembles)
            discontinuities.extend(last_indices)
    # Remove None value 
    discontinuities = discontinuities[1:]

    # Check if we save points of HUXt discontinuity
    if save_discontinuities:
        out_path = save_path / 'discontinuities.npy'
        np.save(out_path, discontinuities, allow_pickle=True)
        print(f"File saved at {out_path}")

    # Check if we wish to delete intermediate files
    if delete_intermediates:
        # Glob pattern to match your modified DataFrames
        modified_files = glob.glob(os.path.join(paths['huxt_data_shared'] / f'HUXt{huxt_run_id}_modified' / "HUXt_rotation_*.parquet"))
        
        for file_path in modified_files:
            try:
                os.remove(file_path)
                print(f"Deleted {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

    print(f"File saved at {output_file}")


