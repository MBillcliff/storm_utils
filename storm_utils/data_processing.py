"""
HUXt preprocessing utilities for converting raw HUXt outputs to ML-ready format.

Functions for processing HUXt ensemble data, merging with OMNI observations,
and creating the unified dataset for machine learning.
"""

import numpy as np
import pandas as pd
import logging
import re
import glob
import os
from pathlib import Path
from tqdm import tqdm
import fastparquet
import gc

from storm_utils.config_paths import get_project_paths
import storm_utils.huxt_utils as HU

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


# ============================================================================
# OMNI Column Definitions
# ============================================================================

col_names = [
    "year", "doy", "hour", "bartels_rot",
    "imf_sc_id", "plasma_sc_id",
    "n_imf", "n_plasma",
    "B_scalar", "B_vector",
    "B_lat", "B_lon",
    "Bx_GSE", "By_GSE", "Bz_GSE",
    "By_GSM", "Bz_GSM",
    "RMS_mag", "RMS_vec",
    "RMS_Bx", "RMS_By", "RMS_Bz",
    "T_sw", "n_sw", "V_sw",
    "flow_lon", "flow_lat",
    "alpha_ratio",
    "sigma_T", "sigma_n", "sigma_V",
    "sigma_phi", "sigma_theta", "sigma_ratio",
    "flow_pressure", "E_field",
    "plasma_beta", "alfven_mach", "magnetosonic_mach",
    "quasi_invariant",
    "Kp", "sunspot_R", "Dst", "ap",
    "f107", "AE", "AL", "AU",
    "pc_index", "lyman_alpha",
    "pflux_gt1", "pflux_gt2", "pflux_gt4",
    "pflux_gt10", "pflux_gt30", "pflux_gt60",
    "flux_flag"
]

col_widths = [
    4, 4, 3, 5,
    3, 3,
    4, 4,
    6, 6,
    6, 6,
    6, 6, 6,
    6, 6,
    6, 6,
    6, 6, 6,
    9, 6, 6,
    6, 6,
    6,
    9, 6, 6,
    6, 6, 6,
    6, 7,
    7, 6, 5,
    7,
    3, 4, 6, 4,
    6, 5, 6, 6,
    6, 9,
    10, 9, 9,
    9, 9, 9,
    3
]

fill_values = {
    'bartels_rot': 9999,
    'imf_sc_id': 99,
    'plasma_sc_id': 99,
    'n_imf': 999,
    'n_plasma': 999,
    'B_scalar': 999.9,
    'B_vector': 999.9,
    'B_lat': 999.9,
    'B_lon': 999.9,
    'Bx_GSE': 999.9,
    'By_GSE': 999.9,
    'Bz_GSE': 999.9,
    'By_GSM': 999.9,
    'Bz_GSM': 999.9,
    'RMS_mag': 999.9,
    'RMS_vec': 999.9,
    'RMS_Bx': 999.9,
    'RMS_By': 999.9,
    'RMS_Bz': 999.9,
    'T_sw': 9999999.,
    'n_sw': 999.9,
    'V_sw': 9999.,
    'flow_lon': 999.9,
    'flow_lat': 999.9,
    'alpha_ratio': 9.999,
    'sigma_T': 9999999.,
    'sigma_n': 999.9,
    'sigma_V': 9999.,
    'sigma_phi': 999.9,
    'sigma_theta': 999.9,
    'sigma_ratio': 9.999,
    'flow_pressure': 99.99,
    'E_field': 999.99,
    'plasma_beta': 999.99,
    'alfven_mach': 999.9,
    'magnetosonic_mach': 9.9,
    'quasi_invariant': 99.99,
    'Kp': 99,
    'sunspot_R': 999,
    'Dst': 99999,
    'ap': 999,
    'f107': 999.9,
    'AE': 99999,
    'AL': 99999,
    'AU': 99999,
    'pc_index': 999.9,
    'lyman_alpha': 0.999999,
    'pflux_gt1': 999999.99,
    'pflux_gt2': 99999.99,
    'pflux_gt4': 99999.99,
    'pflux_gt10': 99999.99,
    'pflux_gt30': 99999.99,
    'pflux_gt60': 99999.99,
    'flux_flag': -1
}


# ============================================================================
# OMNI Data Loading and Processing
# ============================================================================

def process_full_omni(data_dir=None, save=True, save_format='parquet'):
    """
    Load and process OMNI solar wind data.
    
    Adds derived variables (coupling functions, IMF clock angle, etc.)
    and handles fill values.

    Parameters
    ----------
    data_dir : Path or str, optional
        Directory where 'omni_full.txt' is stored. If None, uses project paths.
    save : bool
        Whether to save the processed DataFrame
    save_format : str
        'parquet' or 'csv'

    Returns
    -------
    pd.DataFrame
        Processed OMNI data with datetime index
    """
    if data_dir is None:
        paths = get_project_paths()
        data_dir = paths['data_shared']

    file_path = data_dir / 'omni_full.txt'
    if not file_path.exists():
        raise FileNotFoundError(f"OMNI data not found at {file_path}")

    logger.info(f"Loading OMNI data from {file_path}")
    
    df = pd.read_fwf(
        file_path,
        widths=col_widths,
        names=col_names,
    )

    # Create datetime index
    datetime_index = (pd.to_datetime(df['year'] * 1000 + df['doy'], format='%Y%j') 
                     + pd.to_timedelta(df['hour'], unit='h'))
    df.index = datetime_index
    df.drop(columns=['year', 'doy', 'hour'], inplace=True)

    # Replace fill values with NaN
    for col, fill in fill_values.items():
        if col in df.columns:
            df[col] = df[col].replace(fill, np.nan).astype(float)
    
    # Interpolate missing values
    df = df.interpolate(method='time').ffill().bfill()

    # Add derived variables
    df['B_GSM'] = np.sqrt(df['Bx_GSE']**2 + df['By_GSM']**2 + df['Bz_GSM']**2)
    
    # Clock angles
    df['theta_c'] = np.arctan2(df['By_GSM'], df['Bz_GSM'])
    df['theta_boyle'] = np.arctan2(np.abs(df['By_GSM']), df['Bz_GSM'])
    
    # Bt = sqrt(By^2 + Bz^2)
    Bt = np.sqrt(df['By_GSM']**2 + df['Bz_GSM']**2)
    
    # Coupling functions
    df['Boyle'] = (1e-4 * df['V_sw']**2) + (11.7 * Bt * np.sin(df['theta_boyle']/2)**3)
    df['Newell'] = (df['V_sw']**(4/3)) * (Bt**(2/3)) * (np.abs(np.sin(df['theta_c']/2))**(8/3))
    df['Viscous'] = df['V_sw'] * Bt * np.sin(df['theta_c']/2)**2

    logger.info(f"Processed OMNI data: {len(df)} timesteps from {df.index[0]} to {df.index[-1]}")

    if save:
        out_path = data_dir / f'omni_full.{save_format}'
        if save_format == 'parquet':
            df.to_parquet(out_path)
        elif save_format == 'csv':
            df.to_csv(out_path)
        else:
            raise ValueError("save_format must be 'parquet' or 'csv'")
        logger.info(f"Saved processed OMNI data to: {out_path}")

    return df


def load_full_omni(data_dir=None):
    """
    Load processed OMNI data.
    
    Parameters
    ----------
    data_dir : Path, optional
        Directory containing processed OMNI file
    
    Returns
    -------
    pd.DataFrame
        Processed OMNI data
    """
    if data_dir is None:
        paths = get_project_paths()
        data_dir = paths['data_shared']
    
    omni_path = data_dir / 'omni_full.parquet'
    
    if not omni_path.exists():
        logger.info("Processed OMNI file not found, creating it now...")
        return process_full_omni(data_dir=data_dir, save=True)
    
    return pd.read_parquet(omni_path)


def load_hp30_data(data_dir=None):
    """
    Load Hp30 geomagnetic index data.
    
    Parameters
    ----------
    data_dir : Path, optional
        Directory containing Hp30 file
    
    Returns
    -------
    pd.DataFrame
        Hp30 data with datetime index
    """
    if data_dir is None:
        paths = get_project_paths()
        data_dir = paths['data_shared']
    
    hp30_path = data_dir / 'hp30.parquet'  # Adjust filename as needed
    
    if not hp30_path.exists():
        raise FileNotFoundError(f"Hp30 data not found at {hp30_path}")
    
    return pd.read_parquet(hp30_path)


def load_icme_catalog(filepath=None, verbose=True):
    """
    Load Richardson & Cane ICME catalog directly from Excel file.
    
    No manual preprocessing required - handles merged headers and data cleaning automatically.
    
    Parameters
    ----------
    filepath : str or Path
        Path to ICME catalog Excel file
    verbose : bool
        Print summary statistics
    
    Returns
    -------
    pd.DataFrame
        Cleaned ICME catalog with parsed datetime columns
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from storm_utils.config_paths import get_project_paths

    paths = get_project_paths()
    
    if filepath is None:
        filepath = paths['data_shared'] / 'icmetable2.xlsx'
    # Read Excel file - try different approaches
    try:
        # First attempt: Read with merged header handling
        df = pd.read_excel(filepath, engine='openpyxl', header=0)

        if verbose:
            print(f"Initial load: {len(df)} rows, {len(df.columns)} columns")
            print(f"Columns: {df.columns.tolist()[:5]}...")
    except Exception as e:
        print(f"Error loading file: {e}")
        raise
    
    # Remove completely empty rows (year separator rows)
    df = df.dropna(how='all')
    
    # Remove rows where first column looks like a year (e.g., "1996", "1997")
    first_col = df.columns[0]
    if df[first_col].dtype == 'object':
        # Check if values are just 4-digit years
        year_pattern = df[first_col].astype(str).str.match(r'^\d{4}$')
        df = df[~year_pattern.fillna(False)]
    
    # Reset index after removing rows
    df = df.reset_index(drop=True)
    
    # Remove last column if it's empty (column S mentioned in comments)
    if df.iloc[:, -1].isnull().all():
        df = df.iloc[:, :-1]
    
    if verbose:
        print(f"After cleaning: {len(df)} rows, {len(df.columns)} columns")
    
    # Define proper column names based on the catalog structure
    # Adjust these based on what you see in df.columns
    expected_cols = {
        0: 'Disturbance_Date',
        1: 'ICME_Plasma_Field_Start',
        2: 'ICME_Plasma_Field_End',
        3: 'Comp_Start_hrs',
        4: 'Comp_End_hrs',
        5: 'MC_Start_hrs',
        6: 'MC_End_hrs',
        7: 'BDE',
        8: 'BIF',
        9: 'Quality',
        10: 'dV',
        11: 'V_ICME',
        12: 'V_max',
        13: 'B',
        14: 'MC',
        15: 'Dst',
        16: 'V_transit',
        17: 'LASCO_CME_Date',
    }
    
    # Rename columns
    df = df.rename(columns={df.columns[i]: name for i, name in expected_cols.items() if i < len(df.columns)})
    
    # Parse datetime columns MORE CAREFULLY
    datetime_cols = ['Disturbance_Date', 'ICME_Plasma_Field_Start', 
                    'ICME_Plasma_Field_End', 'LASCO_CME_Date']
    
    for col in datetime_cols:
        if col in df.columns:
            # Skip if already datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                if verbose:
                    print(f"{col}: Already datetime")
                continue
            
            # Clean string values
            if df[col].dtype == 'object':
                # Convert to string and clean
                df[col] = df[col].astype(str).str.strip()
                
                # Replace '...' and 'nan' with empty string
                df[col] = df[col].replace(['...', 'nan', 'None'], '')
                
                # Remove trailing letters (like " H", " P", " Q")
                df[col] = df[col].str.replace(r'\s+[A-Z]+$', '', regex=True)
            
            # Parse datetime - let pandas infer format rather than strict format
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
            n_valid = df[col].notna().sum()
            if verbose:
                print(f"Parsed {col}: {n_valid}/{len(df)} valid dates")
        
    # Clean numeric columns - replace '...' with NaN
    numeric_cols = ['dV', 'V_ICME', 'V_max', 'B', 'Dst', 'V_transit',
                   'Comp_Start_hrs', 'Comp_End_hrs', 'MC_Start_hrs', 'MC_End_hrs']
    
    for col in numeric_cols:
        if col in df.columns:
            # Replace '...' and convert to numeric
            df[col] = pd.to_numeric(df[col].replace('...', np.nan), errors='coerce')
    
    # Clean integer columns
    int_cols = ['BDE', 'BIF', 'MC']
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"ICME Catalog Summary")
        print(f"{'='*60}")
        print(f"Total events: {len(df)}")
        print(f"Date range: {df['Disturbance_Date'].min()} to {df['Disturbance_Date'].max()}")
        print(f"\nDst statistics:")
        print(df['Dst'].describe())
        print(f"{'='*60}\n")
    
    return df


def parse_icme_datetime(date_str):
    """Parse ICME datetime format: '1996/05/27 1500' → datetime"""
    if pd.isna(date_str) or str(date_str).strip() in ['...', '']:
        return pd.NaT
    try:
        date_str = str(date_str).strip()
        # Try format: YYYY/MM/DD HHMM
        return pd.to_datetime(date_str, format='%Y/%m/%d %H%M')
    except:
        try:
            # Try alternative format
            return pd.to_datetime(date_str)
        except:
            return pd.NaT


# ============================================================================
# HUXt Processing Functions
# ============================================================================

def add_solar_wind_flags_to_df(df, icme_df, sir_velocity_threshold=100, sir_time_window_hours=12, verbose=False):
    """
    Add solar wind event flags to the main dataframe.
    
    Memory-efficient version that doesn't create huge broadcast arrays.
    """
    from scipy.ndimage import maximum_filter1d, minimum_filter1d

    if verbose:
        print("Adding solar wind event flags...")
    
    # ===== ICME Flags =====
    icme_starts = pd.to_datetime(icme_df['ICME_Plasma_Field_Start'], errors='coerce')
    icme_ends = pd.to_datetime(icme_df['ICME_Plasma_Field_End'], errors='coerce')
    
    # Remove NaT
    valid_mask = ~(pd.isna(icme_starts) | pd.isna(icme_ends))
    icme_starts = icme_starts[valid_mask]
    icme_ends = icme_ends[valid_mask]

    if verbose:
        print(f"  Found {len(icme_starts)} valid ICME intervals")
    
    # Efficient flagging (loop instead of broadcast)
    icme_flag = np.zeros(len(df), dtype=bool)
    
    for start, end in zip(icme_starts, icme_ends):
        mask = (df.index >= start) & (df.index <= end)
        icme_flag |= mask
    
    df["ICME_flag"] = icme_flag.astype(int)
    
    # ===== MC Flags =====
    if 'MC_Start_hrs' in icme_df.columns and 'MC_End_hrs' in icme_df.columns:
        icme_df_valid = icme_df[valid_mask].reset_index(drop=True)
        
        # Clean and clip offset hours
        mc_start_hrs = pd.to_numeric(icme_df_valid["MC_Start_hrs"], errors='coerce').fillna(0)
        mc_end_hrs = pd.to_numeric(icme_df_valid["MC_End_hrs"], errors='coerce').fillna(0)
        
        mc_start_hrs = np.clip(mc_start_hrs, -100, 100)
        mc_end_hrs = np.clip(mc_end_hrs, -100, 200)
        
        mc_starts = icme_starts + pd.to_timedelta(mc_start_hrs, unit="h")
        mc_ends = icme_ends + pd.to_timedelta(mc_end_hrs, unit="h")
        
        # Remove NaT
        mc_valid = ~(pd.isna(mc_starts) | pd.isna(mc_ends))
        mc_starts = mc_starts[mc_valid]
        mc_ends = mc_ends[mc_valid]

        if verbose:
            print(f"  Found {len(mc_starts)} valid MC intervals")
        
        # Efficient flagging
        mc_flag = np.zeros(len(df), dtype=bool)
        
        for start, end in zip(mc_starts, mc_ends):
            mask = (df.index >= start) & (df.index <= end)
            mc_flag |= mask
        
        df["MC_flag"] = mc_flag.astype(int)
    else:
        df["MC_flag"] = 0

    if verbose:
        print(f"  ICME flagged: {df['ICME_flag'].sum()} timesteps ({df['ICME_flag'].sum()/len(df)*100:.1f}%)")
        print(f"  MC flagged: {df['MC_flag'].sum()} timesteps ({df['MC_flag'].sum()/len(df)*100:.1f}%)")
    
    # ===== SIR Flag =====
    v_sw = df['V_sw'].values
    
    # Check for issues
    v_sw = np.nan_to_num(v_sw, nan=400.0)  # Replace NaN with typical value
    
    # Calculate window size
    time_delta_hours = (df.index[1] - df.index[0]).total_seconds() / 3600
    n_points = int(sir_time_window_hours / time_delta_hours)
    window_size = min(n_points * 2 + 1, len(v_sw))  # Don't exceed array length
    
    # Initialize SIR flag
    sir_flag = np.zeros(len(v_sw), dtype=int)
    
    # For each timestep, check the window around it
    half_window = window_size // 2
    
    for i in range(half_window, len(v_sw) - half_window):
        # Extract window
        window_start = i - half_window
        window_end = i + half_window + 1
        v_window = v_sw[window_start:window_end]
        
        # Condition 1: Velocity range > threshold
        v_max = np.max(v_window)
        v_min = np.min(v_window)
        v_range = v_max - v_min
        
        if v_range > sir_velocity_threshold:
            # Condition 2: Max occurs AFTER min (positive gradient)
            idx_max = np.argmax(v_window)
            idx_min = np.argmin(v_window)
            
            if idx_max > idx_min:  # Velocity increases from min to max
                sir_flag[i] = 1

    df["SIR_flag"] = sir_flag
    
    if verbose:
        print(f"  SIR flagged: {df['SIR_flag'].sum()} timesteps ({df['SIR_flag'].sum()/len(df)*100:.1f}%)")
        return df


def process_huxt_to_modified_df(huxt_run_id, Nens, rotation_number=None, 
                                input_path=None, save=False, spinup_days=3):
    """
    Convert a single HUXt ensemble output (one Carrington rotation) into a
    machine-learning-ready DataFrame.

    This function:
    1. Loads a HUXt ensemble Parquet file for a specified Carrington rotation
    2. Removes spin-up period from the start of the time series
    3. Computes the temporal velocity gradient for each ensemble member
    4. Loads OMNI solar wind data, aligns it to the HUXt time index
    5. Computes ensemble velocity residuals relative to OMNI (v_ensemble - V_sw)
    6. Appends full OMNI parameters and Hp30 data
    7. Reorders columns into interleaved layout: [v_i, vgrad_i, vomni_i] per ensemble
    8. Optionally saves the processed DataFrame

    Parameters
    ----------
    huxt_run_id : str or int
        Identifier for the HUXt run
    Nens : int
        Number of ensemble members in the HUXt run
    rotation_number : int, optional
        Carrington rotation number to process. Required unless `input_path` provided.
    input_path : str or Path, optional
        Explicit path to a HUXt rotation Parquet file. Overrides `rotation_number`.
    save : bool, optional (default=False)
        If True, save to HUXt{huxt_run_id}_modified/HUXt_rotation_<CR>.parquet
    spinup_days : int, optional (default=3)
        Number of days to remove from start of rotation (spin-up period)

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with interleaved HUXt ensemble velocities, gradients,
        OMNI-relative residuals, and aligned OMNI and Hp30 parameters
    """
    # Setup paths
    paths = get_project_paths()
    huxt_data_path = paths['huxt_data_shared']
    save_path = huxt_data_path / f'HUXt{huxt_run_id}_modified'
    save_path.mkdir(parents=True, exist_ok=True)

    # Determine load path
    if input_path is not None:
        load_path = Path(input_path)
        cr = int(re.search(r'HUXt_rotation_(\d+)', load_path.name).group(1))
    elif rotation_number is not None:
        load_path = huxt_data_path / f'HUXt{huxt_run_id}' / f'HUXt_rotation_{rotation_number}.parquet'
        cr = rotation_number
    else:
        raise ValueError("Must provide either 'rotation_number' or 'input_path'")

    # Load HUXt data
    huxt_df = pd.read_parquet(load_path)
    n_ensembles = len(huxt_df.columns)

    # Remove spin-up period
    spinup_steps = spinup_days * 24 * 2  # Days × hours × (30-min steps per hour)
    df = huxt_df.iloc[spinup_steps:].copy()
    
    logger.debug(f"CR {cr}: Removed {spinup_days} days ({spinup_steps} steps) of spin-up, "
                f"kept {len(df)} timesteps")

    # Add velocity gradient for each ensemble member
    gradient_cols = {
        f'vgrad_{i}': np.gradient(df[f'v_{i}'].values)
        for i in range(n_ensembles)
    }
    df = pd.concat([df, pd.DataFrame(gradient_cols, index=df.index)], axis=1)

    # Load OMNI and align to HUXt time index
    omni = load_full_omni()
    omni = omni.reindex(df.index, method='ffill')
    
    # Calculate v - OMNI for each ensemble member
    v_columns = [f'v_{i}' for i in range(Nens)]
    v_minus_omni = pd.DataFrame(
        df[v_columns].values - omni['V_sw'].values[:, None],
        columns=[f'vomni_{i}' for i in range(Nens)],
        index=df.index
    )
    df = pd.concat([df, v_minus_omni], axis=1)

    # Merge with OMNI data
    df = pd.merge(df, omni, left_index=True, right_index=True, how='inner')

    # Merge with Hp30 data
    hpodf = load_hp30_data()
    df = pd.merge(df, hpodf, left_index=True, right_index=True, how='inner')

    icme_df = load_icme_catalog(verbose=False)
    df = add_solar_wind_flags_to_df(df, icme_df, 
                                sir_velocity_threshold=150,
                                sir_time_window_hours=12,
                                verbose=False)
    
    # Save if requested
    if save:
        out_path = save_path / f'HUXt_rotation_{cr}.parquet'
        df.to_parquet(out_path)
        logger.debug(f"Saved processed CR {cr} to {out_path}")

    return df


def process_huxt_chunk(dfs, last_index, huxt_run_id):
    """
    Concatenate and append a batch of processed HUXt DataFrames to a cumulative file.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        List of processed DataFrames for consecutive Carrington rotations
    last_index : pd.Timestamp or None
        Final time index from previously saved chunk (to prevent overlap)
    huxt_run_id : int or str
        HUXt run identifier

    Returns
    -------
    last_indices : list of pd.Timestamp
        Final timestamp from each input DataFrame (for discontinuity tracking)
    new_last_index : pd.Timestamp
        Final timestamp of the merged chunk
    """
    if not dfs:
        return [], last_index
    
    last_indices = [df.index[-1] for df in dfs]
    
    # Concatenate all dataframes
    chunk_df = pd.concat(dfs, axis=0, copy=False)
    
    # Remove duplicate timestamps (keep last occurrence)
    chunk_df = chunk_df[~chunk_df.index.duplicated(keep='last')]
    
    # Remove overlap with previous chunk
    if last_index is not None:
        chunk_df = chunk_df.loc[chunk_df.index > last_index]
    
    # Get new last index
    new_last_index = chunk_df.index[-1] if len(chunk_df) > 0 else last_index
    
    # Append to output file
    paths = get_project_paths()
    out_path = paths['huxt_data_shared'] / f'HUXt{huxt_run_id}_modified' / 'full_df.parquet'
    
    fastparquet.write(out_path, chunk_df, compression="snappy", append=out_path.exists())
    
    # Cleanup
    del chunk_df
    gc.collect()
    
    return last_indices, new_last_index


def process_huxt(huxt_run_id, chunk_size=20, save_discontinuities=True, 
                delete_intermediates=True, spinup_days=3):
    """
    Process HUXt ensemble outputs into ML-ready format.
    
    Complete pipeline:
    1. Process each rotation individually (add gradients, align with OMNI)
    2. Merge rotations into single file in chunks (memory efficient)
    3. Save discontinuity timestamps
    4. Clean up intermediate files
    5. Validate output
    
    Parameters
    ----------
    huxt_run_id : int
        HUXt run ID
    chunk_size : int
        Number of rotations to merge at once (memory management)
    save_discontinuities : bool
        Save discontinuity timestamps to .npy file
    delete_intermediates : bool
        Delete per-rotation files after merging to save space
    spinup_days : int
        Days to remove from start of each rotation (spin-up period)
    
    Returns
    -------
    dict
        Processing summary with output paths and statistics
    """
    paths = get_project_paths()
    huxt_shared_data_path = paths['huxt_data_shared'] / f'HUXt{huxt_run_id}'
    save_path = paths['huxt_data_shared'] / f'HUXt{huxt_run_id}_modified'
    save_path.mkdir(parents=True, exist_ok=True)
    
    n_ensembles = HU.get_huxt_ensemble_number(huxt_shared_data_path)
    rotation_numbers = HU.get_rotation_numbers(huxt_shared_data_path)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing HUXt{huxt_run_id} to ML-Ready Format")
    logger.info(f"{'='*60}")
    logger.info(f"Rotations: {len(rotation_numbers)}")
    logger.info(f"Ensembles: {n_ensembles}")
    logger.info(f"Spin-up removal: {spinup_days} days per rotation")
    logger.info(f"Chunk size: {chunk_size} rotations")
    logger.info(f"Output: {save_path}")
    logger.info(f"{'='*60}\n")
    
    # =================================================================
    # STEP 1: Process Individual Rotations
    # =================================================================
    logger.info("STEP 1: Processing individual rotations...")
    
    failed_processing = []
    
    for cr in tqdm(rotation_numbers, desc="Processing rotations"):
        try:
            df = process_huxt_to_modified_df(
                huxt_run_id, 
                Nens=n_ensembles, 
                rotation_number=cr, 
                save=True,
                spinup_days=spinup_days
            )
            del df
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to process CR {cr}: {e}")
            failed_processing.append((cr, str(e)))
            continue
    
    if failed_processing:
        logger.warning(f"\nFailed to process {len(failed_processing)} rotations:")
        for cr, reason in failed_processing[:5]:
            logger.warning(f"  CR {cr}: {reason}")
    
    # =================================================================
    # STEP 2: Merge into Single File
    # =================================================================
    logger.info("\nSTEP 2: Merging into single file...")
    
    output_file = save_path / 'full_df.parquet'
    
    # Remove existing output
    if output_file.exists():
        output_file.unlink()
        logger.info("Removed existing full_df.parquet")
    
    discontinuities = []
    dfs = []
    last_index = None
    
    for i, cr in enumerate(tqdm(rotation_numbers, desc="Merging chunks")):
        modified_path = save_path / f'HUXt_rotation_{cr}.parquet'
        
        if not modified_path.exists():
            logger.warning(f"Skipping CR {cr} - processed file not found")
            continue
        
        try:
            df = pd.read_parquet(modified_path)
            dfs.append(df)
            
            # Record discontinuity (first timestamp of each rotation except first)
            if i > 0:
                discontinuities.append(df.index[0])
            
            # Process chunk when full or at end
            if len(dfs) >= chunk_size or cr == rotation_numbers[-1]:
                last_indices, last_index = process_huxt_chunk(dfs, last_index, huxt_run_id)
                dfs = []
                
        except Exception as e:
            logger.error(f"Failed to merge CR {cr}: {e}")
            continue
    
    # =================================================================
    # STEP 3: Save Discontinuities
    # =================================================================
    if save_discontinuities and discontinuities:
        disc_path = save_path / 'discontinuities.npy'
        np.save(disc_path, np.array(discontinuities))
        logger.info(f"\nSTEP 3: Saved {len(discontinuities)} discontinuities to {disc_path}")
    
    # =================================================================
    # STEP 4: Delete Intermediate Files
    # =================================================================
    if delete_intermediates:
        logger.info("\nSTEP 4: Cleaning up intermediate files...")
        
        modified_files = list(save_path.glob("HUXt_rotation_*.parquet"))
        
        for file_path in tqdm(modified_files, desc="Deleting"):
            try:
                file_path.unlink()
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")
        
        logger.info(f"Deleted {len(modified_files)} intermediate files")
    else:
        logger.info("\nSkipping STEP 4: Not cleaning up intermediate files")
    
    return {
        'output_file': output_file,
        'n_rotations': len(rotation_numbers),
        'n_processed': len(rotation_numbers) - len(failed_processing),
        'failed_processing': failed_processing,
        'n_discontinuities': len(discontinuities)
    }


def validate_processed_huxt(huxt_run_id):
    """
    Validate the processed HUXt file for ML use.
    
    Checks:
    - File exists and is readable
    - No NaN values in critical columns
    - Time index is monotonic
    - Shape is reasonable
    - Discontinuities align with data
    
    Parameters
    ----------
    huxt_run_id : int
        HUXt run ID to validate
    
    Returns
    -------
    dict
        Validation results with validity flag and list of issues
    """
    paths = get_project_paths()
    modified_path = paths['huxt_data_shared'] / f'HUXt{huxt_run_id}_modified'
    
    full_df_path = modified_path / 'full_df.parquet'
    disc_path = modified_path / 'discontinuities.npy'
    
    if not full_df_path.exists():
        raise FileNotFoundError(f"Processed file not found: {full_df_path}")
    
    # Load
    df = pd.read_parquet(full_df_path)
    
    print(f"\n{'='*60}")
    print(f"Validation: HUXt{huxt_run_id}_modified")
    print(f"{'='*60}")
    print(f"File: {full_df_path}")
    print(f"Shape: {df.shape}")
    print(f"Time range: {df.index[0]} to {df.index[-1]}")
    print(f"Duration: {(df.index[-1] - df.index[0]).days} days")
    
    # Check for issues
    issues = []
    
    # Time index monotonic
    if not df.index.is_monotonic_increasing:
        issues.append("Time index not monotonic")
    
    # NaN in velocity columns
    v_cols = [c for c in df.columns if c.startswith('v_')]
    if df[v_cols].isnull().any().any():
        n_nan = df[v_cols].isnull().sum().sum()
        issues.append(f"{n_nan} NaN values in velocity columns")
    
    # NaN in OMNI columns
    omni_cols = ['V_sw', 'Bz_GSM', 'By_GSM']
    for col in omni_cols:
        if col in df.columns and df[col].isnull().any():
            n_nan = df[col].isnull().sum()
            issues.append(f"{n_nan} NaN values in {col}")
    
    # NaN in target
    if 'hp30' in df.columns and df['hp30'].isnull().any():
        n_nan = df['hp30'].isnull().sum()
        issues.append(f"{n_nan} NaN values in hp30")
    
    # Check discontinuities
    if disc_path.exists():
        discontinuities = np.load(disc_path, allow_pickle=True)
        print(f"\nDiscontinuities: {len(discontinuities)}")
        
        # Convert to pandas timestamps if needed
        disc_timestamps = pd.to_datetime(discontinuities)
        
        # Verify all are in dataframe
        missing_disc = [d for d in disc_timestamps if d not in df.index]
        if missing_disc:
            issues.append(f"{len(missing_disc)} discontinuity timestamps not in dataframe")
        
        # Check they're distributed throughout (not all at start/end)
        disc_positions = [df.index.get_loc(d) for d in disc_timestamps if d in df.index]
        if disc_positions:
            print(f"  First discontinuity at position: {min(disc_positions)}")
            print(f"  Last discontinuity at position: {max(disc_positions)}")
            print(f"  Spacing: ~{len(df) / (len(disc_positions) + 1):.0f} timesteps between")
    else:
        issues.append("discontinuities.npy not found")
    
    # Print results
    if issues:
        print(f"\n⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n✓ All validation checks passed")
    
    print(f"{'='*60}\n")
    
    return {'valid': len(issues) == 0, 'issues': issues, 'shape': df.shape}


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Process a HUXt run
    results = process_huxt(
        huxt_run_id=3,
        chunk_size=20,
        save_discontinuities=True,
        delete_intermediates=False,
        spinup_days=3
    )
    
    # Validate the result
    validation = validate_processed_huxt(huxt_run_id=3)
    
    if validation['valid']:
        print("✓ Processing successful and validated!")
    else:
        print("✗ Validation failed - check issues above")



