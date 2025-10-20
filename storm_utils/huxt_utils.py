import numpy as np
import pandas as pd
import datetime
import time
import requests
import astropy.units as u
from astropy.time import Time, TimeDelta
from sunpy.coordinates.sun import carrington_rotation_time
from io import StringIO
import json
from IPython.display import clear_output
import time
import re
import fastparquet
from storm_utils.config_paths import add_huxt_paths, get_project_paths

add_huxt_paths()
import huxt as H
import huxt_analysis as HA
import huxt_inputs as Hin
import huxt_ensembles as ENS


def get_rotation_numbers(huxt_data_path):
    """Searches a provided directory Path and provides the CRs for which HUXt data exists"""
    files = list(huxt_data_path.glob('HUXt_rotation_*'))
    pattern = re.compile(r'HUXt_rotation_(\d+)')
    return sorted(
        int(match.group(1))
        for file in files
        if (match := pattern.match(file.name))
    )


def get_huxt_ensemble_number(huxt_data_path):
    """Searches a provided directory Path and provides the number of ensembles in the HUXt dataframes"""
# Get all parquet files and sort by filename
    parquet_files = sorted(huxt_data_path.glob('HUXt_rotation_*.parquet'))
    
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {huxt_data_path}")
    
    # Read the first file and get the number of columns
    first_df = pd.read_parquet(parquet_files[0])
    n_ensembles = len(first_df.columns)

    return n_ensembles


def run_multiple_ambient_ensembles(start_cr, n_crs, n_ensemble, huxt_run_id, seed, overwrite=False):
    """
    Function to run multiple ambient HUXt ensembles for a specified time
    
    args:
    - start_cr    : int - which rotation number to start on 
    - n_crs       : int - number of carrington rotations
    - n_ensemble  : int - number of ensemble members
    - huxt_run_id : str (or int) - unique ID for this HUXt_run
    - seed        : int - this fixes the ensemble perturbation parameters

    kwargs:
    - overwrite   : bool - whether to save over previously saved data
                        
    returns:
    - None
    """
    np.random.seed(seed)

    paths = get_project_paths()
    huxt_data_path = paths['huxt_data_shared'] / f'HUXt{huxt_run_id}'
    
    # Create directory if doesn't exist
    huxt_data_path.mkdir(parents=True, exist_ok=True)

    # Create list of Carrington Rotations to simulate
    rotation_numbers = list(range(start_cr, start_cr + n_crs))
    
    # Set some constants for the model 
    HUXT_TIME_SCALE = 3.4497        # magic number used to calibrate the huxt output to hourly output
    SIMTIME = 654                   # Length of simulation (this will give 27.25 days of usable output)
    
    for cr in rotation_numbers:
        print(f'Rotation {cr-start_cr+1} / {n_crs}')

        # Get the start time to the nearest hour
        starttime = carrington_rotation_time(cr)
        starttime = starttime.iso[:13]
        starttime = Time(f'{starttime}:00:00.000')

        print(f'CR start time: {starttime}')

        # Get output map
        vr_map, lons, lats = Hin.get_MAS_vr_map(cr)
        # br_map, br_longs, br_lats = Hin.get_MAS_br_map(cr)

        # Run the ensemble
        times, v_in, v_out = ENS.ambient_ensemble(vr_map, lons, lats,
                                                  starttime=starttime, 
                                                  simtime=SIMTIME*u.hour,
                                                  dt_scale = HUXT_TIME_SCALE,
                                                  N_ens_amb = n_ensemble)

        # Shift times to the nearest hour to adjust for milliseconds error due to HUXt time step
        adjusted_times = times + TimeDelta(15 * 60, format='sec')
        times = Time([t[:14] + ('00' if int(t[14:16]) < 30 else '30') + ':00' for t in adjusted_times.iso]) 

        # Create DataFrame and save
        df = pd.DataFrame(data=v_out.T, columns=[f'v_{i}' for i in range(n_ensemble)], index=times.to_datetime())
        out_path =  huxt_data_path / f'HUXt_rotation_{cr}.parquet'
        df.to_parquet(out_path)
        clear_output(wait=True)
        
