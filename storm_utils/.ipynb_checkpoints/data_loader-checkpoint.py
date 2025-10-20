# Data loader file
import pandas as pd
import numpy as np
import datetime
import pandas as pd
from sunpy.coordinates.sun import carrington_rotation_time
from astropy.time import Time

from storm_utils.config_paths import get_project_paths
from storm_utils.huxt_utils import get_rotation_numbers, get_huxt_ensemble_number


def load_omni_data(data_dir=None):
    """
    Load the OMNI solar wind data.

    Args:
        data_dir (str or Path, optional): Path to the directory containing OMNI_solar_wind.parquet.
                                          If None, attempts to resolve using get_project_paths().

    Returns:
        pd.DataFrame: Cleaned OMNI solar wind data.
    """
    
    if data_dir is None:
        try:
            paths = get_project_paths()
            data_dir = paths['data_shared']
        except Exception as e:
            raise ValueError("No data_dir provided and get_project_paths() failed.") from e

    omni_path = data_dir / 'omni_solar_wind.parquet'

    if not omni_path.exists():
        raise FileNotFoundError(f"Could not find OMNI data at {omni_path}")

    df = pd.read_parquet(omni_path)

    # Replace large placeholder values with 0 to indicate data gaps
    df = df.where(df <= 9000, 0)

    return df


def load_hp30_data(data_dir=None):
    if data_dir is None:
        try:
            paths = get_project_paths()
            data_dir = paths['data_shared']
        except Exception as e:
            raise ValueError('No data_dir provided and get_project_paths() failed.') from e

    hp30_path = data_dir / 'hp30.parquet'

    if not hp30_path.exists():
        raise FileNotFoundError(f'Could not find Hp30 data at {hp30_path}')

    df = pd.read_parquet(hp30_path)

    return df