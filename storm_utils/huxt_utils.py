"""
HUXt utilities for running and managing solar wind ensemble simulations.

Improvements:
- Better progress tracking with tqdm
- Robust error handling and failure tracking
- Resume capability for interrupted runs
- Output validation
- Timing statistics
- Memory management
- Comprehensive visualization tools

Functions:
- Data management: get_rotation_numbers, get_huxt_ensemble_number, get_huxt_run_info
- Running simulations: run_multiple_ambient_ensembles
- Validation: validate_huxt_run, load_and_validate_huxt_run
- Visualization: plot_huxt_ensemble, plot_full_huxt_timeseries, compare_rotation_boundaries
"""

import numpy as np
import pandas as pd
import datetime
import time
import re
import logging
from pathlib import Path
from tqdm import tqdm
import astropy.units as u
from astropy.time import Time, TimeDelta
from sunpy.coordinates.sun import carrington_rotation_time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from storm_utils.config_paths import add_huxt_paths, get_project_paths
from storm_utils.data_processing import load_full_omni

add_huxt_paths()
import huxt as H
import huxt_analysis as HA
import huxt_inputs as Hin
import huxt_ensembles as ENS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Management Functions
# ============================================================================

def get_rotation_numbers(huxt_data_path):
    """
    Searches a provided directory Path and provides the CRs for which HUXt data exists.
    
    Parameters
    ----------
    huxt_data_path : Path
        Directory containing HUXt rotation files
    
    Returns
    -------
    list
        Sorted list of Carrington Rotation numbers
    """
    files = list(huxt_data_path.glob('HUXt_rotation_*'))
    pattern = re.compile(r'HUXt_rotation_(\d+)')
    return sorted(
        int(match.group(1))
        for file in files
        if (match := pattern.match(file.name))
    )


def get_huxt_ensemble_number(huxt_data_path):
    """
    Searches a provided directory Path and provides the number of ensembles in the HUXt dataframes.
    
    Parameters
    ----------
    huxt_data_path : Path
        Directory containing HUXt parquet files
    
    Returns
    -------
    int
        Number of ensemble members
    """
    # Get all parquet files and sort by filename
    parquet_files = sorted(huxt_data_path.glob('HUXt_rotation_*.parquet'))
    
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {huxt_data_path}")
    
    # Read the first file and get the number of columns
    first_df = pd.read_parquet(parquet_files[0])
    n_ensembles = len(first_df.columns)

    return n_ensembles


def get_huxt_run_info(huxt_run_id):
    """
    Get summary information about a HUXt run.
    
    Parameters
    ----------
    huxt_run_id : int or str
        HUXt run ID
    
    Returns
    -------
    dict
        Run information including path, rotations, ensemble size, time range, storage
    """
    paths = get_project_paths()
    huxt_data_path = paths['huxt_data_shared'] / f'HUXt{huxt_run_id}'
    
    if not huxt_data_path.exists():
        raise FileNotFoundError(f"HUXt run {huxt_run_id} not found")
    
    rotation_numbers = get_rotation_numbers(huxt_data_path)
    n_ensemble = get_huxt_ensemble_number(huxt_data_path)
    
    # Get time range
    first_file = huxt_data_path / f'HUXt_rotation_{rotation_numbers[0]}.parquet'
    last_file = huxt_data_path / f'HUXt_rotation_{rotation_numbers[-1]}.parquet'
    
    df_first = pd.read_parquet(first_file)
    df_last = pd.read_parquet(last_file)
    
    start_time = df_first.index[0]
    df_index_gap = df_first.index[1] - df_first.index[0]
    end_time = df_last.index[-1]
    
    # Calculate storage
    total_size = sum(f.stat().st_size for f in huxt_data_path.glob('HUXt_rotation_*.parquet'))
    total_size_gb = total_size / (1024**3)
    
    info = {
        'run_id': huxt_run_id,
        'path': huxt_data_path,
        'n_rotations': len(rotation_numbers),
        'rotation_range': (min(rotation_numbers), max(rotation_numbers)),
        'n_ensemble': n_ensemble,
        'time_range': (start_time, end_time),
        'total_size_gb': total_size_gb,
    }
    
    print(f"\n{'='*60}")
    print(f"HUXt Run {huxt_run_id} Information")
    print(f"{'='*60}")
    print(f"Location: {huxt_data_path}")
    print(f"Rotations: {len(rotation_numbers)} (CR {min(rotation_numbers)} to {max(rotation_numbers)})")
    print(f"Time range: {start_time} to {end_time}")
    print(f"Time resolution: {df_index_gap}")
    print(f"Time steps: {len(df_first.index)}")
    print(f"First df (start, end): ({df_first.index[0]}, {df_first.index[-1]})")
    print(f"Ensemble members: {n_ensemble}")
    print(f"Total storage: {total_size_gb:.2f} GB")
    print(f"{'='*60}\n")
    
    return info


# ============================================================================
# Simulation Functions
# ============================================================================

def run_multiple_ambient_ensembles(start_cr, n_crs, n_ensemble, huxt_run_id, seed, overwrite=False):
    """
    Run multiple ambient HUXt ensembles for a specified time period.
    
    Parameters
    ----------
    start_cr : int
        Starting Carrington Rotation number
    n_crs : int
        Number of Carrington Rotations to simulate
    n_ensemble : int
        Number of ensemble members
    huxt_run_id : str or int
        Unique ID for this HUXt run
    seed : int
        Random seed for ensemble perturbations (for reproducibility)
    overwrite : bool, optional
        If True, overwrite existing files. If False, skip existing rotations.
                        
    Returns
    -------
    dict
        Summary statistics including successful rotations, failures, and timing
    """
    np.random.seed(seed)

    paths = get_project_paths()
    huxt_data_path = paths['huxt_data_shared'] / f'HUXt{huxt_run_id}'
    
    # Create directory if doesn't exist
    huxt_data_path.mkdir(parents=True, exist_ok=True)

    # Create list of Carrington Rotations to simulate
    rotation_numbers = list(range(start_cr, start_cr + n_crs))
    
    # Check for existing files if not overwriting
    if not overwrite:
        existing_crs = set(get_rotation_numbers(huxt_data_path))
        rotation_numbers = [cr for cr in rotation_numbers if cr not in existing_crs]
        logger.info(f"Skipping {n_crs - len(rotation_numbers)} existing rotations (overwrite=False)")
    
    if len(rotation_numbers) == 0:
        logger.info("All rotations already exist. Set overwrite=True to regenerate.")
        return {'status': 'skipped', 'successful': 0, 'failed': 0}
    
    logger.info(f"Running HUXt for {len(rotation_numbers)} Carrington Rotations")
    logger.info(f"Parameters: {n_ensemble} ensemble members, seed={seed}, run_id={huxt_run_id}")
    logger.info(f"Output: {huxt_data_path}")
    
    # Set model constants
    HUXT_TIME_SCALE = 30 / 5.7975     # Magic number to calibrate HUXt output to half hourly (30 mins / default mins (5.7975))
    SIMTIME = 654                     # Length of simulation (gives 27.25 days of usable output)
    
    # Track progress and failures
    start_time = time.time()
    failed_rotations = []
    successful_rotations = []
    rotation_times = []
    discontinuity_times = []  
    
    for cr in tqdm(rotation_numbers, desc="Processing CRs"):
        cr_start_time = time.time()
        
        try:
            # Get the start time to the nearest hour
            starttime = carrington_rotation_time(cr)
            starttime = starttime.iso[:13]
            starttime = Time(f'{starttime}:00:00.000')

            # Get MAS solar wind map
            try:
                vr_map, lons, lats = Hin.get_MAS_vr_map(cr)
            except Exception as e:
                logger.warning(f"Could not get MAS map for CR {cr}: {e}")
                failed_rotations.append((cr, f'MAS_map_error: {e}'))
                continue

            # Run the ensemble
            times, v_in, v_out = ENS.ambient_ensemble(
                vr_map, lons, lats,
                lat_rot_sigma = 20*np.pi/180*u.rad,
                lat_dev_sigma = 0*np.pi/180*u.rad,
                long_dev_sigma = 10*np.pi/180*u.rad,
                starttime=starttime, 
                simtime=SIMTIME * u.hour,
                dt_scale=HUXT_TIME_SCALE,
                N_ens_amb=n_ensemble
            )

            rounded_unix = np.round(times.unix)  # round to nearest second
            times = Time(rounded_unix, format="unix", scale=times.scale).to_datetime()

            # Create DataFrame
            df = pd.DataFrame(
                data=v_out.T, 
                columns=[f'v_{i}' for i in range(n_ensemble)], 
                index=times
            )
            
            # Validate output
            expected_shape = (SIMTIME*2 - 1, n_ensemble)
            if df.shape != expected_shape:
                logger.warning(f"CR {cr}: Unexpected shape {df.shape}, expected {expected_shape}")
            
            if df.isnull().any().any():
                n_null = df.isnull().sum().sum()
                logger.warning(f"CR {cr}: {n_null} NaN values in output")
            
            # Save
            out_path = huxt_data_path / f'HUXt_rotation_{cr}.parquet'
            df.to_parquet(out_path)
            
            successful_rotations.append(cr)

            # save discontinuities
            if cr != rotation_numbers[0]:
                discontinuity_times.append(df.index[0])  # First timestamp = discontinuity
            
            # Track timing
            cr_elapsed = time.time() - cr_start_time
            rotation_times.append(cr_elapsed)
            
            # Cleanup to manage memory
            del times, v_in, v_out, df, vr_map
            
            # Periodic garbage collection for large runs
            if len(successful_rotations) % 10 == 0:
                import gc
                gc.collect()
            
        except KeyboardInterrupt:
            logger.info("\nRun interrupted by user")
            break
            
        except Exception as e:
            logger.error(f"Unexpected error processing CR {cr}: {e}")
            failed_rotations.append((cr, f'unexpected_error: {e}'))
            continue
    
    # Calculate statistics
    elapsed = time.time() - start_time
    n_successful = len(successful_rotations)
    n_failed = len(failed_rotations)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"HUXt Run Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Successfully processed: {n_successful} / {len(rotation_numbers)} rotations")
    logger.info(f"Failed: {n_failed}")
    logger.info(f"Total time: {elapsed/3600:.2f} hours")
    
    if n_successful > 0:
        logger.info(f"Average time per rotation: {np.mean(rotation_times):.1f} seconds")
        if n_successful < n_crs:
            logger.info(f"Estimated remaining time: {(n_crs - n_successful) * np.mean(rotation_times) / 3600:.2f} hours")
    
    logger.info(f"Output location: {huxt_data_path}")
    
    if failed_rotations:
        logger.warning(f"\nFailed rotations ({len(failed_rotations)}):")
        for cr, reason in failed_rotations[:10]:  # Show first 10
            logger.warning(f"  CR {cr}: {reason}")
        if len(failed_rotations) > 10:
            logger.warning(f"  ... and {len(failed_rotations) - 10} more")

    if discontinuity_times:
        disc_path = huxt_data_path / 'discontinuities.npy'
        np.save(disc_path, np.array(discontinuity_times))
        logger.info(f"Saved {len(discontinuity_times)} discontinuities to {disc_path}")
    
    logger.info(f"{'='*60}\n")
    
    # Return summary
    return {
        'status': 'complete',
        'successful': n_successful,
        'failed': n_failed,
        'successful_crs': successful_rotations,
        'failed_crs': failed_rotations,
        'elapsed_time': elapsed,
        'avg_time_per_rotation': np.mean(rotation_times) if rotation_times else None,
        'output_path': huxt_data_path
    }


# ============================================================================
# Validation Functions
# ============================================================================

def validate_huxt_run(huxt_run_id, expected_n_ensemble=None, verbose=True):
    """
    Validate a completed HUXt run.
    
    Parameters
    ----------
    huxt_run_id : int or str
        HUXt run ID to validate
    expected_n_ensemble : int, optional
        Expected number of ensemble members
    verbose : bool
        Print detailed information
    
    Returns
    -------
    dict
        Validation results including validity status, missing rotations, and issues
    """
    paths = get_project_paths()
    huxt_data_path = paths['huxt_data_shared'] / f'HUXt{huxt_run_id}'
    
    if not huxt_data_path.exists():
        raise FileNotFoundError(f"HUXt run {huxt_run_id} not found at {huxt_data_path}")
    
    # Get rotation numbers
    rotation_numbers = get_rotation_numbers(huxt_data_path)
    
    if len(rotation_numbers) == 0:
        raise ValueError(f"No HUXt rotation files found in {huxt_data_path}")
    
    # Get ensemble number from first file
    n_ensemble = get_huxt_ensemble_number(huxt_data_path)
    
    if expected_n_ensemble is not None and n_ensemble != expected_n_ensemble:
        logger.warning(f"Expected {expected_n_ensemble} ensembles, found {n_ensemble}")
    
    # Check for gaps in rotation numbers
    expected_rotations = set(range(min(rotation_numbers), max(rotation_numbers) + 1))
    missing_rotations = expected_rotations - set(rotation_numbers)
    
    # Sample a few files for validation
    sample_files = [rotation_numbers[0], rotation_numbers[len(rotation_numbers)//2], rotation_numbers[-1]]
    
    issues = []
    for cr in sample_files:
        filepath = huxt_data_path / f'HUXt_rotation_{cr}.parquet'
        df = pd.read_parquet(filepath)
        
        # Check shape
        if df.shape[1] != n_ensemble:
            issues.append(f"CR {cr}: Wrong ensemble count ({df.shape[1]} vs {n_ensemble})")
        
        # Check for NaNs
        if df.isnull().any().any():
            n_nan = df.isnull().sum().sum()
            issues.append(f"CR {cr}: Contains {n_nan} NaN values")
        
        # Check time index
        if not df.index.is_monotonic_increasing:
            issues.append(f"CR {cr}: Time index not monotonic")
    
    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"HUXt Run {huxt_run_id} Validation")
        print(f"{'='*60}")
        print(f"Location: {huxt_data_path}")
        print(f"Rotations: {len(rotation_numbers)} (CR {min(rotation_numbers)} to {max(rotation_numbers)})")
        print(f"Ensemble members: {n_ensemble}")
        
        if missing_rotations:
            print(f"\nMissing rotations: {len(missing_rotations)}")
            if len(missing_rotations) <= 10:
                print(f"  {sorted(missing_rotations)}")
            else:
                print(f"  {sorted(list(missing_rotations))[:10]} ... and {len(missing_rotations)-10} more")
        else:
            print(f"\n✓ No gaps in rotation sequence")
        
        if issues:
            print(f"\nIssues found:")
            for issue in issues:
                print(f"  ⚠ {issue}")
        else:
            print(f"\n✓ All sampled files passed validation")
        
        print(f"{'='*60}\n")
    
    return {
        'run_id': huxt_run_id,
        'n_rotations': len(rotation_numbers),
        'n_ensemble': n_ensemble,
        'rotation_range': (min(rotation_numbers), max(rotation_numbers)),
        'missing_rotations': sorted(missing_rotations),
        'issues': issues,
        'valid': len(issues) == 0 and len(missing_rotations) == 0
    }


def load_and_validate_huxt_run(huxt_run_id=1, sample_cr=None, verbose=True):
    """
    Load a HUXt run and validate it was created correctly.
    
    Parameters
    ----------
    huxt_run_id : int
        HUXt run ID to load
    sample_cr : int, optional
        Specific Carrington Rotation to inspect. If None, uses middle rotation.
    verbose : bool
        Print detailed information
    
    Returns
    -------
    dict
        Validation results and sample data
    """
    paths = get_project_paths()
    huxt_data_path = paths['huxt_data_shared'] / f'HUXt{huxt_run_id}'
    
    if not huxt_data_path.exists():
        raise FileNotFoundError(f"HUXt run {huxt_run_id} not found at {huxt_data_path}")
    
    # Get available rotations
    rotation_numbers = get_rotation_numbers(huxt_data_path)
    n_ensemble = get_huxt_ensemble_number(huxt_data_path)
    
    if len(rotation_numbers) == 0:
        raise ValueError(f"No rotation files found in {huxt_data_path}")
    
    # Select rotation to inspect
    if sample_cr is None:
        sample_cr = rotation_numbers[len(rotation_numbers) // 2]  # Middle rotation
    elif sample_cr not in rotation_numbers:
        print(f"CR {sample_cr} not available, using CR {rotation_numbers[0]}")
        sample_cr = rotation_numbers[0]
    
    # Load sample rotation
    sample_file = huxt_data_path / f'HUXt_rotation_{sample_cr}.parquet'
    df = pd.read_parquet(sample_file)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"HUXt Run {huxt_run_id} - Validation")
        print(f"{'='*80}")
        print(f"Location: {huxt_data_path}")
        print(f"Total Rotations: {len(rotation_numbers)}")
        print(f"CR Range: {min(rotation_numbers)} to {max(rotation_numbers)}")
        print(f"Ensemble Members: {n_ensemble}")
        print(f"\nSample Rotation: CR {sample_cr}")
        print(f"  Shape: {df.shape} (timesteps, ensembles)")
        print(f"  Time range: {df.index[0]} to {df.index[-1]}")
        print(f"  Duration: {(df.index[-1] - df.index[0]).days} days")
        print(f"\nVelocity Statistics (all ensembles):")
        print(f"  Mean: {df.mean().mean():.1f} km/s")
        print(f"  Std (across time): {df.std().mean():.1f} km/s")
        print(f"  Std (across ensembles): {df.mean(axis=0).std():.1f} km/s")
        print(f"  Min: {df.min().min():.1f} km/s")
        print(f"  Max: {df.max().max():.1f} km/s")
        print(f"\nData Quality:")
        print(f"  NaN values: {df.isnull().sum().sum()}")
        print(f"  Negative values: {(df < 0).sum().sum()}")
        print(f"  Unrealistic values (>1000 km/s): {(df > 1000).sum().sum()}")
        print(f"{'='*80}\n")
    
    return {
        'df': df,
        'n_rotations': len(rotation_numbers),
        'rotation_numbers': rotation_numbers,
        'n_ensemble': n_ensemble,
        'sample_cr': sample_cr,
        'huxt_data_path': huxt_data_path
    }


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_huxt_ensemble(huxt_run_id=1, sample_cr=None, n_members_to_plot=20):
    """
    Plot HUXt ensemble to visually verify it looks correct.
    
    Parameters
    ----------
    huxt_run_id : int
        HUXt run ID
    sample_cr : int, optional
        Carrington Rotation to plot. If None, uses middle rotation.
    n_members_to_plot : int
        Number of ensemble members to plot (default 20 to avoid clutter)
    """
    # Load and validate
    validation = load_and_validate_huxt_run(huxt_run_id, sample_cr, verbose=True)
    df = validation['df']
    sample_cr = validation['sample_cr']
    n_ensemble = validation['n_ensemble']
    omni_V_sw = load_full_omni()['V_sw']
    omni_V_sw = omni_V_sw.loc[df.index[0]: df.index[-1]]
    
    # Limit ensemble members for plotting
    n_plot = min(n_members_to_plot, n_ensemble)
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Convert index to hours from start
    hours = (df.index - df.index[0]).total_seconds() / 3600
    omni_hours = (omni_V_sw.index - omni_V_sw.index[0]).total_seconds() / 3600
    
    # Panel 1: Individual ensemble members
    ax1 = axes[0]
    for i in range(n_plot):
        ax1.plot(hours, df.iloc[:, i], alpha=0.3, lw=0.5, color='steelblue')

    ax1.plot(omni_hours, omni_V_sw, color='black', label='OMNI V_sw')
    
    ax1.set_ylabel('Solar Wind Velocity (km/s)', fontsize=12)
    ax1.set_title(f'HUXt Run {huxt_run_id} - CR {sample_cr} - Individual Ensemble Members (showing {n_plot}/{n_ensemble})', 
                 fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax1.set_ylim(200, 900)
    
    # Panel 2: Ensemble statistics (mean, percentiles)
    ax2 = axes[1]
    
    ensemble_mean = df.mean(axis=1)
    ensemble_p10 = df.quantile(0.10, axis=1)
    ensemble_p90 = df.quantile(0.90, axis=1)
    ensemble_p25 = df.quantile(0.25, axis=1)
    ensemble_p75 = df.quantile(0.75, axis=1)
    
    ax2.fill_between(hours, ensemble_p10, ensemble_p90, alpha=0.2, color='steelblue', label='10th-90th percentile')
    ax2.fill_between(hours, ensemble_p25, ensemble_p75, alpha=0.4, color='steelblue', label='25th-75th percentile')
    ax2.plot(hours, ensemble_mean, 'k-', lw=2, label='Ensemble mean')
    
    ax2.set_ylabel('Solar Wind Velocity (km/s)', fontsize=12)
    ax2.set_title('Ensemble Statistics', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(200, 900)
    
    # Panel 3: Ensemble spread over time
    ax3 = axes[2]
    
    ensemble_std = df.std(axis=1)
    ensemble_range = df.max(axis=1) - df.min(axis=1)
    
    ax3.plot(hours, ensemble_std, 'b-', lw=1.5, label='Standard deviation')
    ax3.plot(hours, ensemble_range, 'r--', lw=1.5, alpha=0.7, label='Range (max - min)')
    
    ax3.set_xlabel('Hours from CR Start', fontsize=12)
    ax3.set_ylabel('Ensemble Spread (km/s)', fontsize=12)
    ax3.set_title('Ensemble Uncertainty', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print key checks
    print(f"\n{'='*60}")
    print(f"Visual Inspection Checklist:")
    print(f"{'='*60}")
    print(f"✓ Check Panel 1: Do ensemble members look reasonable?")
    print(f"  - Should see variation between members")
    print(f"  - Typical solar wind: 300-600 km/s")
    print(f"  - Fast streams can reach 700-800 km/s")
    print(f"\n✓ Check Panel 2: Does ensemble mean make sense?")
    print(f"  - Should show temporal variability")
    print(f"  - Shaded regions show spread")
    print(f"\n✓ Check Panel 3: Is ensemble spread reasonable?")
    print(f"  - Typical std: 20-50 km/s")
    print(f"  - High spread = high uncertainty")
    print(f"  Current mean std: {ensemble_std.mean():.1f} km/s")
    print(f"  Current mean range: {ensemble_range.mean():.1f} km/s")
    print(f"{'='*60}\n")


def plot_full_huxt_timeseries(huxt_run_id=1, max_rotations=None):
    """
    Plot velocity timeseries across multiple rotations to see discontinuities.
    
    Parameters
    ----------
    huxt_run_id : int
        HUXt run ID
    max_rotations : int, optional
        Maximum number of rotations to plot (for speed). If None, plots all.
    """
    paths = get_project_paths()
    huxt_data_path = paths['huxt_data_shared'] / f'HUXt{huxt_run_id}'
    
    rotation_numbers = get_rotation_numbers(huxt_data_path)
    
    if max_rotations is not None:
        rotation_numbers = rotation_numbers[:max_rotations]
    
    print(f"Loading {len(rotation_numbers)} rotations...")
    
    # Load all rotations
    dfs = []
    for cr in tqdm(rotation_numbers, desc="Loading rotations"):
        df = pd.read_parquet(huxt_data_path / f'HUXt_rotation_{cr}.parquet')
        dfs.append(df)
    
    # Concatenate
    full_df = pd.concat(dfs, axis=0)
    
    print(f"Total timespan: {full_df.index[0]} to {full_df.index[-1]}")
    print(f"Shape: {full_df.shape}")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Panel 1: Sample ensemble members
    ax1 = axes[0]
    n_plot = min(5, full_df.shape[1])
    
    for i in range(n_plot):
        ax1.plot(full_df.index, full_df.iloc[:, i], alpha=0.5, lw=0.5, 
                label=f'Ensemble {i}' if i < 3 else None)
    
    # Mark CR boundaries
    for i in range(len(rotation_numbers) - 1):
        boundary_time = dfs[i+1].index[0]
        ax1.axvline(boundary_time, color='red', linestyle='--', alpha=0.3, lw=1)
    
    ax1.set_ylabel('Solar Wind Velocity (km/s)', fontsize=12)
    ax1.set_title(f'HUXt Run {huxt_run_id} - Sample Ensemble Members (Red = CR Boundaries)', 
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)
    
    # Panel 2: Ensemble mean
    ax2 = axes[1]
    
    ensemble_mean = full_df.mean(axis=1)
    ensemble_std = full_df.std(axis=1)
    
    ax2.plot(full_df.index, ensemble_mean, 'k-', lw=1.5, label='Ensemble mean')
    ax2.fill_between(full_df.index, 
                     ensemble_mean - ensemble_std, 
                     ensemble_mean + ensemble_std,
                     alpha=0.3, color='steelblue', label='±1 std')
    
    # Mark CR boundaries
    for i in range(len(rotation_numbers) - 1):
        boundary_time = dfs[i+1].index[0]
        ax2.axvline(boundary_time, color='red', linestyle='--', alpha=0.5, lw=1.5,
                   label='CR Boundary' if i == 0 else None)
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Solar Wind Velocity (km/s)', fontsize=12)
    ax2.set_title('Ensemble Mean ± Std', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    # Identify discontinuities
    print(f"\nDiscontinuities detected at CR boundaries:")
    print(f"{'='*60}")
    for i in range(len(dfs) - 1):
        v_end = dfs[i].iloc[-1].mean()
        v_start = dfs[i+1].iloc[0].mean()
        jump = abs(v_start - v_end)
        
        if jump > 50:
            flag = "⚠️  LARGE"
        elif jump > 20:
            flag = "⚠️  MODERATE"
        else:
            flag = "✓ SMALL"
        
        print(f"  CR {rotation_numbers[i]:4d} → {rotation_numbers[i+1]:4d}: Δv = {jump:6.1f} km/s {flag}")
    print(f"{'='*60}\n")


def compare_rotation_boundaries(huxt_run_id=1, cr_pair=None):
    """
    Plot the boundary between two consecutive rotations to visualize discontinuities.
    
    Parameters
    ----------
    huxt_run_id : int
        HUXt run ID
    cr_pair : tuple, optional
        Tuple of (cr1, cr2) to compare. If None, uses first consecutive pair.
    """
    paths = get_project_paths()
    huxt_data_path = paths['huxt_data_shared'] / f'HUXt{huxt_run_id}'
    
    rotation_numbers = get_rotation_numbers(huxt_data_path)
    
    if cr_pair is None:
        cr1, cr2 = rotation_numbers[0], rotation_numbers[1]
    else:
        cr1, cr2 = cr_pair
        if cr1 not in rotation_numbers or cr2 not in rotation_numbers:
            raise ValueError(f"CR {cr1} or {cr2} not found in run")
    
    # Load both rotations
    df1 = pd.read_parquet(huxt_data_path / f'HUXt_rotation_{cr1}.parquet')
    df2 = pd.read_parquet(huxt_data_path / f'HUXt_rotation_{cr2}.parquet')
    
    # Extract last 48 hours of cr1 and first 48 hours of cr2
    n_hours = 48
    n_steps = n_hours * 2  # 30-min timesteps
    
    df1_end = df1.iloc[-n_steps:]
    df2_start = df2.iloc[:n_steps]
    
    # Concatenate for plotting
    df_combined = pd.concat([df1_end, df2_start])
    
    # Create time axis centered at discontinuity
    hours = np.arange(len(df_combined)) * 0.5 - n_hours  # Center at 0
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Panel 1: Sample ensemble members
    ax1 = axes[0]
    n_plot = min(10, df1.shape[1])
    
    for i in range(n_plot):
        ax1.plot(hours[:n_steps], df1_end.iloc[:, i], 'b-', alpha=0.4, lw=1,
                label='CR {cr1} ending' if i == 0 else None)
        ax1.plot(hours[n_steps:], df2_start.iloc[:, i], 'r-', alpha=0.4, lw=1,
                label=f'CR {cr2} starting' if i == 0 else None)
    
    ax1.axvline(0, color='black', linestyle='--', lw=2, label='CR Boundary')
    ax1.set_ylabel('Solar Wind Velocity (km/s)', fontsize=12)
    ax1.set_title(f'Rotation Boundary: CR {cr1} → CR {cr2}', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Panel 2: Ensemble mean and spread
    ax2 = axes[1]
    
    mean1 = df1_end.mean(axis=1).values
    mean2 = df2_start.mean(axis=1).values
    std1 = df1_end.std(axis=1).values
    std2 = df2_start.std(axis=1).values
    
    ax2.plot(hours[:n_steps], mean1, 'b-', lw=2, label=f'CR {cr1} (ending)')
    ax2.fill_between(hours[:n_steps], mean1 - std1, mean1 + std1, alpha=0.3, color='blue')
    
    ax2.plot(hours[n_steps:], mean2, 'r-', lw=2, label=f'CR {cr2} (starting)')
    ax2.fill_between(hours[n_steps:], mean2 - std2, mean2 + std2, alpha=0.3, color='red')
    
    ax2.axvline(0, color='black', linestyle='--', lw=2, label='CR Boundary')
    ax2.set_xlabel('Hours from CR Boundary', fontsize=12)
    ax2.set_ylabel('Ensemble Mean ± Std (km/s)', fontsize=12)
    ax2.set_title('Ensemble Mean and Spread', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate discontinuity magnitude
    v_end = df1.iloc[-1].mean()
    v_start = df2.iloc[0].mean()
    jump = abs(v_start - v_end)
    
    print(f"\n{'='*60}")
    print(f"Discontinuity Analysis: CR {cr1} → CR {cr2}")
    print(f"{'='*60}")
    print(f"End of CR {cr1}:   mean = {v_end:.1f} km/s, std = {df1.iloc[-1].std():.1f} km/s")
    print(f"Start of CR {cr2}: mean = {v_start:.1f} km/s, std = {df2.iloc[0].std():.1f} km/s")
    print(f"\nJump magnitude: {jump:.1f} km/s")
    print(f"Jump as % of mean: {jump / ((v_end + v_start)/2) * 100:.1f}%")
    
    if jump > 50:
        print(f"\n⚠️  Large discontinuity detected (>{50} km/s)")
        print(f"   This will be filtered out by ForecastingDataset")
    else:
        print(f"\n✓ Discontinuity within acceptable range (<50 km/s)")
    
    print(f"{'='*60}\n")


def detect_all_discontinuities(huxt_run_id=1, threshold=50, save=True):
    """
    Detect all discontinuities in a HUXt run and optionally save them.
    
    Parameters
    ----------
    huxt_run_id : int
        HUXt run ID
    threshold : float
        Velocity jump threshold (km/s) to flag as significant discontinuity
    save : bool
        If True, save discontinuity timestamps to .npy file
    
    Returns
    -------
    list
        List of discontinuity timestamps
    """
    paths = get_project_paths()
    huxt_data_path = paths['huxt_data_shared'] / f'HUXt{huxt_run_id}'
    
    rotation_numbers = get_rotation_numbers(huxt_data_path)
    
    discontinuities = []
    large_discontinuities = []
    
    print(f"Detecting discontinuities for HUXt run {huxt_run_id}...")
    print(f"Checking {len(rotation_numbers) - 1} CR boundaries\n")
    
    for i in tqdm(range(len(rotation_numbers) - 1), desc="Checking boundaries"):
        cr1 = rotation_numbers[i]
        cr2 = rotation_numbers[i + 1]
        
        # Load end of first rotation and start of second
        df1 = pd.read_parquet(huxt_data_path / f'HUXt_rotation_{cr1}.parquet')
        df2 = pd.read_parquet(huxt_data_path / f'HUXt_rotation_{cr2}.parquet')
        
        # The discontinuity occurs at the FIRST timestamp of cr2
        discontinuity_time = df2.index[0]
        discontinuities.append(discontinuity_time)
        
        # Check velocity jump
        v_end = df1.iloc[-1].mean()
        v_start = df2.iloc[0].mean()
        jump = abs(v_start - v_end)
        
        if jump > threshold:
            large_discontinuities.append({
                'timestamp': discontinuity_time,
                'cr_before': cr1,
                'cr_after': cr2,
                'jump': jump,
                'v_before': v_end,
                'v_after': v_start
            })
    
    # Save if requested
    if save:
        output_path = huxt_data_path / 'discontinuities.npy'
        np.save(output_path, np.array(discontinuities))
        logger.info(f"Saved {len(discontinuities)} discontinuities to {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Discontinuity Detection Results")
    print(f"{'='*60}")
    print(f"Total CR boundaries: {len(discontinuities)}")
    print(f"Large discontinuities (>{threshold} km/s): {len(large_discontinuities)}")
    
    if large_discontinuities:
        print(f"\nLarge discontinuities:")
        for disc in large_discontinuities[:10]:
            print(f"  {disc['timestamp']}: CR {disc['cr_before']}→{disc['cr_after']}, "
                  f"Δv = {disc['jump']:.1f} km/s")
        if len(large_discontinuities) > 10:
            print(f"  ... and {len(large_discontinuities) - 10} more")
    
    print(f"{'='*60}\n")
    
    return discontinuities


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("HUXt Utilities - Example Usage\n")
    
    # Get run info
    info = get_huxt_run_info(huxt_run_id=1)
    
    # Validate
    validation = validate_huxt_run(huxt_run_id=1, expected_n_ensemble=2000)
    
    if validation['valid']:
        print("✓ HUXt run is valid and complete!")
    else:
        print("✗ Issues found - see details above")
    
    # Plot single rotation
    plot_huxt_ensemble(huxt_run_id=1, n_members_to_plot=20)
    
    # Plot multiple rotations
    plot_full_huxt_timeseries(huxt_run_id=1, max_rotations=5)
    
    # Detect and save discontinuities
    discontinuities = detect_all_discontinuities(huxt_run_id=1, threshold=50, save=True)