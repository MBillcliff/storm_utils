import datetime
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import requests
from io import StringIO
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.dates import DateFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from statsmodels.tsa.stattools import pacf

from storm_utils.config_paths import get_project_paths, add_huxt_paths
import logging

logger = logging.getLogger(__name__)

# Get required paths
add_huxt_paths()
paths = get_project_paths()

import huxt as H
import huxt_analysis as HA
import huxt_inputs as Hin
import huxt_ensembles as HE

# ============================================================================
# Utility Functions
# ============================================================================

def save_figure(fig_name, subfolder='exploratory', huxt_id=None, dpi=150):
    """
    Save figure with organized structure.
    
    Parameters
    ----------
    fig_name : str
        Descriptive figure name
    subfolder : str
        Purpose-based subfolder: 'model_comparison', 'case_studies', 
        'distribution_analysis', 'thesis', 'exploratory'
    huxt_id : int, optional
        If provided, includes HUXt ID in filename
    dpi : int
        Resolution for saved figure
    
    Returns
    -------
    Path
        Path to saved figure
    """
    paths = get_project_paths()
    figure_dir = paths['regression_figures'] / subfolder
    figure_dir.mkdir(parents=True, exist_ok=True)
    
    # Add HUXt ID to filename if specified
    if huxt_id is not None:
        fig_name = f'huxt{huxt_id}_{fig_name}'
    
    # Ensure .png extension
    if not fig_name.endswith('.png'):
        fig_name += '.png'
    
    save_path = figure_dir / fig_name
    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    logger.info(f"Saved figure: {save_path}")
    
    return save_path

# ============================================================================
# Plotting Functions
# ============================================================================


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime
import os
from matplotlib.lines import Line2D
    

def plot_carrington_map(cr, savename=None):
    fontsize = 14
    
    vr_map, vr_longs, vr_lats = Hin.get_MAS_vr_map(cr)

    # Convert to degrees
    vr_longs_deg = vr_longs.to(u.deg).value
    vr_lats_deg = vr_lats.to(u.deg).value

    lat_rot_sigma = 7.5  # degrees
    lat_dev_sigma = 2    # degrees
    long_dev_sigma = 2   # degrees

    x = vr_longs_deg

    fig, ax = plt.subplots(1, figsize=(8, 4))
    ax_contour = ax.contourf(vr_longs_deg, vr_lats_deg, vr_map.value, cmap='inferno')

    cbar = plt.colorbar(ax_contour, ax=ax, location='top', pad=0.0)
    cbar.set_label('Velocity (km/s)', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    ax.set_xlabel('Carrington Longitude (deg)', fontsize=fontsize)
    ax.set_ylabel('Latitude (deg)', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)

    ax.text(
        0.01, 0.93,               # X and Y in axes coords (close to top-left)
        f'Carrington Map for Rotation {cr}',   # The text
        transform=ax.transAxes,   # Use axes coordinates
        fontsize=fontsize,
        verticalalignment='top',  # Align top of text to (0.99)
        horizontalalignment='left'  # Align left of text to (0.01)
    )

    if savename:
        save_figure(savename, subfolder='data_exploration', huxt_id=cr)
    
    plt.show()


def plot_carrington_map_and_extractions(cr, Nens, lat_rot_sigma=7.5, long_dev_sigma=2, savename=None,):
    fontsize = 14
    
    vr_map, vr_longs, vr_lats = Hin.get_MAS_vr_map(cr)

    # Convert to degrees
    vr_longs_deg = vr_longs.to(u.deg).value
    vr_lats_deg = vr_lats.to(u.deg).value

    lat_dev_sigma = 2    # degrees

    x = vr_longs_deg

    fig, (upper, lower) = plt.subplots(2, sharex=True, figsize=(8, 6), gridspec_kw={'hspace': 0})
    upper_contour = upper.contourf(vr_longs_deg, vr_lats_deg, vr_map.value, cmap='inferno')

    cbar = plt.colorbar(upper_contour, ax=[upper, lower], location='top', pad=0.0)
    cbar.set_label('Velocity (km/s)', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    r_in = 30 * u.solRad
    np.random.seed(151201)

    # Generate random perturbations in degrees
    lat_rots = np.random.normal(0.0, lat_rot_sigma, Nens)
    long_rots = np.random.random_sample(Nens) * 360  # degrees
    lat_devs = np.random.normal(0.0, lat_dev_sigma, Nens)
    long_devs = np.random.normal(0.0, long_dev_sigma, Nens)

    phi, theta = np.meshgrid(vr_longs_deg, vr_lats_deg, indexing='xy')

    dummymodel = H.HUXt(v_boundary=np.ones((128)) * 400 * (u.km / u.s), simtime=27.27 * u.day,
                        cr_num=cr, lon_out=0.0 * u.deg, r_min=r_in)

    vr_ensemble = np.ones((Nens, len(vr_longs_deg)))
    earth = dummymodel.get_observer('earth')

    reflats = np.interp(vr_longs_deg, np.flipud(earth.lon_c.to(u.deg).value), np.flipud(earth.lat_c.to(u.deg).value))
    lats_E = reflats

    for i in range(Nens):
        this_lat = lats_E + lat_rots[i] * np.sin(np.deg2rad(vr_longs_deg + long_rots[i])) + lat_devs[i]
        this_long = (vr_longs_deg + long_devs[i]) % 360
        order = np.argsort(this_long)

        v = HE.interp2d(this_long[order], this_lat[order], vr_map, phi, theta)
        vr_ensemble[i, :] = v

        upper_label = 'perturbation' if i == 0 else None
        lower_label = 'velocity' if i == 0 else None
        f = lat_rots[i] * np.sin(np.deg2rad(vr_longs_deg + long_rots[i])) + lat_devs[i]

        upper.plot(x, f, color='white', alpha=0.6, label=upper_label)
        lower.plot(x, v, 'b', alpha=0.1, label=lower_label)

    lower.set_xlabel('Carrington Longitude (deg)', fontsize=fontsize)
    upper.set_ylabel('Latitude (deg)', fontsize=fontsize)
    lower.set_ylabel('Velocity (km/s)', fontsize=fontsize)

    upper.tick_params(labelsize=fontsize)
    lower.tick_params(labelsize=fontsize)

    upper.text(
        0.01, 0.93,               # X and Y in axes coords (close to top-left)
        f'Carrington Map for Rotation {cr}',   # The text
        transform=upper.transAxes,   # Use axes coordinates
        fontsize=fontsize,
        verticalalignment='top',  # Align top of text to (0.99)
        horizontalalignment='left'  # Align left of text to (0.01)
    )

    leg = upper.legend(facecolor='whitesmoke', edgecolor='gray', framealpha=0.3, fontsize=fontsize)
    for line in leg.get_lines():
        line.set_alpha(1.0)
    leg = lower.legend(facecolor='whitesmoke', edgecolor='gray', framealpha=0.3, loc='lower right', fontsize=fontsize)
    for line in leg.get_lines():
        line.set_alpha(1.0)

    if savename:
        save_figure(savename, subfolder='data_exploration', huxt_id=cr)
        
    plt.show()


def hp30_autocorrelation_plot(start_day, end_day, savename=None):
    fontsize = 14
    loadpath = paths['data_shared'] / 'hp30.parquet'

    Hp30 = pd.read_parquet(loadpath).to_numpy().squeeze()
    
    # Hp30: 1D numpy array of 30-minute sampled data
    sampling_per_day = 48  # 30-minute resolution
    lags_days = np.arange(start_day, end_day, 1/48)
    lags_samples = lags_days * sampling_per_day
    
    autocorrs = []
    
    for lag in lags_samples:
        lag_floor = int(np.floor(lag))
        frac = lag - lag_floor
    
        # Interpolate the lagged series to allow sub-sample lagging
        shifted = Hp30[lag_floor + 1:] * frac + Hp30[lag_floor:-1] * (1 - frac)
        base = Hp30[:len(shifted)]
    
        # Compute Pearson correlation
        corr = np.corrcoef(base, shifted)[0, 1]
        autocorrs.append(corr)

    # Find best lag
    best_index = np.argmax(autocorrs)
    best_lag = lags_days[best_index]
    best_corr = autocorrs[best_index]
    
    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(lags_days, autocorrs, marker='o')
    plt.title("Autocorrelation of Hp30 vs. Lag (30-min resolution)", fontsize=fontsize)
    plt.xlabel("Lag (days)", fontsize=fontsize)
    plt.ylabel("Pearson Correlation (r)", fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    plt.grid(True)

    # Annotate best lag
    right_shift = (end_day - start_day) / 10
    down_shift = best_corr / 17
    
    plt.axvline(best_lag, color='gray', linestyle='--', alpha=0.7)
    plt.plot(best_lag, best_corr, 'ro', label='Max autocorrelation')
    plt.text(best_lag + right_shift, best_corr - down_shift, f'{best_lag:.2f} days\nr={best_corr:.2f}',
             color='red', ha='center', va='bottom', fontsize=10)
    
    if savename:
        save_figure(savename, subfolder='data_exploration')

    plt.show()


def hp30_autocorrelation_heatmap(lag_start, lag_end, savename=None):
    fontsize = 14
    
    loadpath = paths['data_shared'] / 'hp30.parquet'
    Hp30 = pd.read_parquet(loadpath).to_numpy().squeeze()

    sampling_per_day = 48  # 30-minute resolution
    days_per_year = 365.25
    steps_per_year = int(sampling_per_day * days_per_year)

    # Use whole-day lags only
    lags_days = np.arange(lag_start, lag_end + 0.001, 0.5)
    lags_samples = lags_days * sampling_per_day

    # Number of full years available
    n_years = len(Hp30) // steps_per_year
    print(f"Number of full years in data: {n_years}")

    # Matrix: rows = day lags, cols = years
    corr_matrix = np.zeros((len(lags_days), n_years))

    for y in range(n_years):
        segment = Hp30[y * steps_per_year:(y + 1) * steps_per_year]

        for i, lag in enumerate(lags_samples):
            lag = int(lag)

            if lag >= len(segment):
                corr_matrix[i, y] = np.nan
                continue

            base = segment[:len(segment) - lag]
            shifted = segment[lag:]

            corr = np.corrcoef(base, shifted)[0, 1]
            corr_matrix[i, y] = corr

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8, 4))

    # Make fiery color map and fix color limits from -1 to 1
    c = ax.imshow(
        corr_matrix,
        aspect='auto',
        origin='lower',
        extent=[1995, 1995 + n_years, lag_start, lag_end + 1],
        cmap='hot',
        vmin=0,
        vmax=np.max(corr_matrix)-0.1,
    )

    ax.set_xlabel("Year", fontsize=fontsize)
    ax.set_ylabel("Lag (days)", fontsize=fontsize)
    ax.set_title("Hp30 Autocorrelation Heatmap by Year", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)

    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Pearson Correlation", fontsize=fontsize)
    plt.tight_layout()

    if savename:
        save_figure(savename, subfolder='data_exploration')

    plt.show()

def zoomed_hp30_partial_autocorrelation_plot(nlags, start_lag=0, zoom=1, savename=None, show_days=False):
    loadpath = paths['data_shared'] / 'hp30.parquet'

    Hp30 = pd.read_parquet(loadpath).to_numpy().squeeze()

    # Compute PACF with 95% CI
    pacf_vals, confint = pacf(Hp30, nlags=nlags, alpha=0.05)
    lags = np.arange(len(pacf_vals))

    # Determine significance
    significant = (confint[:, 0] > 0) | (confint[:, 1] < 0)
    colors = np.where(significant, 'blue', 'red')

    # Slice to start_lag onward
    lags = lags[start_lag:]
    pacf_vals = pacf_vals[start_lag:]
    colors = colors[start_lag:]

    fontsize = 15

    # Legend elements
    legend_elements = [
        Patch(facecolor='blue', edgecolor='k', label='Significant (95%)'),
        Patch(facecolor='red', edgecolor='k', label='Not significant'),
        Patch(facecolor='lightgray', edgecolor='k', linestyle='--', label='95% CI'),
    ]

    N = len(Hp30)
    ci = 1.96 / np.sqrt(N)
    significant = np.abs(pacf_vals) > ci
    colors = np.where(significant, 'blue', 'red')

    # Create subplot layout
    fig, ax = plt.subplots(1, figsize=(8, 5))

    ax.axhline(0, color='black', linewidth=1)
    ax.bar(lags, pacf_vals, width=1, color=colors, edgecolor='k')
    
    ax.axhline(ci, linestyle='--', color='k', lw=2)
    ax.axhline(-ci, linestyle='--', color='k', lw=2)
    ax.fill_between(
        x=range(start_lag, lags[-1] + 1),
        y1=ci,
        y2=-ci,
        color='gray',
        alpha=0.2,
        label='95% CI'
    )
    
    ax.legend(handles=legend_elements)
    ax.grid(True)
    ax.set_xlim(start_lag, lags[-1])

    ax.set_ylim(-zoom / 2, zoom)
    ax.set_xlabel("Lag", fontsize=fontsize)
    ax.set_ylabel("Partial Autocorrelation", fontsize=fontsize)
    ax.set_title("Zoomed PACF of Hp30", fontsize=fontsize+1)

    # Choose tick locations in timesteps (e.g. every 48 steps = 1 day)
    max_lag = lags[-1]
    tick_spacing = 24  # every 0.5 days
    tick_locs = np.arange(start=start_lag, stop=max_lag + 1, step=tick_spacing)
    
    # Set tick locations and labels (converted to days)
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([f"{tick / 48:.1f}" for tick in tick_locs])
    ax.set_xlabel("Lag (days)", fontsize=fontsize)

    plt.tight_layout()

    if savename:
        save_figure(savename, subfolder='data_exploration')
        
    plt.show()

def hp30_partial_autocorrelation_plot(nlags, start_lag=0, zoom=1, savename=None, show_days=False):
    loadpath = paths['data_shared'] / 'hp30.parquet'

    Hp30 = pd.read_parquet(loadpath).to_numpy().squeeze()

    # Compute PACF with 95% CI
    pacf_vals, confint = pacf(Hp30, nlags=nlags, alpha=0.05)
    lags = np.arange(len(pacf_vals))

    # Determine significance
    significant = (confint[:, 0] > 0) | (confint[:, 1] < 0)
    colors = np.where(significant, 'blue', 'red')

    # Slice to start_lag onward
    lags = lags[start_lag:]
    pacf_vals = pacf_vals[start_lag:]
    confint = confint[start_lag:]
    colors = colors[start_lag:]

    fontsize = 15

    # Legend elements
    legend_elements = [
        Patch(facecolor='blue', edgecolor='k', label='Significant (95%)'),
        Patch(facecolor='red', edgecolor='k', label='Not significant'),
        Patch(facecolor='lightgray', edgecolor='k', linestyle='--', label='95% CI'),
    ]

    # Create subplot layout
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for i in range(2):
        ax[i].axhline(0, color='black', linewidth=1)
        ax[i].bar(lags, pacf_vals, width=1, color=colors, edgecolor='k')
        N = len(Hp30)
        ci = 1.96 / np.sqrt(N)
        
        ax[i].axhline(ci, linestyle='--', color='k', lw=2)
        ax[i].axhline(-ci, linestyle='--', color='k', lw=2)
        ax[i].fill_between(
            x=range(start_lag, lags[-1] + 1),
            y1=ci,
            y2=-ci,
            color='gray',
            alpha=0.2,
            label='95% CI'
        )
        ax[i].legend(handles=legend_elements)
        ax[i].grid(True)
        ax[i].set_xlim(start_lag, lags[-1])

    

    ax[0].set_title("PACF of Hp30", fontsize=fontsize+1)
    ax[0].set_ylabel("Partial Autocorrelation", fontsize=fontsize)

    ax[1].set_ylim(-zoom / 2, zoom)
    ax[1].set_xlabel("Lag", fontsize=fontsize)
    ax[1].set_ylabel("Partial Autocorrelation", fontsize=fontsize)
    ax[1].set_title("Zoomed PACF of Hp30", fontsize=fontsize+1)

    # Choose tick locations in timesteps (e.g. every 48 steps = 1 day)
    max_lag = lags[-1]
    tick_spacing = 24  # every 0.5 days
    tick_locs = np.arange(start=start_lag, stop=max_lag + 1, step=tick_spacing)
    
    # Set tick locations and labels (converted to days)
    ax[1].set_xticks(tick_locs)
    ax[1].set_xticklabels([f"{tick / 48:.1f}" for tick in tick_locs])
    ax[1].set_xlabel("Lag (days)", fontsize=fontsize)


    plt.tight_layout()

    if savename:
        save_figure(savename, subfolder='data_exploration')

    plt.show()


def huxt_output_autocorrelation_plot(vi=0, start_day=0, end_day=3, savename=None):
    huxt_run_id = 1
    loadpath = paths['huxt_data_shared'] / f'HUXt{huxt_run_id}_modified' / 'full_df.parquet'
    
    v = pd.read_parquet(loadpath, columns=[f'v_{vi}']).to_numpy().squeeze()

    fontsize = 14
    
    # v: 1D numpy array of 30-minute sampled data
    sampling_per_day = 48  # 30-minute resolution
    lags_days = np.arange(start_day, end_day, 1/48)
    lags_samples = lags_days * sampling_per_day
    
    autocorrs = []
    
    for lag in lags_samples:
        lag_floor = int(np.floor(lag))
        frac = lag - lag_floor
    
        # Interpolate the lagged series to allow sub-sample lagging
        shifted = v[lag_floor + 1:] * frac + v[lag_floor:-1] * (1 - frac)
        base = v[:len(shifted)]
    
        # Compute Pearson correlation
        corr = np.corrcoef(base, shifted)[0, 1]
        autocorrs.append(corr)

    # Find best lag
    best_index = np.argmax(autocorrs)
    best_lag = lags_days[best_index]
    best_corr = autocorrs[best_index]
    
    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(lags_days, autocorrs, marker='o')
    plt.title("Autocorrelation of HUXt Vx vs. Lag (30-min resolution)", fontsize=fontsize)
    plt.xlabel("Lag (days)", fontsize=fontsize)
    plt.ylabel("Pearson Correlation (r)", fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    plt.grid(True)

    # Annotate best lag
    right_shift = (end_day - start_day) / 10
    down_shift = best_corr / 17
    
    plt.axvline(best_lag, color='gray', linestyle='--', alpha=0.7)
    plt.plot(best_lag, best_corr, 'ro', label='Max autocorrelation')
    plt.text(best_lag + right_shift, best_corr - down_shift, f'{best_lag:.2f} days\nr={best_corr:.2f}',
             color='red', ha='center', va='bottom', fontsize=10)
    
    if savename:
        save_figure(savename, subfolder='data_exploration', huxt_id=huxt_run_id)

    plt.show()


def huxt_output_partial_autocorrelation_plot(nlags, vi, start_lag=0, zoom=1, savename=None, show_days=False):
    huxt_run_id = 1
    loadpath = paths['huxt_data_shared'] / f'HUXt{huxt_run_id}_modified' / 'full_df.parquet'
    
    v = pd.read_parquet(loadpath, columns=[f'v_{vi}']).to_numpy().squeeze()


    # Compute PACF with 95% CI
    pacf_vals, confint = pacf(v, nlags=nlags, alpha=0.05)
    lags = np.arange(len(pacf_vals))

    # Determine significance
    significant = (confint[:, 0] > 0) | (confint[:, 1] < 0)
    colors = np.where(significant, 'blue', 'red')

    # Slice to start_lag onward
    lags = lags[start_lag:]
    pacf_vals = pacf_vals[start_lag:]
    confint = confint[start_lag:]
    colors = colors[start_lag:]

    fontsize = 15

    # Legend elements
    legend_elements = [
        Patch(facecolor='blue', edgecolor='k', label='Significant (95%)'),
        Patch(facecolor='red', edgecolor='k', label='Not significant'),
        Patch(facecolor='lightgray', edgecolor='k', linestyle='--', label='95% CI'),
    ]

    # Create subplot layout
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for i in range(2):
        ax[i].axhline(0, color='black', linewidth=1)
        ax[i].bar(lags, pacf_vals, width=1, color=colors, edgecolor='k')
        N = len(v)
        ci = 1.96 / np.sqrt(N)
        
        ax[i].axhline(ci, linestyle='--', color='k', lw=2)
        ax[i].axhline(-ci, linestyle='--', color='k', lw=2)
        ax[i].fill_between(
            x=range(start_lag, lags[-1] + 1),
            y1=ci,
            y2=-ci,
            color='gray',
            alpha=0.2,
            label='95% CI'
        )
        ax[i].legend(handles=legend_elements)
        ax[i].grid(True)
        ax[i].set_xlim(start_lag, lags[-1])

    

    ax[0].set_title("PACF of HUXt Vx", fontsize=fontsize+1)
    ax[0].set_ylabel("Partial Autocorrelation", fontsize=fontsize)

    ax[1].set_ylim(-zoom / 2, zoom)
    ax[1].set_xlabel("Lag", fontsize=fontsize)
    ax[1].set_ylabel("Partial Autocorrelation", fontsize=fontsize)
    ax[1].set_title("Zoomed PACF of HUXt Vx", fontsize=fontsize+1)

    # Choose tick locations in timesteps (e.g. every 48 steps = 1 day)
    max_lag = lags[-1]
    tick_spacing = 24  # every 0.5 days
    tick_locs = np.arange(start=start_lag, stop=max_lag + 1, step=tick_spacing)
    
    # Set tick locations and labels (converted to days)
    ax[1].set_xticks(tick_locs)
    ax[1].set_xticklabels([f"{tick / 48:.1f}" for tick in tick_locs])
    ax[1].set_xlabel("Lag (days)", fontsize=fontsize)


    plt.tight_layout()

    if savename:
        save_figure(savename, subfolder='data_exploration', huxt_id=huxt_run_id)

    plt.show()


def huxt_output_autocorrelation_heatmap(lag_start, lag_end, vi=0, savename=None):
    fontsize = 14
    
    huxt_run_id = 1
    loadpath = paths['huxt_data_shared'] / f'HUXt{huxt_run_id}_modified' / 'full_df.parquet'
    
    v = pd.read_parquet(loadpath, columns=[f'v_{vi}']).to_numpy().squeeze()

    sampling_per_day = 48  # 30-minute resolution
    days_per_year = 365.25
    steps_per_year = int(sampling_per_day * days_per_year)

    # Use whole-day lags only
    lags_days = np.arange(lag_start, lag_end + 0.001, 0.5)
    lags_samples = lags_days * sampling_per_day

    # Number of full years available
    n_years = len(v) // steps_per_year
    print(f"Number of full years in data: {n_years}")

    # Matrix: rows = day lags, cols = years
    corr_matrix = np.zeros((len(lags_days), n_years))

    for y in range(n_years):
        segment = v[y * steps_per_year:(y + 1) * steps_per_year]

        for i, lag in enumerate(lags_samples):
            lag = int(lag)

            if lag >= len(segment):
                corr_matrix[i, y] = np.nan
                continue

            base = segment[:len(segment) - lag]
            shifted = segment[lag:]

            corr = np.corrcoef(base, shifted)[0, 1]
            corr_matrix[i, y] = corr

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8, 4))

    # Make fiery color map and fix color limits from -1 to 1
    c = ax.imshow(
        corr_matrix,
        aspect='auto',
        origin='lower',
        extent=[1995, 1995 + n_years, lag_start, lag_end + 1],
        cmap='hot',
        vmin=0,
        vmax=np.max(corr_matrix)-0.1,
    )

    ax.set_xlabel("Year", fontsize=fontsize)
    ax.set_ylabel("Lag (days)", fontsize=fontsize)
    ax.set_title("HUXt Vx Autocorrelation Heatmap by Year", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)

    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label("Pearson Correlation", fontsize=fontsize)
    plt.tight_layout()

    if savename:
        save_figure(savename, subfolder='data_exploration', huxt_id=huxt_run_id)

    plt.show()

    
def plot_icme_timeline(
    icme_df,
    start_date=None,
    end_date=None,
    color_by='Dst',
    figsize=(14, 8),
    show_mc_only=False,
    show_mc_markers=True, 
    save=False,
    save_name=None,
):
    """
    Plot timeline of ICMEs over a specified date range.
    
    Parameters
    ----------
    icme_df : pd.DataFrame
        ICME catalog dataframe
    start_date : str or datetime, optional
        Start date for plot. If None, uses earliest ICME
    end_date : str or datetime, optional
        End date for plot. If None, uses latest ICME
    color_by : str
        Column to use for coloring ICMEs. Options:
        - 'Dst': Color by storm intensity (default)
        - 'V_max': Color by velocity
        - 'B': Color by magnetic field strength
        - 'MC': Color by whether it's a magnetic cloud
    figsize : tuple
        Figure size (width, height)
    show_mc_only : bool
        If True, only plot magnetic clouds
    show_mc_markers : bool
        If True, show gold stars and special styling for magnetic clouds
        If False, treat all ICMEs the same
    
    Examples
    --------
    >>> # With MC markers
    >>> plot_icme_timeline(icme_df, start_date='2000-01-01', end_date='2005-12-31')
    
    >>> # Without MC markers (clean view)
    >>> plot_icme_timeline(icme_df, start_date='2000-01-01', end_date='2005-12-31', 
    ...                   show_mc_markers=False)
    
    >>> # Only MCs
    >>> plot_icme_timeline(icme_df, show_mc_only=True, color_by='B')
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.dates import DateFormatter, MonthLocator, YearLocator
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    import pandas as pd
    
    # Filter by date range
    df_plot = icme_df.copy()
    
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df_plot = df_plot[df_plot['ICME_Plasma_Field_End'] >= start_date]
    else:
        start_date = df_plot['ICME_Plasma_Field_Start'].min()
    
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df_plot = df_plot[df_plot['ICME_Plasma_Field_Start'] <= end_date]
    else:
        end_date = df_plot['ICME_Plasma_Field_End'].max()
    
    # Filter to magnetic clouds only if requested
    if show_mc_only:
        df_plot = df_plot[df_plot['MC'] == 1]
        title_suffix = ' (Magnetic Clouds Only)'
    else:
        title_suffix = ''
    
    # Remove rows with missing dates
    df_plot = df_plot.dropna(subset=['ICME_Plasma_Field_Start', 'ICME_Plasma_Field_End'])
    
    print(f"Plotting {len(df_plot)} ICMEs from {start_date.date()} to {end_date.date()}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up coloring
    if color_by == 'Dst':
        norm = Normalize(vmin=-300, vmax=0)
        cmap = plt.cm.Reds_r
        color_label = 'Dst (nT)'
        use_colorbar = True
    elif color_by == 'V_max':
        norm = Normalize(vmin=300, vmax=900)
        cmap = plt.cm.viridis
        color_label = 'V_max (km/s)'
        use_colorbar = True
    elif color_by == 'B':
        norm = Normalize(vmin=0, vmax=50)
        cmap = plt.cm.plasma
        color_label = 'B (nT)'
        use_colorbar = True
    elif color_by == 'MC':
        # Binary coloring - only makes sense with show_mc_markers=True
        use_colorbar = False
        show_mc_markers = True  # Force MC markers when coloring by MC
    else:
        norm = Normalize()
        cmap = plt.cm.viridis
        color_label = color_by
        use_colorbar = True
    
    # Plot each ICME
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        start = row['ICME_Plasma_Field_Start']
        end = row['ICME_Plasma_Field_End']
        duration = end - start  # Timedelta object
        
        # Check if this is a magnetic cloud
        is_mc = show_mc_markers and pd.notna(row['MC']) and row['MC'] == 1
        
        # Get color
        if color_by == 'MC':
            color = 'gold' if is_mc else 'lightgray'
            edge_color = 'black' if is_mc else 'gray'
            edge_width = 1.5 if is_mc else 0.5
            alpha = 0.8
        else:
            color_val = row[color_by]
            if pd.notna(color_val):
                color = cmap(norm(color_val))
            else:
                color = 'lightgray'
            
            # Different edge styling for MCs if show_mc_markers is True
            if is_mc:
                edge_color = 'gold'
                edge_width = 2.0
            else:
                edge_color = 'black'
                edge_width = 0.5
            
            alpha = 0.7
        
        # Draw bar
        ax.barh(i, duration, left=start, height=0.8, 
               color=color, alpha=alpha, edgecolor=edge_color, linewidth=edge_width)
        
        # Mark magnetic clouds with a star (only if show_mc_markers=True)
        if is_mc:
            ax.scatter(start + duration / 2, i, marker='*', s=100, 
                      color='gold', edgecolors='black', linewidth=0.5, zorder=10)
    
    # Add colorbar
    if use_colorbar:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label=color_label, pad=0.01)
    
    # Add legend
    if show_mc_markers:
        if color_by == 'MC':
            # MC-specific legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='gold', edgecolor='black', linewidth=1.5, label='Magnetic Cloud'),
                Patch(facecolor='lightgray', edgecolor='gray', linewidth=0.5, label='ICME (no MC)'),
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                          markersize=12, markeredgecolor='black', label='MC marker')
            ]
        else:
            # General legend with MC indicator
            legend_elements = [
                mpatches.Patch(facecolor='none', edgecolor='gold', linewidth=2, label='Magnetic Cloud (gold edge)'),
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                          markersize=12, markeredgecolor='black', label='MC marker')
            ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Format x-axis
    ax.set_xlim(start_date, end_date)
    
    # Adjust date formatting based on time range
    time_span = (end_date - start_date).days
    if time_span > 1825:  # > 5 years
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    else:
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax.set_xlabel('Date', fontsize=13)
    ax.set_ylabel('ICME Event Number', fontsize=13)
    
    mc_text = ' (ignoring MC status)' if not show_mc_markers else ''
    ax.set_title(f'ICME Timeline{title_suffix} (colored by {color_by}){mc_text}\n{start_date.date()} to {end_date.date()}', 
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_ylim(-0.5, len(df_plot) - 0.5)
    
    # Invert y-axis so first event is at top
    ax.invert_yaxis()
    
    plt.tight_layout()

    if save and save_name:
        save_figure(save_name, subfolder='icme_analysis')
    
    plt.show()
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"ICME Statistics")
    print(f"{'='*60}")
    print(f"Total ICMEs plotted: {len(df_plot)}")
    if 'MC' in df_plot.columns and show_mc_markers:
        mc_count = df_plot['MC'].sum()
        print(f"Magnetic Clouds: {mc_count} ({mc_count/len(df_plot)*100:.1f}%)")
    if 'Dst' in df_plot.columns:
        print(f"\nDst range: {df_plot['Dst'].min():.0f} to {df_plot['Dst'].max():.0f} nT")
        print(f"Mean Dst: {df_plot['Dst'].mean():.1f} nT")
        print(f"Strongest storms (Dst < -100): {len(df_plot[df_plot['Dst'] < -100])}")
    print(f"{'='*60}\n")


def plot_icme_properties_timeline(
    icme_df,
    start_date=None,
    end_date=None,
    figsize=(14, 10),
    show_mc_shading=True,
    save=False,
    save_name=None,
):
    """
    Plot ICME properties (Dst, V, B) over time with ICME durations shown.
    
    Parameters
    ----------
    icme_df : pd.DataFrame
        ICME catalog dataframe
    start_date : str or datetime, optional
        Start date for plot
    end_date : str or datetime, optional
        End date for plot
    figsize : tuple
        Figure size (width, height)
    show_mc_shading : bool
        If True, shade MC durations differently than regular ICMEs
        If False, all ICMEs shaded the same
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.dates import DateFormatter, YearLocator
    import pandas as pd
    
    # Filter by date range
    df_plot = icme_df.copy()
    
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df_plot = df_plot[df_plot['ICME_Plasma_Field_End'] >= start_date]
    else:
        start_date = df_plot['ICME_Plasma_Field_Start'].min()
    
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df_plot = df_plot[df_plot['ICME_Plasma_Field_Start'] <= end_date]
    else:
        end_date = df_plot['ICME_Plasma_Field_End'].max()
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    ax_dst, ax_v, ax_b = axes
    
    # Plot ICME durations as shaded regions on all panels
    for idx, row in df_plot.iterrows():
        start = row['ICME_Plasma_Field_Start']
        end = row['ICME_Plasma_Field_End']
        
        if pd.isna(start) or pd.isna(end):
            continue
        
        # Determine shading based on show_mc_shading
        if show_mc_shading:
            is_mc = pd.notna(row['MC']) and row['MC'] == 1
            alpha = 0.3 if is_mc else 0.15
            color = 'blue' if is_mc else 'gray'
        else:
            # All ICMEs shaded the same
            alpha = 0.2
            color = 'gray'
        
        for ax in axes:
            ax.axvspan(start, end, alpha=alpha, color=color, zorder=1)
    
    # Panel 1: Dst
    dst_data = df_plot[df_plot['Disturbance_Date'].notna() & df_plot['Dst'].notna()]
    disturbance_times = dst_data['Disturbance_Date']
    dst_values = dst_data['Dst']
    
    ax_dst.scatter(disturbance_times, dst_values, c=dst_values, cmap='Reds_r', 
                  s=50, edgecolors='black', linewidth=0.5, vmin=-300, vmax=0, zorder=10)
    ax_dst.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax_dst.axhline(-50, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Moderate storm')
    ax_dst.axhline(-100, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Intense storm')
    ax_dst.set_ylabel('Dst (nT)', fontsize=12)
    ax_dst.grid(True, alpha=0.3, zorder=0)
    ax_dst.set_ylim(-350, 50)
    
    # Panel 2: Velocity
    v_data = df_plot[df_plot['Disturbance_Date'].notna()]
    v_max_valid = v_data[v_data['V_max'].notna()]
    v_icme_valid = v_data[v_data['V_ICME'].notna()]
    
    ax_v.scatter(v_max_valid['Disturbance_Date'], v_max_valid['V_max'], 
                c='steelblue', s=50, edgecolors='black', linewidth=0.5, alpha=0.7, label='V_max', zorder=10)
    ax_v.scatter(v_icme_valid['Disturbance_Date'], v_icme_valid['V_ICME'], 
                c='lightblue', s=30, edgecolors='black', linewidth=0.5, alpha=0.5, label='V_ICME', zorder=10)
    ax_v.set_ylabel('Velocity (km/s)', fontsize=12)
    ax_v.legend(loc='upper right', fontsize=9)
    ax_v.grid(True, alpha=0.3, zorder=0)
    
    # Panel 3: Magnetic Field
    b_data = df_plot[df_plot['Disturbance_Date'].notna() & df_plot['B'].notna()]
    
    ax_b.scatter(b_data['Disturbance_Date'], b_data['B'], 
                c='purple', s=50, edgecolors='black', linewidth=0.5, alpha=0.7, zorder=10)
    ax_b.set_ylabel('B (nT)', fontsize=12)
    ax_b.set_xlabel('Date', fontsize=12)
    ax_b.grid(True, alpha=0.3, zorder=0)
    
    # Format x-axis
    for ax in axes:
        ax.set_xlim(start_date, end_date)
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    
    # Title
    mc_text = ' (ignoring MC status)' if not show_mc_shading else ''
    fig.suptitle(f'ICME Properties Timeline{mc_text}\n{start_date.date()} to {end_date.date()}', 
                fontsize=15, fontweight='bold')
    
    # Legend for shaded regions
    if show_mc_shading:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.3, label='Magnetic Cloud duration'),
            Patch(facecolor='gray', alpha=0.15, label='ICME duration'),
            mpatches.Patch(color='orange', alpha=0, label=''),  # Spacer
            mpatches.Patch(facecolor='none', edgecolor='orange', linewidth=1, 
                          linestyle='--', label='Moderate storm (Dst=-50)'),
            mpatches.Patch(facecolor='none', edgecolor='red', linewidth=1, 
                          linestyle='--', label='Intense storm (Dst=-100)'),
        ]
        ax_dst.legend(handles=legend_elements, loc='lower left', fontsize=9, ncol=2)
    else:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gray', alpha=0.2, label='ICME duration'),
            mpatches.Patch(color='orange', alpha=0, label=''),  # Spacer
            mpatches.Patch(facecolor='none', edgecolor='orange', linewidth=1, 
                          linestyle='--', label='Moderate storm (Dst=-50)'),
            mpatches.Patch(facecolor='none', edgecolor='red', linewidth=1, 
                          linestyle='--', label='Intense storm (Dst=-100)'),
        ]
        ax_dst.legend(handles=legend_elements, loc='lower left', fontsize=9, ncol=2)
    
    plt.tight_layout()

    if save and save_name:
        save_figure(save_name, subfolder='icme_analysis')
        
    plt.show()


def plot_icme_bars(
    icme_df,
    start_date=None,
    end_date=None,
    color_by='Dst',
    figsize=(14, 8),
    show_mc_only=False,
    show_mc_markers=True  # NEW: Toggle MC features
):
    """
    Bar chart showing ICME durations as horizontal bars.
    
    Parameters
    ----------
    icme_df : pd.DataFrame
        ICME catalog dataframe
    start_date : str or datetime, optional
        Start date for plot
    end_date : str or datetime, optional
        End date for plot  
    figsize : tuple
        Figure size
    color_by : str
        Property to color bars by ('Dst', 'V_max', 'B', 'MC')
    show_mc_only : bool
        If True, only plot magnetic clouds
    show_mc_markers : bool
        If True, mark magnetic clouds with stars and gold edges
        If False, treat all ICMEs uniformly
    
    Examples
    --------
    >>> # Clean view without MC markers
    >>> plot_icme_bars(icme_df, start_date='2000-01-01', end_date='2005-12-31', 
    ...               color_by='Dst', show_mc_markers=False)
    
    >>> # With MC markers
    >>> plot_icme_bars(icme_df, start_date='2000-01-01', end_date='2005-12-31', 
    ...               color_by='V_max', show_mc_markers=True)
    """
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter, YearLocator, MonthLocator
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    import pandas as pd
    
    # Filter by date range
    df_plot = icme_df.copy()
    
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df_plot = df_plot[df_plot['ICME_Plasma_Field_End'] >= start_date]
    else:
        start_date = df_plot['ICME_Plasma_Field_Start'].min()
    
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df_plot = df_plot[df_plot['ICME_Plasma_Field_Start'] <= end_date]
    else:
        end_date = df_plot['ICME_Plasma_Field_End'].max()
    
    # Filter to magnetic clouds only if requested
    if show_mc_only:
        df_plot = df_plot[df_plot['MC'] == 1]
        title_suffix = ' (Magnetic Clouds Only)'
    else:
        title_suffix = ''
    
    # Remove rows with missing dates
    df_plot = df_plot.dropna(subset=['ICME_Plasma_Field_Start', 'ICME_Plasma_Field_End'])
    
    print(f"Plotting {len(df_plot)} ICMEs from {start_date.date()} to {end_date.date()}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up coloring
    if color_by == 'Dst':
        norm = Normalize(vmin=-300, vmax=0)
        cmap = plt.cm.Reds_r
        color_label = 'Dst (nT)'
        use_colorbar = True
    elif color_by == 'V_max':
        norm = Normalize(vmin=300, vmax=900)
        cmap = plt.cm.viridis
        color_label = 'V_max (km/s)'
        use_colorbar = True
    elif color_by == 'B':
        norm = Normalize(vmin=0, vmax=50)
        cmap = plt.cm.plasma
        color_label = 'B (nT)'
        use_colorbar = True
    elif color_by == 'MC':
        use_colorbar = False
        show_mc_markers = True  # Force MC markers when coloring by MC
    else:
        norm = Normalize()
        cmap = plt.cm.viridis
        color_label = color_by
        use_colorbar = True
    
    # Plot each ICME
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        start = row['ICME_Plasma_Field_Start']
        end = row['ICME_Plasma_Field_End']
        duration = end - start
        
        # Check if MC
        is_mc = show_mc_markers and pd.notna(row['MC']) and row['MC'] == 1
        
        # Get color
        if color_by == 'MC':
            color = 'gold' if is_mc else 'lightgray'
            edge_color = 'black' if is_mc else 'gray'
            edge_width = 1.5 if is_mc else 0.5
            alpha = 0.8
        else:
            color_val = row[color_by]
            if pd.notna(color_val):
                color = cmap(norm(color_val))
            else:
                color = 'lightgray'
            
            # Style edges differently for MCs if show_mc_markers=True
            if is_mc:
                edge_color = 'gold'
                edge_width = 2.0
            else:
                edge_color = 'black'
                edge_width = 0.5
            alpha = 0.7
        
        # Draw bar
        ax.barh(i, duration, left=start, height=0.8, 
               color=color, alpha=alpha, edgecolor=edge_color, linewidth=edge_width)
        
        # Mark magnetic clouds with a star
        if is_mc:
            ax.scatter(start + duration / 2, i, marker='*', s=100, 
                      color='gold', edgecolors='black', linewidth=0.5, zorder=10)
    
    # Add colorbar
    if use_colorbar:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label=color_label, pad=0.01)
    
    # Add legend
    if show_mc_markers:
        if color_by == 'MC':
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='gold', edgecolor='black', linewidth=1.5, label='Magnetic Cloud'),
                Patch(facecolor='lightgray', edgecolor='gray', linewidth=0.5, label='ICME (no MC)'),
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                          markersize=12, markeredgecolor='black', label='MC marker')
            ]
        else:
            legend_elements = [
                mpatches.Patch(facecolor='none', edgecolor='gold', linewidth=2, label='Magnetic Cloud (gold edge)'),
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                          markersize=12, markeredgecolor='black', label='MC marker')
            ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Format x-axis
    ax.set_xlim(start_date, end_date)
    
    # Adjust formatting
    time_span = (end_date - start_date).days
    if time_span > 1825:
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    else:
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax.set_xlabel('Date', fontsize=13)
    ax.set_ylabel('ICME Event Number', fontsize=13)
    
    mc_text = ' (ignoring MC status)' if not show_mc_markers else ''
    ax.set_title(f'ICME Timeline{title_suffix} (colored by {color_by}){mc_text}\n{start_date.date()} to {end_date.date()}', 
                fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x', zorder=0)
    ax.set_ylim(-0.5, len(df_plot) - 0.5)
    
    # Invert y-axis
    ax.invert_yaxis()
    
    plt.tight_layout()

    if save and save_name:
        save_figure(save_name, subfolder='icme_analysis')
    
    plt.show()


def compute_and_plot_histograms(y_train, y_test, save=False, savename=None):
    # Compute common bin edges
    max_y = max(y_train.max(), y_test.max())
    bin_edges = np.linspace(0, max_y, int(max_y * 3))

    train_counts, _ = np.histogram(y_train, bins=bin_edges)
    test_counts, _ = np.histogram(y_test, bins=bin_edges)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.hist(y_train, bins=bin_edges, alpha=0.5, label='Train')
    plt.hist(y_test, bins=bin_edges, alpha=0.5, label='Test')
    plt.xlabel('max_target')
    plt.ylabel('Count')
    plt.title('Distribution of max_target')
    plt.legend()
    plt.grid(True)

    if save and savename:
        save_figure(save_name, subfolder='data_exploration')
    plt.show()

    # Return structured histogram data
    return {
        'bin_edges': bin_edges,
        'train_counts': train_counts,
        'test_counts': test_counts
    }