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
from statsmodels.tsa.stattools import pacf

from storm_utils.config_paths import get_project_paths, add_huxt_paths

# Get required paths
add_huxt_paths()
paths = get_project_paths()

import huxt as H
import huxt_analysis as HA
import huxt_inputs as Hin
import huxt_ensembles as HE

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

    savename = f'cmap_cr_{cr}.png'
    outfolder = paths['storm_utils_figures'] / 'carrington_maps'
    outpath = outfolder / savename
    os.makedirs(outfolder, exist_ok=True)
    plt.savefig(outpath, bbox_inches='tight')
    plt.show()


def plot_carrington_map_and_extractions(cr, Nens, savename=None):
    fontsize = 14
    
    vr_map, vr_longs, vr_lats = Hin.get_MAS_vr_map(cr)

    # Convert to degrees
    vr_longs_deg = vr_longs.to(u.deg).value
    vr_lats_deg = vr_lats.to(u.deg).value

    lat_rot_sigma = 7.5  # degrees
    lat_dev_sigma = 2    # degrees
    long_dev_sigma = 2   # degrees

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

    savename = f'{Nens}_sinusoids_extraction_cr_{cr}.png'
    outfolder = paths['storm_utils_figures'] / 'carrington_maps'
    outpath = outfolder / savename
    os.makedirs(outfolder, exist_ok=True)
    plt.savefig(outpath, bbox_inches='tight')
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
    
    if savename is not None:
        outpath = paths['storm_utils_figures'] / 'data_plots' / savename
        plt.savefig(outpath, bbox_inches='tight')


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

    if savename is not None:
        outpath = paths['storm_utils_figures'] / 'data_plots' / savename
        plt.savefig(outpath, bbox_inches='tight')

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
    fig, ax = plt.subplots(1, figsize=(8, 5))

    ax.axhline(0, color='black', linewidth=1)
    ax.bar(lags, pacf_vals, width=1, color=colors, edgecolor='k')
    N = len(Hp30)
    ci = 1.96 / np.sqrt(N)
    
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

    if savename is not None:
        outpath = paths['storm_utils_figures'] / 'data_plots' / f'zoomed_{savename}'
        plt.savefig(outpath, bbox_inches='tight')

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

    if savename is not None:
        outpath = paths['storm_utils_figures'] / 'data_plots' / savename
        plt.savefig(outpath, bbox_inches='tight')


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
    
    if savename is not None:
        outpath = paths['storm_utils_figures'] / 'data_plots' / savename
        plt.savefig(outpath, bbox_inches='tight')

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

    if savename is not None:
        outpath = paths['storm_utils_figures'] / 'data_plots' / savename
        plt.savefig(outpath, bbox_inches='tight')


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

    if savename is not None:
        outpath = paths['storm_utils_figures'] / 'data_plots' / savename
        plt.savefig(outpath, bbox_inches='tight')

    plt.show()

    



    


