# coding=utf-8
import pandas as pd
import os
import warnings
import math
import numpy as np
import seaborn as sns
import xarray as xr
import rasterio as rio
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
from datetime import datetime
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.stats import ttest_1samp
from typing import Dict, Tuple, List
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import matplotlib

matplotlib.use('agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter
from matplotlib.ticker import ScalarFormatter
from cartopy.io.shapereader import Reader
import cartopy.feature as cfeat

import dask

dask.config.set({'dataframe.query-planning': True})

export_figure_type = "pdf"
export_figure_dpi = 450

# Check if the folder exists
def create_path(path, is_dir=False):
    try:
        # Check if the path ends with a directory separator
        if path.endswith("\\") or path.endswith("/") or is_dir:
            # Path ends with a directory separator or explicitly stated to be a folder, assume it's a directory path
            directory_path = path
        else:
            # If the path does not end with a separator, it's assumed to be a file or directory, get the parent directory
            directory_path = os.path.dirname(path)
        # Check if the directory exists, if not, create it
        if not os.path.isdir(directory_path):
            os.makedirs(directory_path,
                        exist_ok=True)  # Use exist_ok=True to prevent an exception if the folder already exists
    except Exception as e:
        print(f"Failed to create folder: {e}")


# Mean Squared Error
def get_mse(records_real, records_predict):
    """
    Mean Squared Error: Measures the deviation between predicted and real values
    """
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None


# Root Mean Squared Error
def get_rmse(records_real, records_predict):
    """
    Root Mean Squared Error: The square root of Mean Squared Error
    """
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None


def draw_gis_figure(ax, top_labels=False, bottom_labels = False, left_labels = False, right_labels = False):
    boundary = [-180, 180, -60, 90]
    proj = ccrs.PlateCarree()  # 创建坐标系
    # 设置绘图范围
    ax.set_extent(boundary, crs=proj)
    # 添加海岸线
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"))
    # 添加经纬度标签
    gl =ax.gridlines(crs=proj, draw_labels=False, xlocs=[-120, -60, 0, 60, 120],
                 ylocs=[-30, 0, 30, 60], linewidth=0.05, color='k', alpha=0.5, linestyle='--', zorder=-1)
    gl.top_labels = top_labels  # 打开顶端的经纬度标签
    gl.bottom_labels = bottom_labels  # 打开底端的经纬度标签
    gl.left_labels = left_labels  # 打开左侧的经纬度标签
    gl.right_labels = right_labels  # 打开右侧的经纬度标签
    gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
    gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式
    return ax


# Function to plot spatial distribution
def single_spatial_distribution(inputPath, outputPath, var_name, level,
                                colorbar_ticks, colorbar_label, figure_name,
                                cmap=plt.cm.Spectral, extend="both",
                                signification_path=None, signification_name=None,
                                boundary_labels=None):
    """
    Plot spatial distribution of variables with optional significance overlay
    """
    # Check the output folder
    create_path(outputPath)
    """Plotting Parameters"""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "24"

    """Plotting"""
    fig = plt.figure(figsize=(10, 10))
    ax_main = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax_main.set_ylim(-60, 90)
    # Add geographical information and boundaries
    boundary_labels = [True, True, False, False] if boundary_labels is None else boundary_labels
    draw_gis_figure(ax_main, left_labels=boundary_labels[0], top_labels=boundary_labels[1],
                    bottom_labels=boundary_labels[2], right_labels=boundary_labels[3])
    ####################################################################################################################
    # Load data
    data = xr.open_dataset(inputPath)
    variable = data[var_name].sel(lat=slice(-60, 90), lon=slice(-180, 180))
    with rio.open(r"../dataFile/ForestCharacteristics/ForestManagement.tif") as ds:
        ifl = ds.read(1)[::-1]
    variable.values[ifl == ifl[0, 0]] = np.nan
    if var_name == "frequency_sum":
        variable.values = variable.values / 4
    # Set color map and data range
    norm = mcolors.Normalize(vmin=level[0], vmax=level[1], clip=False)
    # Plot main map
    contour_plot = variable.plot.contourf(ax=ax_main, transform=ccrs.PlateCarree(),
                                          levels=np.linspace(level[0], level[1], 5),
                                          extend=extend,
                                          cmap=cmap, add_colorbar=False)

    # Add color bar
    cb = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb.set_array([])
    cax = fig.add_axes([0.18, 0.29, .6, .025])
    fig.colorbar(cb, cax=cax, orientation='horizontal', extend=extend,
                 ticks=colorbar_ticks, label=colorbar_label)

    # Read significance data
    if (signification_path is not None) and (signification_name is not None):
        sign_data = xr.open_dataset(signification_path)
        sign_variable = sign_data[signification_name].sel(lat=slice(-60, 90), lon=slice(-180, 180))
        significance_mask = np.abs(sign_variable.values) > 1.96
        # Mark significant data
        lons, lats = np.meshgrid(variable.lon.values, variable.lat.values)
        ax_main.scatter(lons[significance_mask], lats[significance_mask], color='k', s=0.5, marker='*',
                        facecolors='k', edgecolors='none', transform=ccrs.PlateCarree())

    # Show or save the figure
    fig.savefig(os.path.join(outputPath, f"{figure_name}.{export_figure_type}"), dpi=export_figure_dpi, bbox_inches='tight')
    plt.close()


# Function to plot spatial distribution
def single_spatial_distribution_without_ft(inputPath, outputPath, var_name, level,
                                           colorbar_ticks, colorbar_label, figure_name,
                                           cmap=plt.cm.Spectral, extend="both",
                                           signification_path=None, signification_name=None,
                                           boundary_labels=None):
    """
    Plot spatial distribution of variables with optional significance overlay
    """
    # Check the output folder
    create_path(outputPath)
    """Plotting Parameters"""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "24"

    """Plotting"""
    fig = plt.figure(figsize=(10, 10))
    ax_main = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax_main.set_ylim(-60, 90)
    # Add geographical information and boundaries
    boundary_labels = [True, True, False, False] if boundary_labels is None else boundary_labels
    draw_gis_figure(ax_main, left_labels=boundary_labels[0], top_labels=boundary_labels[1],
                    bottom_labels=boundary_labels[2], right_labels=boundary_labels[3])
    ####################################################################################################################
    # Load data
    data = xr.open_dataset(inputPath)
    variable = data[var_name].sel(lat=slice(-60, 90), lon=slice(-180, 180))
    if var_name == "frequency_sum":
        variable.values = variable.values / 4
    # Set color map and data range
    norm = mcolors.Normalize(vmin=level[0], vmax=level[1], clip=False)
    # Plot main map
    contour_plot = variable.plot.contourf(ax=ax_main, transform=ccrs.PlateCarree(),
                                          levels=np.linspace(level[0], level[1], 5),
                                          extend=extend,
                                          cmap=cmap, add_colorbar=False)

    # Add color bar
    cb = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb.set_array([])
    cax = fig.add_axes([0.18, 0.29, .6, .025])
    fig.colorbar(cb, cax=cax, orientation='horizontal', extend=extend,
                 ticks=colorbar_ticks, label=colorbar_label)

    # Read significance data
    if (signification_path is not None) and (signification_name is not None):
        sign_data = xr.open_dataset(signification_path)
        sign_variable = sign_data[signification_name].sel(lat=slice(-60, 90), lon=slice(-180, 180))
        significance_mask = np.abs(sign_variable.values) > 1.96
        # Mark significant data
        lons, lats = np.meshgrid(variable.lon.values, variable.lat.values)
        ax_main.scatter(lons[significance_mask], lats[significance_mask], color='k', s=0.5, marker='*',
                        facecolors='k', edgecolors='none', transform=ccrs.PlateCarree())

    # Show or save the figure
    fig.savefig(os.path.join(outputPath, f"{figure_name}.{export_figure_type}"), dpi=export_figure_dpi, bbox_inches='tight')
    plt.close()


# Function to plot spatial distribution with Mann-Kendall test
def spatial_distribution_mk_test(data_path, outputPath, var_name, level, colorbar_ticks, tick_label, colorbar_label,
                                 figure_name, colors=plt.cm.Spectral, extend="both"):
    """
    Plot spatial distribution with Mann-Kendall test for trend analysis
    """
    # Check the output folder
    create_path(outputPath)
    """Plotting Parameters"""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "24"

    """Plotting"""
    # Create figure and axes
    fig = plt.figure(figsize=(10, 10))
    ax_main = fig.add_subplot(111, projection=ccrs.PlateCarree())
    # Add geographical information and boundaries
    draw_gis_figure(ax_main, top_labels=False, bottom_labels=False, left_labels=False, right_labels=False)
    ####################################################################################################################
    # Load data
    data = xr.open_dataset(data_path)
    variable = data[var_name].sel(lat=slice(-90, 90), lon=slice(-180, 180))
    with rio.open(r"../dataFile/ForestCharacteristics/ForestManagement.tif") as ds:
        ifl = ds.read(1)[::-1]
    variable.values[ifl == ifl[0, 0]] = np.nan
    level = np.asarray(level)
    cmap, norm = mcolors.from_levels_and_colors(level, colors, extend=extend)
    # Plot main map
    contour_plot = variable.plot.contourf(ax=ax_main, transform=ccrs.PlateCarree(),
                                          levels=level,
                                          cmap=cmap,
                                          extend=extend, add_colorbar=False)

    # # Add color bar
    # cax = fig.add_axes([0.155, -0.05, .02, .3])  # Left, bottom, width, height
    # colorbar = fig.colorbar(contour_plot, ax=ax_main, cax=cax, orientation='vertical', ticks=colorbar_ticks,
    #                         extendfrac=0.2,
    #                         extendrect=True)
    # colorbar.set_label(colorbar_label, labelpad=-370)
    # colorbar.set_ticklabels(tick_label)

    # Show or save the figure
    fig.savefig(os.path.join(outputPath, f"{figure_name}.{export_figure_type}"), dpi=export_figure_dpi, bbox_inches='tight')
    plt.close()


# Plotting the latitudinal mean
def lat_mean_plot(
        outputPath: str,
        inputPath: tuple,
        var_name: tuple,
        colorbar_label: tuple,
        line_ticks: tuple,
        figure_name: str,
        line_xlim: tuple,
        line_colors: tuple = (("#f44d38", "#fdccb8"), ("#23a884", "#97d492")),
):
    """
    Plot the latitudinal mean for the variables with shaded areas indicating the uncertainty
    """
    # Check the output folder
    create_path(outputPath)
    """Plotting Parameters"""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "18"

    """Plotting"""
    fig = plt.figure(figsize=(2, 10))
    main_ax = fig.add_axes([0.1, 0.4, 0.8, 0.5])

    # Set main axis properties
    main_ax.set_ylim(-50, 75)
    main_ax.set_yticks([-30, 0, 30, 60])
    main_ax.set_yticklabels(["30°S", "0°", "30°N", "60°N"])
    main_ax.tick_params(bottom=False, labelbottom=False)
    main_ax.yaxis.tick_right()
    main_ax.tick_params(axis='y', direction='in', length=3)
    main_ax.tick_params(axis='x', direction='out', length=0)
    main_ax.axvline(x=0, color="black", linestyle='-', lw=2)
    ####################################################################################################################
    with rio.open(r"../dataFile/ForestCharacteristics/ForestManagement.tif") as ds:
        ifl = ds.read(1)[::-1]
    for i, path in enumerate(inputPath):
        # Load data
        data = xr.open_dataset(path)
        variable = data[var_name[i]].sel(lat=slice(-60, 90), lon=slice(-180, 180))
        variable.values[ifl == ifl[0, 0]] = np.nan
        variable = variable.sel(lat=slice(-60, 70), lon=slice(-180, 180))
        # Plot the mean curve for each latitudinal band
        lat_bins = np.arange(-45, 90, 5)  # Latitude bin boundaries
        lat_groups = variable.groupby_bins("lat", lat_bins).mean(dim="lat")
        lat_mean = lat_groups.mean(dim='lon')
        # Calculate standard error
        lat_se = lat_groups.std(dim='lon') / np.sqrt(lat_groups.count(dim='lon'))
        lat_lower = lat_mean - lat_se
        lat_upper = lat_mean + lat_se
        # Plot the mean curve and shaded area for uncertainty
        lat_centers = [(lat_bins[j] + lat_bins[j + 1]) / 2 for j in range(len(lat_bins) - 1)]
        ####################################################################################################################
        # If not the first dataset, create a new x-axis that overlays with the original one
        ax = main_ax.twiny()  # Create a new x-axis sharing the y-axis
        ax.spines["top"].set_position(("axes", -0.08 - 0.11 * i))  # Set the new x-axis position
        ax.spines["top"].set_visible(True)
        ax.axvline(x=0, color="black", linestyle='--', lw=2)
        ax.plot(lat_mean.values, lat_centers, color=line_colors[i][0], linewidth=2)
        # ax.fill_betweenx(lat_centers, lat_lower.values, lat_upper.values,
        #                  color=line_colors[i][1], alpha=0.1)

        # Set x-axis labels
        ax.set_xlabel(colorbar_label[i], color=line_colors[i][0])
        ax.set_xlim(line_xlim[i])
        ax.set_xticks(line_ticks[i])
        ax.xaxis.set_label_coords(0.5, -0.12 - (0.11 * i))  # x position stays the same, y position moves down
        ax.tick_params(axis='x', direction='out', length=3, pad=0)

        # Hide all but the bottom x-axis tick labels
        if i < len(colorbar_label) - 1:
            ax.tick_params(labelbottom=False)
    # Show or save the figure
    fig.savefig(os.path.join(outputPath, f"{figure_name}.{export_figure_type}"), dpi=export_figure_dpi, bbox_inches='tight')
    plt.close()


# Function to plot delta NDVI spatial distribution
def delta_dv_spatial_distribution(inputPath, outputPath, var_name, colorbar_label, figure_name,
                                  cmap=["#00441b", "#4db163", "#08306b", "#3f8fc5"], extend="both",
                                  ):
    """
    Plot the delta NDVI spatial distribution with a custom color map
    """
    # Check the output folder
    create_path(outputPath)
    """Plotting Parameters"""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "12"

    """Plotting"""
    fig = plt.figure(figsize=(10, 10))
    ax_main = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax_main.set_ylim(-60, 90)
    # Add geographical information and boundaries
    draw_gis_figure(ax_main, left_labels=True, top_labels=True, bottom_labels=False, right_labels=False)
    ####################################################################################################################
    # Load data
    data = xr.open_dataset(inputPath)
    variable = data[var_name].sel(lat=slice(-60, 90), lon=slice(-180, 180))
    with rio.open(r"../dataFile/ForestCharacteristics/ForestManagement.tif") as ds:
        ifl = ds.read(1)[::-1]
    variable.values[ifl == ifl[0, 0]] = np.nan

    variable_classify = variable.copy()
    variable_classify.values[((ifl == 11) & (variable.values < 0))] = 1
    variable_classify.values[((ifl == 11) & (variable.values > 0))] = 2
    variable_classify.values[((ifl != 11) & (variable.values < 0))] = 3
    variable_classify.values[((ifl != 11) & (variable.values > 0))] = 4

    # Define custom color map and bounds
    cmap = mcolors.ListedColormap(cmap)
    bounds = [1, 2, 3, 4, 5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot main map
    contour_plot = ax_main.pcolormesh(variable.lon, variable.lat, variable_classify,
                                      cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

    # # Add color bar
    # cb = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # cb.set_array([])
    # cax = fig.add_axes([0.18, 0.29, .6, .025])
    # cb = fig.colorbar(cb, cax=cax, orientation='horizontal', extend=extend,
    #                   ticks=[1.5, 2.5, 3.5, 4.5],
    #                   label=colorbar_label)  # Adjust ticks to be at the center of the intervals
    # cb.set_ticklabels(['decrease IF', 'increase IF', 'decrease MF', 'increase MF'])

    # Show or save the figure
    fig.savefig(os.path.join(outputPath, f"{figure_name}.{export_figure_type}"), dpi=export_figure_dpi, bbox_inches='tight')
    plt.close()


# Function for calculating characteristic latitudinal mean and plotting
def characteristic_lat_mean(
        outputPath: str,
        inputPath: tuple,
        var_name: tuple,
        colorbar_label: tuple,
        line_ticks: tuple,
        figure_name: str,
        line_xlim: tuple,
        line_colors: tuple = (("#f44d38", "#fdccb8"), ("#23a884", "#97d492")),
):
    """
    Plot the characteristic latitudinal mean with shading for uncertainty
    """
    # Check the output folder
    create_path(outputPath)
    """Plotting Parameters"""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "12"

    """Plotting"""
    fig = plt.figure(figsize=(2, 7))
    main_ax = fig.add_axes([0.1, 0.4, 0.8, 0.5])

    # Set main axis properties
    main_ax.set_ylim(-50, 75)
    main_ax.set_yticks([-30, 0, 30, 60])
    main_ax.set_yticklabels(["30°S", "0°", "30°N", "60°N"])
    main_ax.tick_params(bottom=False, labelbottom=False)
    main_ax.yaxis.tick_right()
    main_ax.tick_params(axis='y', direction='in', length=3)
    main_ax.tick_params(axis='x', direction='out', length=0)
    main_ax.axvline(x=0, color="black", linestyle='-', lw=2)
    ####################################################################################################################
    with rio.open(r"../dataFile/ForestCharacteristics/ForestManagement.tif") as ds:
        ifl = ds.read(1)[::-1]
    for i, path in enumerate(inputPath):
        # Load data
        data = xr.open_dataset(path)
        variable = data[var_name[i]].sel(lat=slice(-60, 90), lon=slice(-180, 180))
        variable.values[ifl == ifl[0, 0]] = np.nan
        variable = variable.sel(lat=slice(-60, 70), lon=slice(-180, 180))
        variable *= 10
        # Plot the mean curve for each latitudinal band
        lat_bins = np.arange(-45, 90, 9)  # Latitude bin boundaries
        lat_groups = variable.groupby_bins("lat", lat_bins).mean(dim="lat")
        lat_mean = lat_groups.mean(dim='lon')
        # Calculate standard error
        lat_se = lat_groups.std(dim='lon') / np.sqrt(lat_groups.count(dim='lon'))
        lat_lower = lat_mean - lat_se
        lat_upper = lat_mean + lat_se
        # Plot the mean curve and shaded area for uncertainty
        lat_centers = [(lat_bins[j] + lat_bins[j + 1]) / 2 for j in range(len(lat_bins) - 1)]
        ####################################################################################################################
        # If not the first dataset, create a new x-axis that overlays with the original one
        ax = main_ax.twiny()  # Create a new x-axis sharing the y-axis
        ax.spines["top"].set_position(("axes", -0.065 - 0.11 * i))  # Set the new x-axis position
        ax.spines["top"].set_visible(True)
        ax.axvline(x=0, color="black", linestyle='--', lw=2)
        ax.plot(lat_mean.values, lat_centers, color=line_colors[i][0], linewidth=2)
        # ax.fill_betweenx(lat_centers, lat_lower.values, lat_upper.values,
        #                  color=line_colors[i][1], alpha=0.1)

        # Set x-axis labels
        ax.set_xlabel(colorbar_label[i], color=line_colors[i][0])
        ax.set_xlim(line_xlim[i])
        ax.set_xticks(line_ticks[i])
        ax.xaxis.set_label_coords(0.5, -0.11 - (0.11 * i))  # x position stays the same, y position moves down
        ax.tick_params(axis='x', direction='out', length=3, pad=0)

        # Hide all but the bottom x-axis tick labels
        if i < len(colorbar_label) - 1:
            ax.tick_params(labelbottom=False)
    # Show or save the figure
    fig.savefig(os.path.join(outputPath, f"{figure_name}.{export_figure_type}"), dpi=export_figure_dpi, bbox_inches='tight')
    plt.close()


# Function for plotting the delta NDVI distribution with significant testing
def delta_ndvi_distribution_plot(datas, classify_name, colors, outputPath, figure_name,
                                 colorbar_label):
    """
    Plot the delta NDVI distribution with KDE and significance testing
    """
    # Plotting settings
    rcParams['font.size'] = 10
    rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(5, 2))
    # Initialize max density variable
    max_density = 0
    mean_values = []
    p_values = []
    # Plot KDE and significance markers
    for i, c in enumerate(classify_name):
        domain_values = datas[i]
        domain_values = domain_values[~np.isnan(domain_values)]

        # Only take values between the 5th and 95th percentiles
        lower_bound = np.percentile(domain_values, 1)
        upper_bound = np.percentile(domain_values, 99)
        filtered_values = domain_values[(domain_values >= lower_bound) & (domain_values <= upper_bound)]

        # Compute KDE
        kde = gaussian_kde(filtered_values)
        x_d = np.linspace(lower_bound, upper_bound, 1000)
        y_d = kde(x_d)
        max_density = max(max_density, y_d.max())

        # Fill the area under the curve
        plt.fill_between(x_d, y_d, color=colors[i], alpha=0.3, edgecolor=None, linewidth=0, label=f'{c}')

    plt.axvline(x=0, color="black", linestyle='-', lw=2)
    for i, c in enumerate(classify_name):
        domain_values = datas[i]
        domain_values = domain_values[~np.isnan(domain_values)]
        # Vertical line for mean
        mean_val = np.mean(domain_values)
        print(f" >>> {c}: {mean_val}", end="")
        mean_values.append(mean_val)
        plt.axvline(x=mean_val, color=colors[i], linestyle='-', lw=1)

        # Only take values between the 5th and 95th percentiles
        lower_bound = np.percentile(domain_values, 1)
        upper_bound = np.percentile(domain_values, 99)
        filtered_values = domain_values[(domain_values >= lower_bound) & (domain_values <= upper_bound)]

        # Perform t-test
        t_stat, p_value = stats.ttest_1samp(filtered_values, 0)
        p_values.append(p_value)

    plt.legend(loc='upper left', frameon=False)
    plt.xlabel(colorbar_label)
    plt.ylabel('Density')
    plt.tick_params(direction='in')

    # Set y-axis range slightly higher than the max density
    plt.ylim(bottom=0, top=max_density * 1.1)

    plt.savefig(os.path.join(outputPath, f"{figure_name}.{export_figure_type}"), dpi=export_figure_dpi, bbox_inches='tight')
    plt.close()
    # Reset matplotlib global settings
    rcParams.update(rcParamsDefault)


# Hexbin plot function for visualizing the relationship between two variables with hue
def hexbin_plot(pPath, tPath, xlabel, ylabel,
                inputPath, outputPath, var_name,
                level, colorbar_ticks, colorbar_label, figure_name, cmap):
    """
    Create a hexbin plot to visualize the relationship between two variables with color-coded hue
    """
    # Extract data
    with rio.open(pPath) as ds:
        x = ds.read(1)[300:1800]
    with rio.open(tPath) as ds:
        y = ds.read(1)[300:1800]
    hue = xr.open_dataset(inputPath)[var_name].sel(lat=slice(-60, 90))
    with rio.open(r"../dataFile/ForestCharacteristics/ForestManagement.tif") as ds:
        ifl = ds.read(1)[::-1]
    hue.values[ifl == ifl[0, 0]] = np.nan

    mask = ~np.isnan(hue.values)
    x_flatten = x[mask].flatten()
    y_flatten = y[mask].flatten()
    hue_flatten = hue.values[mask].flatten()
    ####################################################################################################################
    # Create 2D histogram grid
    xedges = np.arange(200, 2800, 200)
    yedges = np.arange(-14, 32, 4)
    H, xedges, yedges = np.histogram2d(x_flatten, y_flatten, bins=(xedges, yedges))
    # Initialize accumulators and counters for each bin
    hue_sum = np.zeros(H.shape)
    hue_count = np.zeros(H.shape)
    # Calculate which bin each point belongs to
    xidx = np.digitize(x_flatten, xedges) - 1
    yidx = np.digitize(y_flatten, yedges) - 1
    # Fix out-of-bounds indices
    xidx = np.clip(xidx, 0, H.shape[0] - 1)
    yidx = np.clip(yidx, 0, H.shape[1] - 1)
    # Accumulate hue values and count occurrences
    for i in range(x_flatten.size):
        hue_sum[xidx[i], yidx[i]] += hue_flatten[i]
        hue_count[xidx[i], yidx[i]] += 1
    # Calculate the average hue value for each bin, avoiding division by zero
    hue_mean = np.divide(hue_sum, hue_count, where=hue_count != 0)
    # Calculate the center coordinates for each bin
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    # Set tick labels for the grid
    step = 2
    xticks = [f'{v:.0f}' for i, v in enumerate(xcenters) if i % step == 0]
    yticks = [f'{v:.0f}' for i, v in enumerate(ycenters) if i % step == 0]
    xtick_positions = np.arange(0, len(xcenters), step) + 0.5
    ytick_positions = np.arange(0, len(ycenters), step) + 0.5
    ####################################################################################################################
    # Plotting settings
    rcParams['font.size'] = 15
    rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(hue_mean.T, ax=ax,
                cmap=cmap, norm=mcolors.Normalize(vmin=level[0], vmax=level[1]),
                cbar_kws={'ticks': colorbar_ticks})
    # Add color bar
    cbar = ax.collections[0].colorbar
    cbar.set_label(colorbar_label)
    ax.set_xticks(xtick_positions)
    ax.set_yticks(ytick_positions)
    ax.set_xticklabels(xticks, rotation=90)
    ax.set_yticklabels(yticks, rotation=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    outputFigurePath = os.path.join(outputPath, f"{figure_name}.{export_figure_type}")
    create_path(outputPath)
    plt.savefig(outputFigurePath, dpi=export_figure_dpi, bbox_inches='tight')
    plt.close()
    # Reset matplotlib global settings
    rcParams.update(rcParamsDefault)


# Function to select numbers based on a step rule
def select_numbers_by_step_rule(array, num=5):
    """
    Select numbers from an array according to a step rule
    """
    array = np.asarray(array)  # Ensure input is a numpy array
    step_size = len(array) // num
    step_size += 1 if step_size <= (len(array) % num) else 0
    start_index = step_size // 2 + step_size % 2 - 1

    indices = np.arange(start_index, len(array), step_size)
    selected_numbers = array[indices]

    return selected_numbers


# Function to analyze and visualize interaction effects between two features
def interaction_effects(df, main_feature, interaction_feature, classify_col,
                        colormaps, scatter_size, line_colors, colorbar_ticks, colorbar_extend,
                        x_lims, y_lims, y_ticks, x_label, colorbar_label, group_size,
                        outputPath, figure_name, ver_lines):
    """
    Visualize interaction effects between two features, with line plots showing the effects by groups
    """
    # Plotting settings
    rcParams['font.size'] = 27
    rcParams['font.family'] = 'Times New Roman'

    # Create interaction effect plot
    df = df.sort_values(by=interaction_feature, ascending=False)
    y = df[f"{main_feature} vs {interaction_feature}"] * 2
    X = df[main_feature]
    if main_feature == "Den":
        X = X / 10000
    elif main_feature == "MAP":
        X = X / 100
    hue = df[interaction_feature]
    fig, ax = plt.subplots(figsize=(5, 5))
    norm = Normalize(vmin=min(colorbar_ticks), vmax=max(colorbar_ticks))
    scatter = ax.scatter(X, y, c=hue, marker='o', cmap=colormaps, norm=norm, s=scatter_size)

    # # Add color bar
    # cb = plt.colorbar(scatter, ticks=colorbar_ticks, extend=colorbar_extend)
    # cb.set_label(label=colorbar_label)

    # Plot auxiliary lines
    plt.axhline(y=0, color='black', linestyle='--')
    if ver_lines is not None:
        plt.axvline(x=ver_lines, color='black', linestyle='--')

    # Create line plots for each classification group
    bins = np.concatenate(
        ([-np.inf], np.arange(x_lims[0] + group_size, x_lims[1] + group_size, group_size), [np.inf]))
    group_label = np.arange(x_lims[0], x_lims[1] + group_size, group_size)
    for i, classify in enumerate(np.setdiff1d(df[classify_col].unique(), ["nan"])):
        mask = df[classify_col] == classify
        sub_X = X[mask]
        sub_y = y[mask]
        line_color = line_colors[classify]
        sub_df = pd.DataFrame({'X': sub_X, 'y': sub_y})

        # Use pd.cut to bin the data
        sub_df['bin'] = pd.cut(sub_df['X'], bins=bins, right=False, labels=group_label)

        # Calculate means and standard errors for each bin
        grouped = sub_df.groupby('bin').agg(mean_y=('y', 'mean'), std_y=('y', 'std'), count=('y', 'count'))
        grouped['se_y'] = grouped['std_y'] / np.sqrt(grouped['count'])

        # Exclude bins with too few samples
        grouped = grouped[grouped["count"] >= 50]

        ax.plot(grouped.index, grouped['mean_y'], '-', color=line_color, lw=2, label=classify)
        # ax.fill_between(grouped.index, grouped['mean_y'] - grouped['se_y'], grouped['mean_y'] + grouped['se_y'],
        #                 color=line_color, alpha=0.1)

    ax.set_xlabel(x_label)
    ax.set_ylabel("ΔNDVI")
    ax.set_xlim(group_label[0] - group_size * 0.5, group_label[-1] + group_size * 0.5)
    ax.set_xticks(select_numbers_by_step_rule(group_label, num=5))
    ax.set_ylim(y_lims)
    ax.set_yticks(y_ticks)
    ax.tick_params(direction='in', length=3)

    output_figure_path = os.path.join(outputPath, f"{figure_name}.{export_figure_type}")
    create_path(output_figure_path)
    fig.savefig(output_figure_path, dpi=export_figure_dpi, bbox_inches='tight')
    plt.close(fig)

    # Reset matplotlib global settings
    rcParams.update(rcParamsDefault)

# Function for plotting the main effects with lines
def main_effect_lines(y, X, outputPath, figure_name, x_lim, y_lim, group_size, labels, label_decimal=0):
    """
    Plot the main effects with lines for grouped data, including statistical significance
    """
    # Plotting settings
    rcParams['font.size'] = 24
    rcParams['font.family'] = 'Times New Roman'
    warnings.simplefilter(action='ignore', category=FutureWarning)

    fig, ax = plt.subplots(figsize=(5, 5))
    group = np.concatenate(([-np.inf], np.arange(x_lim[0] + group_size, x_lim[1] + group_size, group_size), [np.inf]))
    group_label = np.arange(x_lim[0], x_lim[1] + group_size, group_size)

    df = pd.DataFrame({"y": y, "X": X})
    df["group_label"] = pd.cut(X, bins=group, right=False, labels=group_label)

    colors = []
    for label in group_label:
        group_data = df[df["group_label"] == label]["y"]
        if len(group_data) > 1:
            threshold = 0
            median_val = group_data.median()

            if median_val > threshold:
                colors.append('#1f77b4')
            elif median_val < -threshold:
                colors.append('#ff7f0e')
            else:
                colors.append('black')
        else:
            colors.append('white')
    # Create vertical box plot with custom styling
    sns.boxplot(x="group_label", y="y", data=df, showfliers=False,
                palette=colors,
                width=0.5,
                fill=False, linewidth=1.5)

    ax.axhline(y=0, color="black", linestyle='--', lw=1)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_ylim(y_lim[0], y_lim[1])

    if group_label.max() >= 10000:
        group_label = group_label / 1e4

    ax.set_xticks(np.arange(len(group_label)))
    if labels[0] == "Age /years":
        ax.set_xticklabels([1, "", 50, "", 100, "", 150, "", 200, "", 250])
    elif labels[0] == "CH /m":
        ax.set_xticklabels([1, "", 10, "", 20, "", 30, "", 40])
    elif label_decimal != 0:
        ax.set_xticklabels([f"{np.round(label, decimals=label_decimal)}"
                            if i % 2 == 0 else '' for i, label in enumerate(group_label)])
    else:
        ax.set_xticklabels([f"{int(label)}"
                            if i % 2 == 0 else '' for i, label in enumerate(group_label)])

    ax.tick_params(axis="x", direction='in')
    ax.tick_params(axis="y", direction='in')
    ax.tick_params(length=3)

    output_figure_path = os.path.join(outputPath, f"{figure_name}.{export_figure_type}")
    create_path(output_figure_path)
    fig.savefig(output_figure_path, dpi=export_figure_dpi, bbox_inches='tight')
    plt.close(fig)

    # Reset matplotlib global settings
    rcParams.update(rcParamsDefault)
    warnings.simplefilter(action='ignore', category=FutureWarning)

# Forest management box plot for analyzing the forest types and corresponding characteristics
def forest_management_bar(df, classify_col, object_val, colors, xlabel, ylabel, outputpath, figure_name):
    """
    Create a vertical box plot to show forest management characteristics
    """
    # Plotting settings
    rcParams['font.size'] = 20
    rcParams['font.family'] = 'Times New Roman'
    create_path(outputpath, is_dir=True)
    plt.figure(figsize=(7, 5))  # Keep square figure size

    # Create vertical box plot with custom styling
    sns.boxplot(x=classify_col, y=object_val, data=df, showfliers=False,
                palette=colors, width=0.5,
                boxprops=dict(edgecolor='black', linewidth=1.5),
                medianprops=dict(color='black', linewidth=1.5),
                whiskerprops=dict(color='black', linewidth=1.5),
                capprops=dict(color='black', linewidth=1.5))

    # Add horizontal reference line at 0
    plt.axhline(y=0, color='black', linestyle='--', zorder=0)

    # Axis labels and limits
    plt.xlabel(ylabel)  # Swapped labels
    plt.ylabel(xlabel)  # Swapped labels
    plt.ylim(-1.75, 1.75)
    plt.yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(outputpath, f"{figure_name}.{export_figure_type}"), dpi=export_figure_dpi)

    # Reset default parameters
    rcParams.update(rcParamsDefault)


# Forest management box plot for analyzing the forest types and corresponding characteristics
def forest_management_practices_bar(df, classify_col, object_val, colors, ylabel, xlabel, outputpath, figure_name):
    """
    Create a vertical box plot to show forest management characteristics
    """
    # Plotting settings
    rcParams['font.size'] = 20
    rcParams['font.family'] = 'Times New Roman'
    create_path(outputpath, is_dir=True)
    plt.figure(figsize=(7, 5))  # Keep square figure size

    # Create vertical box plot with custom styling
    sns.boxplot(x=classify_col, y=object_val, data=df, showfliers=False,
                palette=colors, width=0.5,
                boxprops=dict(edgecolor='black', linewidth=1.5),
                medianprops=dict(color='black', linewidth=1.5),
                whiskerprops=dict(color='black', linewidth=1.5),
                capprops=dict(color='black', linewidth=1.5))

    # Add horizontal reference line at 0
    plt.axhline(y=0, color='black', linestyle='--', zorder=0)

    # Axis labels and limits
    plt.xlabel(xlabel)  # Swapped labels
    plt.ylabel(ylabel)  # Swapped labels
    plt.ylim(-1.75, 1.75)
    plt.yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(outputpath, f"{figure_name}.{export_figure_type}"), dpi=export_figure_dpi)

    # Reset default parameters
    rcParams.update(rcParamsDefault)


def consistency_lat_delta(
        outputPath: str,
        inputPath: str,
        var_name: dict,
        figure_name: str,
        x_axis_ticks: tuple = (0, 50, 100),
        bar_xlim: tuple = (0, 100),
        bar_colors: dict = {
            "delta_p": "#1E90FF",  # Dodger blue
            "delta_ta": "#FF4500",  # Orange red
            "both": "#800080",  # Purple
        },
):
    """
    Plot the consistency of latitudinal delta variables with the specified settings
    """
    # Check the output folder
    create_path(outputPath)
    """Plotting Parameters"""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "22"

    """Plotting"""
    fig = plt.figure(figsize=(1.5, 3))
    main_ax = fig.add_axes([0, 0, 1, 1])

    # Set main axis properties
    main_ax.set_ylim(-50, 75)
    main_ax.set_yticks([-30, 0, 30, 60])
    main_ax.set_yticklabels(["30°S", "0°", "30°N", "60°N"])
    main_ax.yaxis.tick_right()
    main_ax.tick_params(axis='y', direction='in', length=3)
    main_ax.tick_params(axis='x', direction='in', length=3)
    ####################################################################################################################
    with rio.open(r"../dataFile/ForestCharacteristics/ForestManagement.tif") as ds:
        ifl = ds.read(1)[::-1]
    # Load data into a dictionary
    data_dict = {}
    for key, val in var_name.items():
        file_path = os.path.join(inputPath, f"{val}.nc")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} does not exist.")
            continue
        try:
            data = xr.open_dataset(file_path)[val]
            data.values[ifl == ifl[0, 0]] = np.nan
            data_dict[key] = data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # Ensure both delta_p and delta_ta are loaded
    if "delta_p" not in data_dict or "delta_ta" not in data_dict:
        print("Error: Both 'delta_p' and 'delta_ta' datasets must be loaded.")
        return

    delta_p = data_dict["delta_p"]
    delta_ta = data_dict["delta_ta"]

    # Define latitude groups
    lat_groups = [
        (90, 60),
        (60, 30),
        (30, 0),
        (0, -30),
        (-30, -60),
    ]
    lat_labels = [
        65,
        40,
        13,
        -13,
        -40
    ]
    conditions = ["delta_p", "delta_ta", "both"]
    proportions = {
        "Latitude Group": [],
        "delta_p": [],
        "delta_ta": [],
        "both": []
    }

    # Loop through each latitude group
    for group, group_label in zip(lat_groups, lat_labels):
        max_lat, min_lat = group
        proportions["Latitude Group"].append(group_label)

        try:
            delta_p_subset = delta_p.sel(lat=slice(min_lat, max_lat)).values.flatten()
            delta_ta_subset = delta_ta.sel(lat=slice(min_lat, max_lat)).values.flatten()
        except Exception as e:
            print(f"Error selecting data for latitude group {group_label}: {e}")
            proportions["delta_p"].append(np.nan)
            proportions["delta_ta"].append(np.nan)
            proportions["both"].append(np.nan)
            continue

        mask = ~np.isnan(delta_p_subset) & ~np.isnan(delta_ta_subset)
        delta_p_clean = delta_p_subset[mask]
        delta_ta_clean = delta_ta_subset[mask]

        total = len(delta_p_clean)
        if total == 0:
            print(f"No valid data points for latitude group {group_label}.")
            proportions["delta_p"].append(0)
            proportions["delta_ta"].append(0)
            proportions["both"].append(0)
            continue

        # Calculate conditions
        condition1 = (delta_p_clean < -0.5) & (delta_ta_clean <= 0.5)
        condition2 = (delta_ta_clean > 0.5) & (delta_p_clean >= -0.5)
        condition3 = (delta_p_clean < -0.5) & (delta_ta_clean > 0.5)

        prop1 = np.sum(condition1) / total
        prop2 = np.sum(condition2) / total
        prop3 = np.sum(condition3) / total

        proportions["delta_p"].append(prop1)
        proportions["delta_ta"].append(prop2)
        proportions["both"].append(prop3)

    # Convert proportions to DataFrame
    df_proportions = pd.DataFrame(proportions)
    df_proportions.fillna(0, inplace=True)
    df_proportions['Total'] = df_proportions[conditions].sum(axis=1)

    df_proportions = df_proportions[df_proportions['Total'] > 0]
    df_proportions.drop(columns=['Total'], inplace=True)
    df_proportions.set_index("Latitude Group", inplace=True)

    df_proportions_percent = df_proportions * 100
    ####################################################################################################################
    bottom = np.zeros(len(df_proportions_percent))

    for condition in conditions:
        main_ax.barh(df_proportions_percent.index, df_proportions_percent[condition], left=bottom,
                     color=bar_colors[condition], label=condition, height=10)
        bottom += df_proportions_percent[condition].values

    main_ax.set_xlabel("Proportion (%)")
    main_ax.set_xlim(bar_xlim)
    main_ax.set_xticks(x_axis_ticks)

    fig.savefig(os.path.join(outputPath, f"{figure_name}.{export_figure_type}"), dpi=export_figure_dpi, bbox_inches='tight')
    plt.close()
    rcParams.update(rcParamsDefault)


# Consistency calculation between reference and comparison variables
def calculate_consistency_percentage(reference: xr.DataArray, comparison: xr.DataArray, var_name: str,
                                     threshold: float = 0) -> float:
    """
    Calculate the consistency percentage between the reference and comparison variables, considering only values
    greater than the threshold.
    """
    valid_mask = (~np.isnan(reference)) & (~np.isnan(comparison)) & (np.abs(reference) > threshold)

    if valid_mask.sum() == 0:
        consistency_percentage = np.nan
    else:
        same_sign = ((reference.values[valid_mask] * comparison.values[valid_mask]) > 0) & (
                np.abs(comparison.values[valid_mask]) > threshold)
        consistency_percentage = (same_sign.sum() / valid_mask.sum()) * 100
    return consistency_percentage


# Function to calculate and visualize consistency between biophysical variables based on latitude
def consistency_bio_lat(
        outputPath: str,
        inputPath: str,
        var_names: Dict[str, str],
        figure_name: str,
        ref_variable: str = "NDVI",
        variables_to_analyze: List[str] = ["LAI", "SIF"],
        x_axis_ticks: Tuple[int, ...] = (0, 50, 100),
        line_xlim: Tuple[int, int] = (0, 100),
        forest_mask: str = None,
        output_csv_summary: bool = False,
        output_csv_lat: bool = False
):
    """
    Calculate and plot consistency between biophysical variables (e.g., LAI, SIF) and a reference variable (e.g., NDVI)
    by latitude.
    """
    # Check and create the output folder
    create_path(outputPath)

    # Set plotting parameters
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "26"

    # Load the forest classification data and reverse the vertical direction
    forest_mask = r"../dataFile/ForestCharacteristics/ForestManagement.tif" if forest_mask is None else forest_mask
    try:
        with rio.open(forest_mask) as ds:
            forest_mask = ds.read(1)[::-1]
    except Exception as e:
        raise FileNotFoundError(f"Failed to load the forest classification file {forest_mask}: {e}")

    def read_nc_data(var_key: str):
        """
        Read the NetCDF data for a variable and apply the forest classification mask.
        """
        file_path = os.path.join(inputPath, f"{var_names[var_key]}.nc")
        try:
            ds = xr.open_dataset(file_path)
            data = ds[var_names[var_key]].sel(lat=slice(-60, 90), lon=slice(-180, 180)).copy()
            data.values = np.where(forest_mask == forest_mask[0, 0], np.nan, data.values)
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    # Load the reference variable (NDVI)
    ref_var = read_nc_data(ref_variable)
    if ref_var is None:
        raise ValueError(f"{ref_variable} data cannot be loaded, aborting.")

    # Create latitude bin boundaries
    lat_bins = np.arange(-60, 90, 1)
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

    consistency_summary = {}
    consistency_lat_dict = {"Latitude": lat_centers}

    for var in variables_to_analyze:
        variable = read_nc_data(var)
        if variable is None:
            print(f"{var} data cannot be loaded, skipping.")
            continue

        # Calculate overall consistency
        consistency_percentage = calculate_consistency_percentage(ref_var, variable, var)
        consistency_summary[var] = consistency_percentage

        # Calculate consistency by latitude
        lat_consistency = []
        for bin_lower, bin_upper in zip(lat_bins[:-1], lat_bins[1:]):
            var_bin = variable.sel(lat=slice(bin_lower, bin_upper))
            ref_var_bin = ref_var.sel(lat=slice(bin_lower, bin_upper))

            consistency_bin = calculate_consistency_percentage(ref_var_bin, var_bin, f"{var}-{bin_lower}")
            lat_consistency.append(consistency_bin)

        consistency_lat_dict[var] = lat_consistency

        # Plot the consistency by latitude
        fig, ax = plt.subplots(figsize=(2.5, 5))
        ax.set_ylim(-60, 90)
        ax.set_yticks([-30, 0, 30, 60])
        ax.set_yticklabels(["30°S", "0°", "30°N", "60°N"])
        ax.yaxis.tick_right()
        ax.tick_params(axis='y', direction='in', length=3)
        ax.tick_params(axis='x', direction='in', length=3)

        # Plot consistency curve
        ax.plot(lat_consistency, lat_centers, linestyle='-', color="#009e73")
        ax.fill_betweenx(lat_centers, 0, lat_consistency, color="#009e73", alpha=0.3)

        ax.set_xlabel('Consistency (%)')
        ax.set_xlim(line_xlim)
        ax.set_xticks(x_axis_ticks)
        plt.tight_layout()

        # Save the figure
        fig_path = os.path.join(outputPath, f"{figure_name}_{var}.{export_figure_type}")
        fig.savefig(fig_path, dpi=export_figure_dpi, bbox_inches='tight')
        plt.close(fig)

    # Reset matplotlib global settings
    plt.rcParams.update(rcParamsDefault)

    # Save the consistency summary as a CSV file
    if output_csv_summary:
        summary_df = pd.DataFrame(list(consistency_summary.items()), columns=['Variable', 'Consistency (%)'])
        summary_csv_path = os.path.join(outputPath, "consistency_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Consistency summary saved to {summary_csv_path}")

    # Save the consistency by latitude as a CSV file
    if output_csv_lat:
        lat_df = pd.DataFrame(consistency_lat_dict)
        lat_csv_path = os.path.join(outputPath, "consistency_lat_dict.csv")
        lat_df.to_csv(lat_csv_path, index=False)
        print(f"Consistency by latitude saved to {lat_csv_path}")


# Model evaluation function to assess the agreement between simulated and observed values
def model_evaluation(simulated, observed, outputPath, figure_name, make_text=False):
    """
    Evaluate the model performance by comparing simulated vs observed values and plotting the results.
    """
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(observed, simulated)
    # Calculate predicted values and RMSE
    predicted = intercept + slope * np.array(simulated)
    rmse = get_rmse(observed, predicted)
    ######################################################################################################################
    # Plotting settings
    rcParams['font.size'] = 24
    rcParams['font.family'] = 'Times New Roman'

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(5, 5))

    # Use LogNorm for color mapping
    norm = LogNorm(vmin=1, vmax=1000)  # Set vmin to 1 instead of 0 since log(0) is undefined

    # Plot hexbin plot
    hb = ax.hexbin(observed, simulated, gridsize=100, cmap='viridis', norm=norm)
    cb = fig.colorbar(hb, ax=ax, extend='max', format=LogFormatter(10, labelOnlyBase=False))

    # Custom colorbar formatting for exponent values
    def log_tick_formatter(val, pos=None):
        return f"{int(np.log10(val))}" if val != 0 else "0"

    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(log_tick_formatter))
    cb.set_label('Simple Number / log10(N)')

    # Plot the fit line and y=x line
    fit_line_x = np.array([-10, 10])
    fit_line_y = intercept + slope * fit_line_x
    # ax.plot(fit_line_x, fit_line_y, 'k--', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')
    # ax.plot(fit_line_x, fit_line_x, 'k-', linewidth=2)

    if make_text:
        # Add text annotations with model statistics
        ax.text(0.05, 0.95, f'$y = {slope:.4f}x {"-" if intercept < 0 else "+"}{abs(intercept):.4f}$'
                            f'\n$R^2 = {r_value ** 2:.2f}$, $P {"< 0.01" if p_value < 0.01 else "< 0.05" if p_value < 0.05 else "NS"}$'
                            f'\n$RMSE = {rmse:.2f}$, $SE = {std_err:.2f}$',
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
                style='italic', fontsize=rcParams['font.size'] / 2, fontname=rcParams['font.family']
                )

    # Set axis labels and limits
    ax.set_xlabel('Observed ΔNDVI')
    ax.set_ylabel('Simulated ΔNDVI')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-4, 4)
    ax.set_xticks([-3, 0, 3])
    ax.set_yticks([-3, 0, 3])
    ax.tick_params(axis='both', direction='out', length=3, pad=0)
    # Show or save the figure
    output_figure_path = os.path.join(outputPath, f"{figure_name}.{export_figure_type}")
    create_path(output_figure_path)
    fig.savefig(output_figure_path, dpi=export_figure_dpi, bbox_inches='tight')
    plt.close(fig)

    # Reset matplotlib global settings
    rcParams.update(rcParamsDefault)


def delta_ndvi_between_ft_and_fm(df, classify_col, sub_classify_col, value_col, color, ylabel, xlabel,
                                 outputpath, figure_name):
    """
    Plot split violin plots, grouped by classify_col on the x-axis,
    with sub_classify_col categories split on the left and right sides,
    and with user-defined colors.

    Parameters:
    df: DataFrame, containing the data
    classify_col: str, column name for grouping on the x-axis
    sub_classify_col: str, column name for the left/right split
    value_col: str, column name for values
    color: list, list of colors corresponding to the two categories in sub_classify_col
    xlabel: str, label for the x-axis
    ylabel: str, label for the y-axis
    legend_title: str, title of the legend
    outputpath: str, path to save the output figure
    figure_name: str, file name of the figure
    """
    rcParams['font.size'] = 20
    rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(7, 5))

    # Determine order of categories for x-axis and hue
    order = df[classify_col].unique().tolist()  # Keep the original order from the data
    hue_order = ['IF', 'MF']  # Ensure subcategory order matches the color list

    # Draw split violin plots
    ax = sns.violinplot(
        x=classify_col,
        y=value_col,
        hue=sub_classify_col,
        data=df,
        split=True,
        palette=color,
        order=order,
        hue_order=hue_order,
        cut=0,              # Limit the violin shape to the data range
        inner="quartile",   # Show quartile lines inside
        scale="width",      # Scale violins to have equal area
        bw=0.1,             # Control kernel bandwidth for smoothing
        # alpha=0.5,        # Optionally set fill transparency
        linewidth=1.5,      # Width of violin edges
        linecolor="black",
        saturation=0.75,
        legend=False,
    )

    # Add horizontal reference line
    plt.axhline(y=0, color='black', linestyle='--', zorder=0)

    # Axis labels and limits
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(-1.75, 1.75)
    plt.yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])

    # Legend (optional)
    # handles, labels = ax.get_legend_handles_labels()
    # legend = ax.legend(
    #     handles=handles,
    #     labels=labels,
    #     loc='upper right',
    #     bbox_to_anchor=(1, 1.1),  # Offset from the upper right corner (x, y)
    #     ncol=2,                   # Arrange in two columns
    #     frameon=False,            # Remove background box
    #     borderaxespad=0.3,        # Reduce spacing between legend and axis
    #     handletextpad=0.3,        # Reduce spacing between marker and label
    #     columnspacing=0.5,        # Reduce spacing between columns
    # )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(outputpath, f"{figure_name}.{export_figure_type}"), dpi=export_figure_dpi)
    plt.close()

    # Reset default parameters
    rcParams.update(rcParamsDefault)

