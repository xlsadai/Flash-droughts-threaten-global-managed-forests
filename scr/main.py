# coding=utf-8
import os
import pandas as pd
import numpy as np
import rasterio as rio
import xarray as xr
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import dask

dask.config.set({'dataframe.query-planning': True})
import dask.dataframe as dd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
from Tools.utils import (create_path, single_spatial_distribution, spatial_distribution_mk_test, lat_mean_plot,
                         delta_dv_spatial_distribution, characteristic_lat_mean, delta_ndvi_distribution_plot,
                         hexbin_plot, interaction_effects, main_effect_lines,
                         forest_management_bar, consistency_lat_delta, consistency_bio_lat, model_evaluation,
                         delta_ndvi_between_ft_and_fm,
                         single_spatial_distribution_without_ft, forest_management_practices_bar)


def figure_1abcd():
    data_path = os.path.join(manuscript_data, r"fd_event_stats/")
    outputPath = os.path.join(figure_path, "Figure 1/")
    create_path(outputPath, is_dir=True)
    var_name = [
        "frequency_sum",
        "development_speed",
        "condition_duration",
        "condition_peak",
    ]
    figure_name = {
        "frequency_sum": "Figure 1 a",
        "development_speed": "Figure 1 b",
        "condition_duration": "Figure 1 c",
        "condition_peak": "Figure 1 d",
    }
    colorbar_ticks = {
        "condition_peak": (-2.1, -1.9, -1.7, -1.5, -1.3),
        "condition_duration": (3, 4, 5, 6, 7, 8),
        "development_speed": (-1.2, -1, -0.8, -0.6, -0.4),
        "frequency_sum": (1, 2, 3, 4, 5),
    }
    colorbar_label = {
        "condition_peak": "Peak Stress /dimensionless",
        "condition_duration": "Stress Duration /Pentads",
        "development_speed": "Onset Rate /Pentad⁻¹",  # ⁻ ⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹ (U+2070 到 U+2079)
        "frequency_sum": "Frequency /Events per decade",
    }
    level = {
        "condition_peak": (-2.1, -1.28),
        "condition_duration": (3, 8),
        "development_speed": (-1, -0.4),
        "frequency_sum": (1, 5),
    }
    cmap = {
        "condition_peak": LinearSegmentedColormap.from_list("red_cmap",
                                                            [(0, '#d7191cff'), (0.75, '#ffffbfff'), (1, 'white')]),
        "condition_duration": LinearSegmentedColormap.from_list("blue_cmap",
                                                                [(0, 'white'), (0.25, '#bdd7e7'), (1, '#08306b')]),
        "development_speed": LinearSegmentedColormap.from_list("green_cmap",
                                                               [(0, '#00441b'), (0.75, '#a6cee3'), (1, 'white')]),
        "frequency_sum": LinearSegmentedColormap.from_list("colorblind_cmap",
                                                           ["#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"]
                                                           # RdYlBu
                                                           ),
    }
    extend = {
        "condition_peak": "min",
        "condition_duration": "max",
        "development_speed": "min",
        "frequency_sum": "max",
    }
    boundary_labels = {  # [left_labels, top_labels, bottom_labels, right_labels]
        "condition_peak": [True, True, False, False],
        "condition_duration": [True, True, False, False],
        "development_speed": [True, True, False, False],
        "frequency_sum": [True, True, False, False],
    }
    for var in var_name:
        print(f"Plot ==> {figure_name[var]}", end="")
        inputPath = data_path + f"{var}.nc"
        single_spatial_distribution(inputPath=inputPath,
                                    outputPath=outputPath,
                                    var_name=var,
                                    level=level[var],
                                    colorbar_ticks=colorbar_ticks[var],
                                    colorbar_label=colorbar_label[var],
                                    figure_name=figure_name[var],
                                    cmap=cmap[var],
                                    extend=extend[var],
                                    boundary_labels=boundary_labels[var]
                                    )
        print(" >>>>>>> Finish !")


def supplementary_figure_4():
    data_path = os.path.join(manuscript_data, r"fd_event_stats/")
    outputPath = os.path.join(figure_path, "Supplementary Figure 4/")
    create_path(outputPath, is_dir=True)
    var_name = [
        "frequency_sum",
    ]
    figure_name = {
        "frequency_sum": "Supplementary Figure 4",
    }
    colorbar_ticks = {
        "frequency_sum": (1, 2, 3, 4, 5),
    }
    colorbar_label = {
        "frequency_sum": "Frequency /Events per decade",
    }
    level = {
        "frequency_sum": (1, 5),
    }
    cmap = {
        "frequency_sum": LinearSegmentedColormap.from_list("colorblind_cmap",
                                                           ["#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"]
                                                           # RdYlBu
                                                           ),
    }
    extend = {
        "frequency_sum": "max",
    }
    boundary_labels = {  # [left_labels, top_labels, bottom_labels, right_labels]
        "frequency_sum": [True, True, False, False],
    }
    for var in var_name:
        print(f"Plot ==> {figure_name[var]}", end="")
        inputPath = data_path + f"{var}.nc"
        single_spatial_distribution_without_ft(inputPath=inputPath,
                                               outputPath=outputPath,
                                               var_name=var,
                                               level=level[var],
                                               colorbar_ticks=colorbar_ticks[var],
                                               colorbar_label=colorbar_label[var],
                                               figure_name=figure_name[var],
                                               cmap=cmap[var],
                                               extend=extend[var],
                                               boundary_labels=boundary_labels[var]
                                               )
        print(" >>>>>>> Finish !")


def supplementary_figure_3abcd():
    data_path = os.path.join(manuscript_data, r"fd_event_stats_sm/")
    outputPath = os.path.join(figure_path, "Supplementary Figure 3/")
    create_path(outputPath, is_dir=True)
    var_name = [
        "frequency_sum",
        "development_speed",
        "condition_duration",
        "condition_peak",
    ]
    figure_name = {
        "frequency_sum": "Supplementary Figure 3 a",
        "development_speed": "Supplementary Figure 3 b",
        "condition_duration": "Supplementary Figure 3 c",
        "condition_peak": "Supplementary Figure 3 d",
    }
    colorbar_ticks = {
        "condition_peak": (5, 6, 7, 8, 9, 10),
        "condition_duration": (3, 4, 5, 6, 7, 8),
        "development_speed": (-25, -20, -15, -10, -5),
        "frequency_sum": (1, 3, 5, 7, 9, 12),
    }
    colorbar_label = {
        "condition_peak": "Peak Stress /%",
        "condition_duration": "Stress Duration /Pentads",
        "development_speed": "Onset Rate /Pentad⁻¹",  # ⁻ ⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹ (U+2070 到 U+2079)
        "frequency_sum": "Frequency /Events per decade",
    }
    level = {
        "condition_peak": (5, 10),
        "condition_duration": (3, 8),
        "development_speed": (-25, -5),
        "frequency_sum": (1, 12),
    }
    cmap = {
        "condition_peak": LinearSegmentedColormap.from_list("red_cmap",
                                                            [(0, '#d7191cff'), (0.75, '#ffffbfff'), (1, 'white')]),
        "condition_duration": LinearSegmentedColormap.from_list("blue_cmap",
                                                                [(0, 'white'), (0.25, '#bdd7e7'), (1, '#08306b')]),
        "development_speed": LinearSegmentedColormap.from_list("green_cmap",
                                                               [(0, '#00441b'), (0.75, '#a6cee3'), (1, 'white')]),
        "frequency_sum": LinearSegmentedColormap.from_list("colorblind_cmap",
                                                           ["#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"]
                                                           # RdYlBu
                                                           ),
    }
    extend = {
        "condition_peak": "min",
        "condition_duration": "max",
        "development_speed": "min",
        "frequency_sum": "max",
    }
    boundary_labels = {  # [left_labels, top_labels, bottom_labels, right_labels]
        "condition_peak": [True, True, False, False],
        "condition_duration": [True, True, False, False],
        "development_speed": [True, True, False, False],
        "frequency_sum": [True, True, False, False],
    }
    for var in var_name:
        print(f"Plot ==> {figure_name[var]}", end="")
        inputPath = data_path + f"{var}.nc"
        single_spatial_distribution(inputPath=inputPath,
                                    outputPath=outputPath,
                                    var_name=var,
                                    level=level[var],
                                    colorbar_ticks=colorbar_ticks[var],
                                    colorbar_label=colorbar_label[var],
                                    figure_name=figure_name[var],
                                    cmap=cmap[var],
                                    extend=extend[var],
                                    boundary_labels=boundary_labels[var]
                                    )
        print(" >>>>>>> Finish !")


def figure_1e():
    data_path = os.path.join(manuscript_data, r"fd_event_stats/")
    outputPath = os.path.join(figure_path, "Figure 1/")
    figure_name = {
        "Figure 1 e"
    }
    input_path = {
        "Figure 1 e": (
            data_path + f"frequency_sum.nc",
            data_path + f"condition_peak.nc",
            data_path + f"condition_duration.nc",
            data_path + f"development_speed.nc",
        )
    }
    var_name = {
        "Figure 1 e": ("frequency_sum", "condition_peak", "condition_duration", "development_speed")
    }
    colorbar_label = {
        "Figure 1 e": ("Frequency", "Peak Stress", "Stress Duration", "Onset Rate"),
    }
    line_colors = {
        "Figure 1 e": (
            ("#e69f00", "#ffffbfff"), ("#d55e00", "#fb7050"), ("#0072b2", "#529dcc"), ("#009e73", "#55b567"))
    }
    line_ticks = {
        "Figure 1 e": ((1, 10, 18), (-1.7, -1.9, -2.1), (3, 4, 5, 6), (-0.6, -0.8, -1))
    }
    line_xlim = {
        "Figure 1 e": ((1, 18), (-1.7, -2.1), (3, 6), (-0.6, -1))
    }
    for fig in figure_name:
        print(f"Plot ==> {fig}", end="")
        vars = var_name[fig]
        lat_mean_plot(inputPath=input_path[fig],
                      outputPath=outputPath,
                      var_name=vars,
                      colorbar_label=colorbar_label[fig],
                      figure_name=fig,
                      line_ticks=line_ticks[fig],
                      line_colors=line_colors[fig],
                      line_xlim=line_xlim[fig]
                      )
        print(" >>>>>>> Finish !")


def supplementary_figure_3_e():
    data_path = os.path.join(manuscript_data, r"fd_event_stats_sm/")
    outputPath = os.path.join(figure_path, "Supplementary Figure 3/")
    figure_name = {
        "Supplementary Figure 3 e"
    }
    input_path = {
        "Supplementary Figure 3 e": (
            data_path + f"frequency_sum.nc",
            data_path + f"condition_peak.nc",
            data_path + f"condition_duration.nc",
            data_path + f"development_speed.nc",
        )
    }
    var_name = {
        "Supplementary Figure 3 e": ("frequency_sum", "condition_peak", "condition_duration", "development_speed")
    }
    colorbar_label = {
        "Supplementary Figure 3 e": ("Frequency", "Peak Stress", "Stress Duration", "Onset Rate"),
    }
    line_colors = {
        "Supplementary Figure 3 e": (
            ("#e69f00", "#ffffbfff"), ("#d55e00", "#fb7050"), ("#0072b2", "#529dcc"), ("#009e73", "#55b567"))
    }
    line_ticks = {
        "Supplementary Figure 3 e": ((1, 20, 40), (10, 7.5, 5), (4, 5, 6, 7), (-5, -17.5, -30))
    }
    line_xlim = {
        "Supplementary Figure 3 e": ((1, 40), (10, 5), (4, 7), (-5, -30))
    }

    for fig in figure_name:
        print(f"Plot ==> {fig}", end="")
        vars = var_name[fig]
        lat_mean_plot(inputPath=input_path[fig],
                      outputPath=outputPath,
                      var_name=vars,
                      colorbar_label=colorbar_label[fig],
                      figure_name=fig,
                      line_ticks=line_ticks[fig],
                      line_colors=line_colors[fig],
                      line_xlim=line_xlim[fig]
                      )
        print(" >>>>>>> Finish !")


def figure_2a():
    data_path = os.path.join(manuscript_data, r"vd_stats/")
    outputPath = os.path.join(figure_path, "Figure 2/")
    create_path(outputPath, is_dir=True)
    var_name = [
        "delta_ndvi_avg",
    ]
    figure_name = {
        "delta_ndvi_avg": "Figure 2 a",

    }
    colorbar_label = {
        "delta_ndvi_avg": "ΔNDVI",
    }
    cmap = {
        "delta_ndvi_avg": ["#f0e442", "#009e73", "#d55e00", "#56b4e9"],
    }
    extend = {
        "delta_ndvi_avg": "neither",
    }
    for var in var_name:
        print(f"Plot ==> {figure_name[var]}", end="")
        inputPath = data_path + f"{var}.nc"
        delta_dv_spatial_distribution(inputPath=inputPath,
                                      outputPath=outputPath,
                                      var_name=var,
                                      colorbar_label=colorbar_label[var],
                                      figure_name=figure_name[var],
                                      cmap=cmap[var],
                                      extend=extend[var]
                                      )
        print(" >>>>>>> Finish !")


def supplementary_figure_6_a():
    data_path = os.path.join(manuscript_data, r"vd_stats/")
    outputPath = os.path.join(figure_path, "Supplementary Figure 6/")
    create_path(outputPath, is_dir=True)
    var_name = [
        "delta_ndvi_avg_sm",
    ]
    figure_name = {
        "delta_ndvi_avg_sm": "Supplementary Figure 6 a",

    }
    colorbar_label = {
        "delta_ndvi_avg_sm": "ΔNDVI",
    }
    cmap = {
        "delta_ndvi_avg_sm": ["#f0e442", "#009e73", "#d55e00", "#56b4e9"],
    }
    extend = {
        "delta_ndvi_avg_sm": "neither",
    }
    for var in var_name:
        print(f"Plot ==> {figure_name[var]}", end="")
        inputPath = data_path + f"{var}.nc"
        delta_dv_spatial_distribution(inputPath=inputPath,
                                      outputPath=outputPath,
                                      var_name=var,
                                      colorbar_label=colorbar_label[var],
                                      figure_name=figure_name[var],
                                      cmap=cmap[var],
                                      extend=extend[var]
                                      )
        print(" >>>>>>> Finish !")


def figure_2b():
    data_path = os.path.join(manuscript_data, f"vd_stats", "delta_ndvi_avg.nc")
    outputPath = os.path.join(figure_path, "Figure 2/")
    create_path(outputPath, is_dir=True)

    data = xr.open_dataset(data_path)["delta_ndvi_avg"]
    data = data.values.astype(float)
    with rio.open(r"../dataFile/ForestCharacteristics/ForestManagement.tif") as ds:
        ifl = ds.read(1)[::-1].astype(float)
        ifl[ifl == ifl[0, 0]] = np.nan
    data[ifl == ifl[0, 0]] = np.nan
    data[ifl == 40] = np.nan
    df = pd.DataFrame({classify_col: ifl.flatten(), object_val: data.flatten()})
    df = df.dropna(subset=[classify_col, object_val])
    df[classify_col] = df[classify_col].apply(lambda x: 1 if x in [11] else 2 if ~np.isnan(x) else x)
    print(f"Plot ==> Figure 2 b", end="")
    datas = [
        df[object_val].values,
        df[object_val][df[classify_col] == 1].values,
        df[object_val][df[classify_col] == 2].values
    ]
    classify_name = ["Forest", "IF", "MF"]
    colors = ['#8b4513', '#3fa95b', "#3f8fc5"]
    delta_ndvi_distribution_plot(datas, classify_name, colors, outputPath, f"Figure 2 b",
                                 "ΔNDVI")
    print(" >>>>>>> Finish !")


def supplementary_figure_6_b():
    data_path = os.path.join(manuscript_data, f"vd_stats", "delta_ndvi_avg_sm.nc")
    outputPath = os.path.join(figure_path, "Supplementary Figure 6/")
    create_path(outputPath, is_dir=True)

    data = xr.open_dataset(data_path)["delta_ndvi_avg_sm"]
    data = data.values.astype(float)
    with rio.open(r"../dataFile/ForestCharacteristics/ForestManagement.tif") as ds:
        ifl = ds.read(1)[::-1].astype(float)
        ifl[ifl == ifl[0, 0]] = np.nan
    data[ifl == ifl[0, 0]] = np.nan
    df = pd.DataFrame({classify_col: ifl.flatten(), object_val: data.flatten()})
    df = df.dropna(subset=[classify_col, object_val])
    df[classify_col] = df[classify_col].apply(lambda x: 1 if x in [11] else 2 if ~np.isnan(x) else x)
    print(f"Plot ==> Supplementary Figure 6 b", end="")
    datas = [
        df[object_val].values,
        df[object_val][df[classify_col] == 1].values,
        df[object_val][df[classify_col] == 2].values
    ]
    classify_name = ["Forest", "IF", "MF"]
    colors = ['#8b4513', '#3fa95b', "#3f8fc5"]
    delta_ndvi_distribution_plot(datas, classify_name, colors, outputPath, f"Supplementary Figure 6 b",
                                 "ΔNDVI")
    print(" >>>>>>> Finish !")


def figure_2c():
    pPath = os.path.join(manuscript_data, r"Climate/total_precipitation.tif")
    tPath = os.path.join(manuscript_data, r"Climate/2m_temperature.tif")
    xlabel = "MAP /mm"
    ylabel = "MAT /℃"
    data_path = os.path.join(manuscript_data, r"vd_stats/")
    outputPath = os.path.join(figure_path, "Figure 2/")
    var_name = [
        "delta_ndvi_avg",
    ]
    figure_name = {
        "delta_ndvi_avg": "Figure 2 c",
    }
    colorbar_ticks = {
        "delta_ndvi_avg": (-1, -0.5, 0, 0.5, 1),
    }
    colorbar_label = {
        "delta_ndvi_avg": "ΔNDVI",
    }
    level = {
        "delta_ndvi_avg": (-1, 1),
    }
    cmap = {
        "delta_ndvi_avg": LinearSegmentedColormap.from_list("blue_cmap",
                                                            [(0, "#ff7f0e"), (1 / 2, "white"),
                                                             (1, '#1f77b4')]),
    }

    for var in var_name:
        print(f"Plot ==> {figure_name[var]}", end="")
        inputPath = data_path + f"{var}.nc"
        hexbin_plot(pPath=pPath,
                    tPath=tPath,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    inputPath=inputPath,
                    outputPath=outputPath,
                    var_name=var,
                    level=level[var],
                    colorbar_ticks=colorbar_ticks[var],
                    colorbar_label=colorbar_label[var],
                    figure_name=figure_name[var],
                    cmap=cmap[var]
                    )
        print(" >>>>>>> Finish !")


def supplementary_figure_6_c():
    pPath = os.path.join(manuscript_data, r"Climate/total_precipitation.tif")
    tPath = os.path.join(manuscript_data, r"Climate/2m_temperature.tif")
    xlabel = "MAP /mm"
    ylabel = "MAT /℃"
    data_path = os.path.join(manuscript_data, r"vd_stats/")
    outputPath = os.path.join(figure_path, "Supplementary Figure 6/")
    var_name = [
        "delta_ndvi_avg_sm",
    ]
    figure_name = {
        "delta_ndvi_avg_sm": "Supplementary Figure 6 c",
    }
    colorbar_ticks = {
        "delta_ndvi_avg_sm": (-0.5, -0.25, 0, 0.25, 0.5),
    }
    colorbar_label = {
        "delta_ndvi_avg_sm": "ΔNDVI",
    }
    level = {
        "delta_ndvi_avg_sm": (-0.5, 0.5),
    }
    cmap = {
        "delta_ndvi_avg_sm": LinearSegmentedColormap.from_list("blue_cmap",
                                                               [(0, "#ff7f0e"), (1 / 2, "white"),
                                                                (1, '#1f77b4')]),
    }

    for var in var_name:
        print(f"Plot ==> {figure_name[var]}", end="")
        inputPath = data_path + f"{var}.nc"
        hexbin_plot(pPath=pPath,
                    tPath=tPath,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    inputPath=inputPath,
                    outputPath=outputPath,
                    var_name=var,
                    level=level[var],
                    colorbar_ticks=colorbar_ticks[var],
                    colorbar_label=colorbar_label[var],
                    figure_name=figure_name[var],
                    cmap=cmap[var]
                    )
        print(" >>>>>>> Finish !")


def figure_3():
    outputPath = os.path.join(figure_path, "Figure 3/")
    create_path(outputPath, is_dir=True)
    figure_name = f"Figure 3"
    fdc_factors = [
        "ΔTa", "ΔP",
        "MAT", "MAP",
        "Age", "CH", "MRD", "Den",
        "OR (-)",
        "PS (-)",
        "SD",
    ]
    classify_factors = [
        "FM"
    ]
    labels = {
        "MAT": "MAT /℃",
        "MAP": "MAP /10² mm",
        "FM": "Forest management",
        "ΔP": "ΔP",
        "ΔTa": "ΔTa",
        "OR (-)": "OR (-) /Pentad⁻¹",  # ⁻ ⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹ (U+2070 到 U+2079)
        "SD": "SD /Pentads",
        "PS (-)": "PS (-)",
        "Age": "Age /years",
        "CH": "CH /m",
        "MRD": "MRD /m",
        "Den": "Den /10⁴ trees per grid"
    }
    y_lims = [-0.1, 0.1]
    y_ticks = [-0.1, -0.05, 0, 0.05, 0.1]
    x_lims = {
        "MAT": [-12, 30],
        "MAP": [3, 40],
        "ΔP": [-1.2, 0.3],
        "ΔTa": [-1, 2.3],
        "OR (-)": [0.4, 1.9],
        "SD": [3, 17],
        "PS (-)": [1.3, 3],
        "Age": [0, 250],
        "CH": [4, 40],
        "MRD": [0, 15],
        "Den": [0, 18],
    }
    scatter_size = {
        "OR (-)": 0.001,
        "SD": 1,
        "PS (-)": 0.005,
        "ΔP": 0.001,
        "ΔTa": 0.001,
        "MAT": 0.001,
        "MAP": 0.001,
        "Age": 0.001,
        "CH": 0.001,
        "MRD": 0.001,
        "Den": 0.001,
    }
    colorbar_ticks = {
        "MAT": [0, 5, 10, 15, 20, 25],
        "FM": [1, 2],
        # "FM": [11, 20, 31, 32, 40, 53]
    }
    colorbar_extend = {
        "FM": "neither",
    }
    cmap = {
        "FM": ListedColormap(['#C7E5CD', "#C5DEED"]),
    }
    colors = {
        "FM": {"IF": "#00441b", "MF": "#2b83ba"},
    }
    group_size = {
        "ΔP": 0.1,
        "ΔTa": 0.2,
        "OR (-)": 0.1,
        "SD": 1,
        "PS (-)": 0.1,
        "MAT": 3,
        "MAP": 2,
        "Age": 10,
        "CH": 1,
        "MRD": 1,
        "Den": 1,
    }
    ver_lines = {
        "ΔP": -0.66,
        "ΔTa": 0.62,
        "OR (-)": None,
        "SD": None,
        "PS (-)": 1.8,
        "MAT": None,
        "MAP": None,
        "Age": None,
        "CH": 23,
        "MRD": None,
        "Den": None,
    }
    shap_values_path = os.path.join(manuscript_data, r"shap_output", "ΔNDVI shap stats.csv")
    df = dd.read_csv(shap_values_path)
    df = df.compute()
    df["FM_Classify"] = df["FM"].replace({1: 'IF', 2: 'MF'})
    figure_name_sub = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
    n = 0
    for interaction_feature in classify_factors:
        for main_feature in fdc_factors:
            sub_df = df[[main_feature, f"{main_feature} vs {interaction_feature}",
                         interaction_feature, f"{interaction_feature}_Classify"]]
            print(f"Plot ==> {figure_name} {figure_name_sub[n]}", end="")
            interaction_effects(sub_df, main_feature=main_feature, interaction_feature=interaction_feature,
                                classify_col=f"{interaction_feature}_Classify",
                                colormaps=cmap[interaction_feature],
                                scatter_size=scatter_size[main_feature],
                                line_colors=colors[interaction_feature],
                                colorbar_ticks=colorbar_ticks[interaction_feature],
                                colorbar_extend=colorbar_extend[interaction_feature],

                                x_label=labels[main_feature], colorbar_label=labels[interaction_feature],

                                x_lims=x_lims[main_feature],
                                y_lims=y_lims, y_ticks=y_ticks,

                                group_size=group_size[main_feature],

                                outputPath=outputPath, figure_name=f"{figure_name} {figure_name_sub[n]}",
                                ver_lines=ver_lines[main_feature])
            n += 1
            print(" >>>>>>> Finish !")


def supplementary_figure_12():
    outputpath = os.path.join(figure_path, f"Supplementary Figure 12")
    for mode, num in zip(["train", "vali"], ["a", "b"]):
        print(f"Plot ==> Supplementary Figure 12 {num}", end="")
        inputpath = os.path.join(manuscript_data, "shap_output", f"ΔNDVI_model_output({mode}).csv")
        df = dd.read_csv(inputpath, usecols=["Simulated", "Observed"])
        df = df.compute()
        df = df.dropna(how="any", axis=0)
        model_evaluation(df["Simulated"].values, df["Observed"], outputPath=outputpath,
                         figure_name=f"Supplementary Figure 12 {num}")
        print(" >>>>>>> Finish !")


def supplementary_figure_1_ace():
    data_path = os.path.join(manuscript_data, r"fd_event_stats/")
    outputPath = os.path.join(figure_path, f"Supplementary Figure 1")
    create_path(outputPath, is_dir=True)
    var_name = [
        "development_speed_sen",
        "condition_duration_sen",
        "condition_peak_sen",
    ]
    figure_name = {
        "development_speed_sen": "Supplementary Figure 1 a",
        "condition_duration_sen": "Supplementary Figure 1 c",
        "condition_peak_sen": "Supplementary Figure 1 e",
    }
    colorbar_ticks = {
        "condition_peak_sen": (-0.1, -0.05, 0, 0.05, 0.1),
        "condition_duration_sen": (-0.1, -0.05, 0, 0.05, 0.1),
        "development_speed_sen": (-0.1, -0.05, 0, 0.05, 0.1),
    }
    colorbar_label = {
        "condition_peak_sen": "Peak Stress /10yr⁻¹",  # ⁻ ⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹ (U+2070 到 U+2079)
        "condition_duration_sen": "Stress Duration /Pentads·10yr⁻¹",
        "development_speed_sen": "Onset Rate /Pentad⁻¹·10yr⁻¹"
    }
    level = {
        "condition_peak_sen": (-0.1, 0.1),
        "condition_duration_sen": (-0.1, 0.1),
        "development_speed_sen": (-0.1, 0.1),
    }
    cmap = {
        "condition_peak_sen": LinearSegmentedColormap.from_list("blue_cmap",
                                                                [(0, '#67000dcc'),
                                                                 (0.2, "#cc666699"),
                                                                 (0.4, "#ff999980"),
                                                                 (1 / 2, "#ffffff80"),
                                                                 (0.6, "#ffd8b199"),
                                                                 (0.8, "#ffe680b3"),
                                                                 (1, "#ffea46cc")]),
        "condition_duration_sen": LinearSegmentedColormap.from_list("blue_cmap",
                                                                    [(0, "#08306bcc"),
                                                                     (0.2, "#2e4f8ab3"),
                                                                     (0.4, "#4a74a899"),
                                                                     (1 / 2, "#ffffff80"),
                                                                     (0.6, "#8b5fae80"),
                                                                     (0.8, "#70407099"),
                                                                     (1, '#912b81cc')]),
        "development_speed_sen": LinearSegmentedColormap.from_list("blue_cmap",
                                                                   [(0, "#d35400cc"),
                                                                    (0.2, "#e67e22b3"),
                                                                    (0.4, "#f39c12a0"),
                                                                    (1 / 2, "#ffffff80"),
                                                                    (0.6, "#87a37c80"),
                                                                    (0.8, "#5f7a5e99"),
                                                                    (1, '#00441bcc')]),
    }
    extend = {
        "condition_peak_sen": "both",
        "condition_duration_sen": "both",
        "development_speed_sen": "both",
    }
    for var in var_name:
        print(f"Plot ==> {figure_name[var]}", end="")
        inputPath = os.path.join(data_path, f"{var}.nc")
        single_spatial_distribution(inputPath=inputPath,
                                    outputPath=outputPath,
                                    var_name=var,
                                    level=level[var],
                                    colorbar_ticks=colorbar_ticks[var],
                                    colorbar_label=colorbar_label[var],
                                    figure_name=figure_name[var],
                                    cmap=cmap[var],
                                    extend=extend[var]
                                    )
        print(" >>>>>>> Finish !")


def supplementary_figure_1_bdf():
    data_path = os.path.join(manuscript_data, r"fd_event_stats/")
    outputPath = os.path.join(figure_path, f"Supplementary Figure 1")
    var_name = [
        "development_speed_mk",
        "condition_duration_mk",
        "condition_peak_mk",
    ]
    figure_name = {
        "development_speed_mk": "Supplementary Figure 1 b",
        "condition_duration_mk": "Supplementary Figure 1 d",
        "condition_peak_mk": "Supplementary Figure 1 f",
    }
    colorbar_ticks = [(-999 - 2.58) / 2, (-2.58 - 1.96) / 2, 0, (2.58 + 1.96) / 2, (999 + 2.58) / 2]
    tick_label = [
        "Extremely significant decreasing",
        "Significantly decrease",
        "No significant",
        "Significantly increase",
        "Extremely significant increasing"]
    colorbar_label = "Mann-Kendall Test"
    level = [-999, -2.58, -1.96, 1.96, 2.58, 999]
    cmap = {
        "condition_peak_mk": ['#d55e00', "#e69f00", '#FFFFFF00', '#56b4e9', '#009e73'],
        "condition_duration_mk":  ['#d55e00', "#e69f00", '#FFFFFF00', '#56b4e9', '#009e73'],
        "development_speed_mk":  ['#d55e00', "#e69f00", '#FFFFFF00', '#56b4e9', '#009e73'],
    }
    for var in var_name:
        print(f"Plot ==> {figure_name[var]}", end="")
        inputPath = data_path + f"{var}.nc"
        spatial_distribution_mk_test(data_path=inputPath,
                                     outputPath=outputPath,
                                     var_name=var,
                                     level=level,
                                     colorbar_ticks=colorbar_ticks,
                                     tick_label=tick_label,
                                     colorbar_label=colorbar_label,
                                     figure_name=figure_name[var],
                                     colors=cmap[var],
                                     extend="neither"
                                     )
        print(" >>>>>>> Finish !")


def supplementary_figure_1_g():
    data_path = os.path.join(manuscript_data, r"fd_event_stats/")
    outputPath = os.path.join(figure_path, f"Supplementary Figure 1")
    figure_name = {
        "Supplementary Figure 1 g"
    }
    input_path = {
        "Supplementary Figure 1 g": (
            data_path + f"condition_peak_sen.nc",
            data_path + f"condition_duration_sen.nc",
            data_path + f"development_speed_sen.nc",
        )
    }
    var_name = {
        "Supplementary Figure 1 g": ("condition_peak_sen", "condition_duration_sen", "development_speed_sen")
    }
    colorbar_label = {
        "Supplementary Figure 1 g": ("Peak Stress", "Stress Duration", "Onset Rate"),
    }
    line_colors = {
        "Supplementary Figure 1 g": (("#d55e00", "#fb7050"), ("#0072b2", "#529dcc"), ("#009e73", "#55b567"))
    }
    line_ticks = {
        "Supplementary Figure 1 g": ((-0.2, 0, 0.2), (-0.4, 0, 0.4), (-0.2, 0, 0.2))
    }
    line_xlim = {
        "Supplementary Figure 1 g": ((0.25, -0.25), (-0.45, 0.45), (0.25, -0.25))
    }
    for fig in figure_name:
        print(f"Plot ==> {fig}", end="")
        vars = var_name[fig]
        characteristic_lat_mean(inputPath=input_path[fig],
                                outputPath=outputPath,
                                var_name=vars,
                                colorbar_label=colorbar_label[fig],
                                figure_name=fig,
                                line_ticks=line_ticks[fig],
                                line_colors=line_colors[fig],
                                line_xlim=line_xlim[fig]
                                )
        print(" >>>>>>> Finish !")


def supplementary_figure_2_ab():
    data_path = os.path.join(manuscript_data, r"fd_event_stats/")
    outputPath = os.path.join(figure_path, f"Supplementary Figure 2")
    create_path(outputPath, is_dir=True)
    var_name = [
        "delta_ta",
        "delta_p",
    ]
    figure_name = {
        "delta_ta": "Supplementary Figure 2 a",
        "delta_p": "Supplementary Figure 2 b",

    }
    colorbar_ticks = {
        "delta_p": (-0.8, -0.6, -0.4, -0.2, 0, 0.2),
        "delta_ta": (-0.5, -0, 0.5, 1, 1.5),
    }
    colorbar_label = {
        "delta_p": "Normalized ΔP",
        "delta_ta": "Normalized ΔTa",
    }
    level = {
        "delta_p": (-0.8, 0.2),
        "delta_ta": (-0.5, 1.5),
    }
    cmap = {
        "delta_p": LinearSegmentedColormap.from_list("blue_cmap",
                                                     [(0, '#7b3294ff'),
                                                      (0.3, '#c2a5cf80'),
                                                      (0.5, '#add8e660'),
                                                      (0.8, 'white'),
                                                      (1, '#a6dbaaff')]),
        "delta_ta": LinearSegmentedColormap.from_list("blue_cmap",
                                                      [(0, "#2b83baff"), (0.25, "white"), (0.5, '#ffffbfff'),
                                                       (0.75, '#fdae61ff'), (1, '#d7191cff')]),
    }
    extend = {
        "delta_p": "both",
        "delta_ta": "both",
    }
    boundary_labels = {  # [left_labels, top_labels, bottom_labels, right_labels]
        "delta_p": [False, True, False, True],
        "delta_ta": [True, True, False, False],
    }
    for var in var_name:
        print(f"Plot ==> {figure_name[var]}", end="")
        inputPath = data_path + f"{var}.nc"
        single_spatial_distribution(inputPath=inputPath,
                                    outputPath=outputPath,
                                    var_name=var,
                                    level=level[var],
                                    colorbar_ticks=colorbar_ticks[var],
                                    colorbar_label=colorbar_label[var],
                                    figure_name=figure_name[var],
                                    cmap=cmap[var],
                                    extend=extend[var],
                                    boundary_labels=boundary_labels[var]
                                    )
        print(" >>>>>>> Finish !")


def supplementary_figure_2_c():
    data_path = os.path.join(manuscript_data, r"fd_event_stats/")
    outputPath = os.path.join(figure_path, "Supplementary Figure 2")
    create_path(outputPath, is_dir=True)

    file_name = {
        "delta_p": "delta_p",
        "delta_ta": "delta_ta",
    }

    colors = {
        "delta_p": "#800080",
        "delta_ta": "#FF4500",
        "both": "#1E90FF",
    }

    consistency_lat_delta(outputPath=outputPath, inputPath=data_path, var_name=file_name,
                          figure_name="Supplementary Figure 2c", x_axis_ticks=(0, 40, 80),
                          bar_xlim=(0, 85), bar_colors=colors)


def supplementary_figure_5_abc():
    data_path = os.path.join(manuscript_data, r"vd_stats/")
    outputPath = os.path.join(figure_path, f"Supplementary Figure 5")
    create_path(outputPath, is_dir=True)
    var_name = [
        "delta_ndvi_avg",
        "delta_lai_avg",
        "delta_sif_avg",
    ]
    figure_name = {
        "delta_ndvi_avg": "Supplementary Figure 5 a",
        "delta_lai_avg": "Supplementary Figure 5 b",
        "delta_sif_avg": "Supplementary Figure 5 c",
    }
    colorbar_ticks = {
        "delta_ndvi_avg": (-1, -0.5, 0, 0.5, 1),
        "delta_lai_avg": (-1, -0.5, 0, 0.5, 1),
        "delta_sif_avg": (-1, -0.5, 0, 0.5, 1),
    }
    colorbar_label = {
        "delta_ndvi_avg": "ΔNDVI",
        "delta_lai_avg": "ΔLAI",
        "delta_sif_avg": "ΔSIF",
    }
    level = {
        "delta_ndvi_avg": (-1, 1),
        "delta_lai_avg": (-1, 1),
        "delta_sif_avg": (-1, 1),
    }
    cmap = {
        "delta_ndvi_avg": LinearSegmentedColormap.from_list("blue_cmap",
                                                            [(0, '#cc79a7'), (1 / 2, "white"),
                                                             (1, "#009e73")]),
        "delta_lai_avg": LinearSegmentedColormap.from_list("blue_cmap",
                                                            [(0, '#cc79a7'), (1 / 2, "white"),
                                                             (1, "#009e73")]),
        "delta_sif_avg": LinearSegmentedColormap.from_list("blue_cmap",
                                                            [(0, '#cc79a7'), (1 / 2, "white"),
                                                             (1, "#009e73")]),
    }
    extend = {
        "delta_ndvi_avg": "both",
        "delta_lai_avg": "both",
        "delta_sif_avg": "both",
    }
    for var in var_name:
        print(f"Plot ==> {figure_name[var]}", end="")
        inputPath = data_path + f"{var}.nc"
        single_spatial_distribution(inputPath=inputPath,
                                    outputPath=outputPath,
                                    var_name=var,
                                    level=level[var],
                                    colorbar_ticks=colorbar_ticks[var],
                                    colorbar_label=colorbar_label[var],
                                    figure_name=figure_name[var],
                                    cmap=cmap[var],
                                    extend=extend[var],
                                    )
        print(" >>>>>>> Finish !")


def supplementary_figure_5_de():
    data_path = os.path.join(manuscript_data, r"vd_stats/")
    outputPath = os.path.join(figure_path, f"Supplementary Figure 5")
    create_path(outputPath, is_dir=True)
    var_name = {
        "NDVI": "delta_ndvi_avg",
        "LAI": "delta_lai_avg",
        "SIF": "delta_sif_avg",
    }

    print(f"Plot ==> Supplementary Figure 5 d and e", end="")
    consistency_bio_lat(outputPath=outputPath, inputPath=data_path, var_names=var_name,
                        figure_name="Supplementary Figure 5", x_axis_ticks=(50, 80),
                        line_xlim=(40, 85))
    print(" >>>>>>> Finish !")


def supplementary_figure_8():
    outputPath = os.path.join(figure_path, f"Supplementary Figure 8")
    create_path(outputPath, is_dir=True)
    X_cols = [
        "Age", "CH", "MRD", "Den",
        "ΔTa", "ΔP",
        "MAT", "MAP",
        "OR (-)", "SD", "PS (-)",
    ]
    X_labels = {
        "MAT": "MAT /℃",
        "MAP": "MAP /mm",
        "ΔP": "ΔP",
        "ΔTa": "ΔTa",
        "OR (-)": "OR (-) /Pentad⁻¹",  # ⁻ ⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹ (U+2070 到 U+2079)
        "SD": "SD /Pentads",
        "PS (-)": "PS (-)",
        "Age": "Age /years",
        "CH": "CH /m",
        "MRD": "MRD /m",
        "Den": "Den /10⁴ trees per grid"
    }
    figure_name = {
        "Age": "Supplementary Figure 8 a",
        "CH": "Supplementary Figure 8 b",
        "MRD": "Supplementary Figure 8 c",
        "Den": "Supplementary Figure 8 d",
        "ΔTa": "Supplementary Figure 8 e",
        "ΔP": "Supplementary Figure 8 f",
        "MAT": "Supplementary Figure 8 g",
        "MAP": "Supplementary Figure 8 h",
        "OR (-)": "Supplementary Figure 8 i",
        "SD": "Supplementary Figure 8 j",
        "PS (-)": "Supplementary Figure 8 k",

    }
    x_lims = {
        "MAT": [-15, 30],
        "MAP": [0, 3500],
        "ΔP": [-1, 0.6],
        "ΔTa": [-2.5, 3],
        "OR (-)": [0.4, 2],
        "SD": [3, 17],
        "PS (-)": [1.3, 2],
        "Age": [1, 250],
        "CH": [1, 40],
        "MRD": [0, 15],
        "Den": [1, 200000],
    }
    y_lims = {
        "MAT": [-0.55, 0.55],
        "MAP": [-0.55, 0.55],
        "ΔP": [-0.55, 0.55],
        "ΔTa": [-0.55, 0.55],
        "OR (-)": [-0.55, 0.55],
        "SD": [-0.55, 0.55],
        "PS (-)": [-0.55, 0.55],
        "Age": [-0.55, 0.55],
        "CH": [-0.55, 0.55],
        "MRD": [-0.55, 0.55],
        "Den": [-0.55, 0.55],
    }
    group_size = {
        "MAT": 5,
        "MAP": 500,
        "ΔP": 0.2,
        "ΔTa": 0.5,
        "OR (-)": 0.2,
        "SD": 2,
        "PS (-)": 0.1,
        "Age": 25,
        "CH": 5,
        "MRD": 2,
        "Den": 20000,
    }
    label_decimal = {
        "MAT": 0,
        "MAP": 0,
        "ΔP": 1,
        "ΔTa": 1,
        "OR (-)": 1,
        "SD": 0,
        "PS (-)": 1,
        "Age": 0,
        "CH": 0,
        "MRD": 0,
        "Den": 0,
    }
    shap_values_path = os.path.join(manuscript_data, r"shap_output", "ΔNDVI shap stats.csv")
    df = dd.read_csv(shap_values_path)
    df = df.compute()

    for col in X_cols:
        print(f"Plot ==> {figure_name[col]}", end="")
        X = df[col].values
        y = df[f"{col} vs {col}"].values
        main_effect_lines(y, X, outputPath, figure_name[col],
                          x_lim=x_lims[col],
                          y_lim=y_lims[col],
                          group_size=group_size[col],
                          labels=[X_labels[col], f"ΔNDVI"],
                          label_decimal=label_decimal[col]
                          )
        print(" >>>>>>> Finish !")


def supplementary_figure_9_a():
    data_path = os.path.join(manuscript_data, r"vd_stats/delta_ndvi_avg.nc")
    outputPath = os.path.join(figure_path, f"Supplementary Figure 9/")
    create_path(outputPath, is_dir=True)
    print(f"Plot ==> Supplementary Figure 9 a", end="")
    data = xr.open_dataset(data_path)["delta_ndvi_avg"].values

    classify_path = os.path.join(manuscript_data, "ForestCharacteristics", "ForestManagement.tif")
    with rio.open(classify_path) as ds:
        fm = ds.read(1)[::-1].astype("float32")
        fm[fm == fm[0, 0]] = np.nan
    data[np.isnan(fm)] = np.nan
    df = pd.DataFrame({classify_col: fm.flatten(), object_val: data.flatten()})
    df_clean = df.dropna(subset=[classify_col, object_val]).copy()
    classifyInfoPath = os.path.join(manuscript_data, "ForestCharacteristics", "forest_management_info.csv")
    classifyInfo = pd.read_csv(classifyInfoPath)
    df_clean[classify_col] = df_clean[classify_col].astype(int)
    df_clean = df_clean.sort_values(by=classify_col, ascending=True)
    mapping = classifyInfo.set_index('Code')['Abbr']
    df_clean = df_clean[df_clean[classify_col].isin(mapping.index)].copy()
    df_clean[classify_col] = df_clean[classify_col].map(mapping)
    # 按自定义顺序排序
    # df_clean = df_clean.sort_values(classify_col)
    colors = ['#009e73', '#0072b2', '#e69f00', '#d55e00', '#cc79a7']
    forest_management_bar(df_clean, classify_col, object_val, colors,
                          xlabel=object_val, ylabel="Forest management",
                          outputpath=outputPath, figure_name=f"Supplementary Figure 9 a")
    print(" >>>>>>> Finish !")


def supplementary_figure_9_b():
    label_name = "Forest Management Practice"
    colors = ["#607D8B", "#795548", "#78909C", "#8D6E63", "#90A4AE"]
    data_path = os.path.join(manuscript_data, r"vd_stats/meta_analysis_data.csv")
    outputPath = os.path.join(figure_path, f"Supplementary Figure 9/")
    create_path(outputPath, is_dir=True)
    print(f"Plot ==> Supplementary Figure 9 b", end="")
    df = pd.read_csv(data_path)
    fmp_col = "Management type"
    df = df.dropna(subset=[fmp_col, object_val], how="any", axis=0)

    abbr_replace_dict = {
        "Afforestation/Reforestation": "A/R",
        "Clear-cut": "CC",
        "Fertilization": "F",
        "Selective logging": "SL",
        "Thinning": "T",

    }
    df[fmp_col] = df[fmp_col].replace(abbr_replace_dict)
    custom_order = ['A/R', 'CC', 'SL', 'T', 'F']  # 定义你想要的顺序

    # 创建分类类型并排序
    df[fmp_col] = pd.Categorical(df[fmp_col], categories=custom_order, ordered=True)
    df = df.sort_values(fmp_col)

    forest_management_practices_bar(df, fmp_col, object_val, colors,
                                    ylabel=object_val, xlabel=label_name,
                                    outputpath=outputPath, figure_name=f"Supplementary Figure 9 b")
    print(" >>>>>>> Finish !")


def supplementary_figure_7_a():
    data_path = os.path.join(manuscript_data, r"vd_stats/delta_ndvi_avg.nc")
    outputPath = os.path.join(figure_path, f"Supplementary Figure 7")
    create_path(outputPath, is_dir=True)
    print(f"Plot ==> Supplementary Figure 7 a", end="")
    with rio.open(os.path.join(manuscript_data, "ForestCharacteristics", "ForestManagement.tif")) as ds:
        fm = ds.read(1)[::-1].astype("float32")
        fm[fm == fm[0, 0]] = np.nan
    with rio.open(os.path.join(manuscript_data, "ForestCharacteristics", "ForestType.tif")) as ds:
        ft = ds.read(1)[::-1].astype("float32")
        ft[ft == ft[0, 0]] = np.nan
        ft_info_path = os.path.join(manuscript_data, "ForestCharacteristics", "forest_type.csv")
        ft_info = pd.read_csv(ft_info_path)

    data = xr.open_dataset(data_path)["delta_ndvi_avg"].values
    df = pd.DataFrame({"FM": fm.flatten(), "FT": ft.flatten(), object_val: data.flatten()})
    df_clean = df.dropna(how="any", axis=0).copy()
    df_clean["FM"] = df_clean['FM'].apply(lambda x: "IF" if x in [11] else "MF" if ~np.isnan(x) else x)
    df_clean = df_clean.sort_values(by="FT", ascending=True)
    ft_mapping = ft_info.set_index('Code')['Abbr']
    df_clean = df_clean[df_clean["FT"].isin(ft_mapping.index)]
    df_clean["FT"] = df_clean["FT"].map(ft_mapping)
    ###################################################################################################################
    colors = ['#7FC7AD', '#7FB2E5']

    delta_ndvi_between_ft_and_fm(df_clean, classify_col="FT", sub_classify_col="FM", value_col=object_val,
                                 color=colors, ylabel=object_val, xlabel="Forest types",
                                 outputpath=outputPath, figure_name=f"Supplementary Figure 7a")
    print(" >>>>>>> Finish !")


def supplementary_figure_7_b():
    data_path = os.path.join(manuscript_data, r"vd_stats/delta_ndvi_avg_sm.nc")
    outputPath = os.path.join(figure_path, f"Supplementary Figure 7")
    create_path(outputPath, is_dir=True)
    print(f"Plot ==> Supplementary Figure 7 b", end="")
    with rio.open(os.path.join(manuscript_data, "ForestCharacteristics", "ForestManagement.tif")) as ds:
        fm = ds.read(1)[::-1].astype("float32")
        fm[fm == fm[0, 0]] = np.nan
    with rio.open(os.path.join(manuscript_data, "ForestCharacteristics", "ForestType.tif")) as ds:
        ft = ds.read(1)[::-1].astype("float32")
        ft[ft == ft[0, 0]] = np.nan
        ft_info_path = os.path.join(manuscript_data, "ForestCharacteristics", "forest_type.csv")
        ft_info = pd.read_csv(ft_info_path)

    data = xr.open_dataset(data_path)["delta_ndvi_avg_sm"].values
    df = pd.DataFrame({"FM": fm.flatten(), "FT": ft.flatten(), object_val: data.flatten()})
    df_clean = df.dropna(how="any", axis=0).copy()
    df_clean["FM"] = df_clean['FM'].apply(lambda x: "IF" if x in [11] else "MF" if ~np.isnan(x) else x)
    df_clean = df_clean.sort_values(by="FT", ascending=True)
    ft_mapping = ft_info.set_index('Code')['Abbr']
    df_clean = df_clean[df_clean["FT"].isin(ft_mapping.index)]
    df_clean["FT"] = df_clean["FT"].map(ft_mapping)
    ###################################################################################################################
    colors = ['#7FC7AD', '#7FB2E5']
    delta_ndvi_between_ft_and_fm(df_clean, classify_col="FT", sub_classify_col="FM", value_col=object_val,
                                 color=colors, ylabel=object_val, xlabel="Forest types",
                                 outputpath=outputPath, figure_name=f"Supplementary Figure 7b")
    print(" >>>>>>> Finish !")


if __name__ == "__main__":
    manuscript_data = "../dataFile"  # "The input path of manuscript data"
    figure_path = "../Expected output"
    object_val = "ΔNDVI"
    classify_col = "FM"

    """Fig.1. Spatial patterns and latitudinal variations in flash drought (FD) characteristics across global forests.
    (a) FD frequency (events per decade), (b) onset rate (pentad⁻¹), (c) stress duration (pentads), (d) peak stress
    magnitude (dimensionless), and (e) zonal averages of FD characteristics by latitude (yellow: frequency; red: peak
    stress magnitude; blue: stress duration; green: onset rate). """
    figure_1abcd()  # Subplot a, b, c, and d
    figure_1e()  # Subplot e

    """Fig. 2. Impacts of flash drought events (FDs) on global forest. (a) Spatial distribution of forest response
    (ΔNDVI) in intact forests (IFs; green indicates ΔNDVI > 0, yellow indicates ΔNDVI < 0) and managed forests (MFs;
    blue indicates ΔNDVI > 0, purple indicates ΔNDVI < 0). (b) Frequency distribution of ΔNDVI responses: vertical
    lines indicate mean values (brown: all forests; green: IFs; blue: MFs); the black vertical line represents ΔNDVI
    = 0. (c) Heatmap showing average ΔNDVI responses binned by mean annual temperature (MAT) and mean annual
    precipitation (MAP). ΔNDVI denotes anomalies in the Normalized Difference Vegetation Index during FDs relative
    to concurrent normal conditions."""
    figure_2a()  # Subplot a
    figure_2b()  # Subplot b
    figure_2c()  # Subplot c

    """Fig. 3 Interactive effects of regulating factors on forest responses (ΔNDVI) to flash drought events between
    intact (IF) and managed forests (MF). Panels show relationships between ΔNDVI and (a) temperature anomalies (ΔTa),
    (b) precipitation anomalies (ΔP), (c) mean annual temperature (MAT), (d) mean annual precipitation (MAP), (e)
    forest age, (f) canopy height (CH), (g) maximum rooting depth (MRD), (h) tree density (Den), (i) onset rate
    (OR; pentad⁻¹), (j) peak stress (PS; negative scale), and (k) stress duration (SD; pentads). ΔNDVI denotes
    anomalies in the Normalized Difference Vegetation Index during FD events relative to concurrent normal conditions.
    Interactive effects quantify how the influence of forest management (MF and IF) changes in forest response to
    variations in other regulating factors, after removing individual effects, as computed using the Shapley Additive
    Explanations (SHAP) framework. Blue and green dots represent event-scale data points for IF and MF, respectively,
    with corresponding solid lines representing binned means. Binning intervals for subplots: ΔTa (0.2), ΔP (0.1),
    MAT (3 °C), MAP (200 mm), Age (10 years), CH (1 m), MRD (1 m), Den (10⁴ trees/grid), OR (0.1 pentad⁻¹), PS (0.1),
    and SD (1 pentad). Dashed horizontal lines indicate ΔNDVI = 0; vertical dashed lines represent thresholds where
    management effects diverge significantly."""
    figure_3()

    """Fig. 4 Schematic illustration of the method used to identify a flash drought event (FD). An FD begins when the
    SPEI drops below –1.28 and decreases by less than –2 within no more than five pentads. The event must persist with
    SPEI ≤ –1.28 for at least three pentads. The end of the FD is defined when SPEI increases above the threshold of
    –1.28 for two consecutive pentads. """
    # The conceptual diagram was created using Microsoft PowerPoint, without using any code.

    """Supplementary Figure 1. Spatial distribution of temporal trends in flash drought event characteristics. Rate of
    change for  a) onset rate, c) stress duration, and e) peak stress, calculated based on Sen’s slope; Significance of
    change for  b) onset rate, d) stress duration, and f) peak stress, showing Mann-Kendall test results. “Extremely
    significant  increasing”, “Significantly increasing”, “No significant”, “Significantly decreasing”, and “Extremely
    significant decreasing” correspond to z-values of Mann-Kendall test results where z < -2.58, -2.58 ≤ z < -1.96,
    -1.96 ≤ z ≤ 1.96, 1.96 < z ≤ 2.58, and z > 2.58, respectively; and g) Zonal averages of the rate of change by
    latitude (red: peak stress, blue: stress duration, green: onset rate). """
    supplementary_figure_1_ace()  # Subplot a, c, and e
    supplementary_figure_1_bdf()  # Subplot b, d, and f
    supplementary_figure_1_g()  # Subplot g

    """Supplementary Figure 2. Spatial distribution of (a) normalized temperature anomalies (ΔTa), (b) normalized
    precipitation anomalies (ΔP), and (c) consistency between ΔP and ΔTa during flash drought events. Legend for panel
    (c): “ΔTa” denotes grid cells with temperature-dominated anomalies (ΔTa > 0.5 and ΔP ≥ -0.5); “ΔP” denotes grid
    cells with precipitation-dominated anomalies (ΔTa ≤ 0.5 and ΔP < -0.5); “ΔTa × ΔP” denotes grid cells with
    concurrent anomalies (ΔTa > 0.5 and ΔP < -0.5). Thresholds in parentheses refer to normalized values shown
    in panels (a) and (b)."""
    supplementary_figure_2_ab()  # Subplot a and b
    supplementary_figure_2_c()  # Subplot c

    """Supplementary Figure 3. Spatial patterns and latitudinal variations in flash drought (FD) characteristics across
    global forests (soil moisture-based). (a) FD frequency (events per decade), (b) onset rate (pentad⁻¹), (c) stress
    duration (pentads), (d) peak stress magnitude (dimensionless), and (e) zonal averages of FD characteristics by
    latitude (yellow: frequency; red: peak stress magnitude; blue: stress duration; green: onset rate). FDs were
    identified using soil moisture data (0–100 cm depth) from the land component of the fifth-generation European
    ReAnalysis dataset (Muñoz-Sabater, J. et al., 2021) and the detection algorithm described in O and Park (2024)."""
    # reference:
    # Muñoz-Sabater, J. et al. ERA5-Land: a state-of-the-art global reanalysis dataset for land applications.
    # Earth Syst. Sci. Data 13, 4349–4383 (2021).
    # O, S. & Park, S. K. Global ecosystem responses to flash droughts are modulated by background climate and
    # vegetation conditions. Commun Earth Environ 5, 1–7 (2024).
    supplementary_figure_3abcd()  # Subplot a, b, c, and d
    supplementary_figure_3_e()  # Subplot e

    """Supplementary Figure 4. The spatial distribution of flash drought events frequency. The same applies to Fig. 1a
    without the forest mask application."""
    supplementary_figure_4()

    """Supplementary Figure 5. Spatial patterns of mean anomalies during flash drought events. (a) Normalized Difference
    Vegetation Index (NDVI), (b) Leaf Area Index (LAI), and (c) Solar-Induced Chlorophyll Fluorescence (SIF) and zonal
    consistency of change with NDVI by latitude for (d) LAI and (e) SIF. “Consistency” indicates both indicators
    increased or decreased together at the same grid cells."""
    supplementary_figure_5_abc()  # Subplot a, b, and c
    supplementary_figure_5_de()  # Subplot d and e

    """Supplementary Figure 6. Impacts of flash drought events (FDs) on global forest (soil moisture-based). (a) Spatial
    distribution of forest response (ΔNDVI) in intact forests (IFs; green indicates ΔNDVI > 0, yellow indicates ΔNDVI
    < 0) and managed forests (MFs; blue indicates ΔNDVI > 0, purple indicates ΔNDVI < 0). (b) Frequency distribution
    of ΔNDVI responses: vertical lines indicate mean values (brown: all forests; green: IFs; blue: MFs); the black
    vertical line represents ΔNDVI = 0. (c) Heatmap showing average ΔNDVI responses binned by mean annual temperature
    (MAT) and mean annual precipitation (MAP). ΔNDVI denotes anomalies in the Normalized Difference Vegetation Index
    during FDs relative to concurrent normal conditions."""
    supplementary_figure_6_a() # Subplot a
    supplementary_figure_6_b() # Subplot b
    supplementary_figure_6_c() # Subplot c

    """Supplementary Figure 7. Normalized Difference Vegetation Index anomalies (ΔNDVI) during (a) Standardized
    Precipitation-Evapotranspiration Index-base, (b) soil moisture-base flash drought events relative to concurrent
    normal conditions under different forest types. Evergreen needle-leaved forest (ENF); Evergreen broad-leaved forest
    (EBF); Deciduous needle-leaved forest (DNF); Deciduous broad-leaved forest (DBF); Intact Forest (IF); Managed Forest
    (MF). The dashed line represents ΔNDVI = 0. Internal violin plot lines (top to bottom): 25th percentile, median,
    and 75th percentile."""
    supplementary_figure_7_a() # Subplot a
    supplementary_figure_7_b() # Subplot b

    """Supplementary Figure 8. The individual effects of regulating factors on forest response (ΔNDVI) to flash drought
    events. Relationships between ΔNDVI and (a) Temperature anomalies (ΔTa), (b) Precipitation anomalies (ΔP), (c) Mean
    annual temperature (MAT), (d) Mean annual precipitation (MAP), (e) Forest age, (f) Canopy height (CH), (g) Maximum
    rooting depth (MRD), (h) Tree density (Den), (i) Onset rate (OR; negative scale), (j) Stress duration (SD), (k) Peak
    stress (PS; negative scale). ΔNDVI represents the anomalies of the Normalized Difference Vegetation Index during
    flash drought events relative to concurrent normal conditions. The individual impact quantifies the influence of
    each factor that does not change with variations in other regulating factors, calculated using Shapley Additive
    Explanations framework. The dashed horizontal line represents ΔNDVI = 0. Box plot coloration (green/red) denotes
    whether the median ΔNDVI value within each bin is greater than (green) or less than (red) zero."""
    supplementary_figure_8()

    """Supplementary Figure 9. Normalized Difference Vegetation Index anomalies (ΔNDVI) during flash drought events
    relative to concurrent normal conditions under different (a) Forest management: Intact forest (IF, naturally
    regenerating forests); Naturally regenerating managed forest (NRMF, exhibiting signs of management activities);
    Planted forest (PF, established through afforestation/reforestation); Plantation rotation forest (PRF, rotation
    periods ≤15 years); Agroforestry (AF). (b) Forest management practices: Afforestation/reforestation (A/R,
    establishing forests on land historically lacking forest cover or recently deforested); Clear-cut (CC, harvesting
    method removing all merchantable trees in a single operation); Selective logging (SL, harvesting primarily the
    largest and highest-quality trees while leaving the residual stand); Thinning (T, removing selected trees, typically
    weaker, diseased, or crowded individuals, to reduce resource competition); Fertilization (F, application of
    nutrients to forest soil to enhance growth rates). The dashed line represents ΔNDVI = 0."""
    supplementary_figure_9_a()
    supplementary_figure_9_b()

    """Supplementary Figure 10. Forest management. IF – forests without any signs of management, including primary
    forests and naturally regenerating forests; MF – forests with clear signs of management, including NRMF, PF, PRF,
    and AF; NRMF – naturally regenerating forests with signs of management, such as logging or clear-cutting; PF –
    planted forests; PRF – intensively managed forest plantations for timber with a short rotation period (maximum 15
    years); AF – agroforestry, including fruit trees or sparse trees on agricultural fields."""
    # Created using ArcGIS 10.2, without using code. The original data is located at
    # "../dataFile/ForestCharacteristics/ForestManagement.tif".
    # The specific meanings of the grid values can be found in
    # "../dataFile/ForestCharacteristics/forest_management_info.csv".

    """Supplementary Figure 11. The preferred reporting items for systematic reviews and meta-analyses flow diagram."""
    # Created Using PRISMA 2020 flow diagram (https://www.prisma-statement.org/), without using code.

    """Supplementary Figure 12. Performance evaluation of XGBoost model with (a) training dataset (n=808,407); (b)
    validation datasets (n=538,939). Metrics: F1-score, coefficient of determination (r²), and root mean square
    error (RMSE)."""
    supplementary_figure_12()
