#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
diretorio = os.getcwd()
import sys
sys.path.append(diretorio + '/Modules')
from RESULTS import results

#inserir nomes das colunas com valores numéricos
columns_with_num_val = ['temperature_carbonization', 'surface_area']

#instanciando o DataFrame consolidado
results = results(results_DF_input_name = '_CONSOLIDATED_DF')
#filling numerical intervals
results.process_columns_with_num_val(DF_columns=columns_with_num_val)

#splitting nGrams
results.split_2grams_terms(DF_column = 'raw_materials')

#grouping (o groups_name deve ser o nome do arquivo JSON no folder "~/Inputs")
results.group_cat_columns_with_input_classes(DF_column = 'applications', groups_name = 'classes_applications')


#plotting
'''
results.plot_cat_loc_gridplot_bins(DF_column = 'raw_materials', 
                                   min_occurrences = 10, size_factor = 0.2, colobar_nticks = 10, palette = 'Turbo256',
                                   background_fill_color = 'white', plot_width=2000, plot_height=2000)

results.plot_cat_cat_gridplot_bins(DF_columns = ['raw_materials', 'applications'],
                                   min_occurrences = 10, size_factor = 0.3, colobar_nticks = 12, palette = 'Turbo256',
                                   background_fill_color = 'white', plot_width=2000, plot_height=1500)

results.plot_cat_cat_gridplot_bins(DF_columns = ['raw_materials_0', 'raw_materials_1'],
                                   min_occurrences = 10, size_factor = 0.3, colobar_nticks = 10, palette = 'Turbo256',
                                   background_fill_color = 'white', plot_width=2000, plot_height=2000)

results.plot_cat_cat_stacked_barplots(DF_columns = ['Publication Year', 'raw_materials'], 
                                        axes_labels = ['Publication Year', 'Raw Materials'],                                         
                                        min_occurrences = 100,
                                        cat_to_filter_min_occurrences = 'raw_materials')

results.plot_1column_network_graph(DF_column = 'processes', max_circle_size = 60, min_circle_size = 15, min_occurrences = 20)

results.plot_2column_network_chord(DF_columns = ['raw_materials', 'classes_applications'], min_occurrences = 10)
results.plot_multicolumn_network_graph(DF_columns = ['raw_materials', 'classes_applications'], max_circle_size = 100, min_circle_size = 20, min_occurrences = 10)

results.plot_group_num_boxplot_correlation(DF_column_with_cat_vals = 'raw_materials',
                                           DF_column_with_num_vals = 'temperature_carbonization',
                                           axes_labels=['biochar raw materia', 'Avg. carb. temperature'],
                                           y_quantiles = ['NA', 0.98],
                                           min_values_for_column = 30,
                                           colobar_nticks = 10,
                                           palette = 'Viridis256',
                                           background_fill_color = 'blue',
                                           size_factor_boxplot = 0.7,
                                           grouplabel_x_offset = 10,
                                           box_plot_plot_width=2000, box_plot_plot_height=1000,
                                           size_factor_anova_grid = 0.4,
                                           grid_plot_width=1000, grid_plot_height=1000)

results.plot_group_num_boxplot_correlation(DF_column_with_cat_vals = 'raw_materials',
                                           DF_column_with_num_vals = 'surface_area', 
                                           axes_labels=['biochar raw material', 'Avg. surface Area'],
                                           y_quantiles = ['NA', 0.98],
                                           min_values_for_column = 20,
                                           grouplabel_x_offset = 5,
                                           colobar_nticks = 6,
                                           palette = 'Viridis256',
                                           background_fill_color = 'blue',
                                           size_factor_boxplot = 0.7,
                                           box_plot_plot_width=2000, box_plot_plot_height=1000,
                                           size_factor_anova_grid = 0.3,
                                           grid_plot_width=1000, grid_plot_height=1000)
'''
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
results.plot_num_num_hexbin_correlation(DF_columns_with_num_vals = ['temperature_carbonization', 'time_pyrolysis'], PUs = ['C', 'min'], 
                                        x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98],
                                        x_min = None, x_max = None, y_min = None, y_max = None,
                                        hex_size = 20, plot_width=800, plot_height=1200)

results.plot_num_num_hexbin_correlation(DF_columns_with_num_vals = ['temperature_carbonization', 'surface_area'], PUs = ['C', 'm2 g-1'], 
                                        x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98],
                                        x_min = None, x_max = None, y_min = None, y_max = 1200,
                                        hex_size = 20, plot_width=800, plot_height=1200)

#results.open_PDF_for_top_values('surface_area')
results.plot_group_num_num_hexbin_correlation(DF_columns_with_num_vals = ['temperature_carbonization', 'surface_area'], 
                                              PUs = ['C', 'm2 g-1'], 
                                              x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98],
                                              DF_column_with_cat_vals_to_group = 'raw_materials',
                                              x_min = None, x_max = None, y_min = None, y_max = 1000,
                                              min_values_for_column = 60,
                                              hex_size = 20, plot_width=1200, plot_height=1200)
