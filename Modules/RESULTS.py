#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class results(object):
    
    def __init__(self, results_DF_input_name = 'DF_FULL'):
        
        import os
        import pandas as pd
        from FUNCTIONS import get_filenames_from_folder
        self.diretorio = os.getcwd()

        #importando a DF consolidada        
        self.results_DF = pd.read_csv(self.diretorio + f'/Outputs/DataFrames/{results_DF_input_name}.csv', index_col=[0,1])
        #print(self.results_DF)

        #importando a DF com a localização de cada documento
        self.doc_location_DF = pd.read_csv(self.diretorio + '/Outputs/doc_location.csv', index_col=0)
        #print(self.doc_location_DF)

        #copiando a DF original
        self.results_DF_processed = self.results_DF.copy()
        self.results_DF_processed.index.names = 'doc', 'counter'

        #importando a DF com o ano de publicação de cada documento
        self.DB_year = pd.read_csv(self.diretorio + '/Outputs/DB_TF_YEAR.csv', index_col=0)
        self.DB_year.index.name = 'doc'
        #adicionando a coluna do ano de publicação
        self.results_DF_processed = self.results_DF_processed.join(self.DB_year[['Publication Year']])

        #limpando as colunas com index
        for column in self.results_DF_processed.columns:
            if column[ -len('_index') : ] == '_index':
                self.results_DF_processed.drop(columns=[column], inplace=True)

        #checando o diretorio
        if not os.path.exists(self.diretorio + '/Outputs/DataFrames/Plots'):
            os.makedirs(self.diretorio + '/Outputs/DataFrames/Plots')

        #econtrando o ultimo numerador dos arquivos de figuras
        filenames = get_filenames_from_folder(folder = self.diretorio + '/Outputs/DataFrames/Plots', file_type='png')
        try:
            if len(filenames) > 0:
                fig_filenames = [file for file in filenames if 'Plot' in file]
                last_counter = int(fig_filenames[-1][ len('Plot') :  ]) + 1
                #atualizando o número do plot_index
                self.last_fig_filename_index = ( ( len('0000') - len(str(last_counter)) ) * '0' ) + str(last_counter)
        except TypeError:
            self.last_fig_filename_index = '0001'
            
        #gerando as paletas
        self.palette_bins = [["#d6deff","#c2ceff","#adbeff","#99adff","#859dff","#708dff","#5c7cff","#476cff"],
                             ["#ffdaeb","#fec9e3","#feb6d9","#fea4d0","#fd92c4","#fd80bd","#fe6eb3","#fc58a7"],
                             ["#ffe299","#ffd15c","#ffc533","#ffba0a","#a37500","#7a5800","#523a00","#3d2c00"],
                             ["#caff99","#c0ff85","#a0ff47","#8bff1f","#76f500","#63cc00","#59b800","#377300"],
                             ["#d5d7d5","#cccdcb","#c1c3c1","#adafac","#a3a5a1","#8e918d","#848782","#5f615e"]]
                             
        #heatmap1 = ["#e3d7ff","#afa2ff","#7a89c2","#72788d","#636b61","#e4e9b2","#e7e08b","#f28f3b","#c8553d","#e2856e"]
        self.palette_heatmap1 = 'Viridis256'
        #heatmap2 = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
        self.palette_heatmap2 = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]



    def cutting_outliers_in_DF(self, column = 'a', quantiles = [0, 1]):
        
        import numpy as np
        temp_series = self.results_DF_processed[column].dropna()
                
        #definindo os limites
        q1, q2 = quantiles
        
        if type(q1) is str and q1.lower() == 'na':
            min_val , max_val = 0 , np.quantile( temp_series.values, q2)
        else:
            min_val , max_val  = np.quantile( temp_series.values, q1) , np.quantile( temp_series.values, q2)
        
        def cut_outliers(entry):
            if min_val <= entry <= max_val:
                return entry
            else:
                return None
                       
        final_Series = temp_series.map(cut_outliers)
        final_Series.dropna(how = 'any', inplace=True)
        
        return final_Series
        


    def process_columns_with_num_val(self, DF_columns=['a', 'b']):
        
        import regex as re
        
        #checando se a coluna está no DF de reultados        
        for column in DF_columns:
            if column not in self.results_DF_processed.columns:
                print(f'Erro! A coluna "{column}" não existe na DF de resultados.')
                return
            
        def avg_intervals(entry):
            try:
                float(entry)
                return round(float(entry), 4)
            except ValueError:
                int1, int2 = re.findall(r'([\-\w\.]+(?=\s)|(?<=\s)[\-\w\.]+)', entry)
                avg_val = round( ( float(int1) + float(int2) ) / 2 , 4)
                #print(int1, int2, avg_val)
                return avg_val
        
        #varrendo a DF
        for column in DF_columns:
            self.results_DF_processed[column] = self.results_DF_processed[column].map(avg_intervals)
        


    def split_2grams_terms(self, DF_column = 'column1'):
        
        import pandas as pd
        import regex as re
        
        def separate_2gram_t1(entry):
            if len(re.findall(r'(\w+(?=\s)|(?<=\s)\w+)', entry)) == 2:
                term1, term2 = re.findall(r'(\w+(?=\s)|(?<=\s)\w+)', entry)
            return term1
        
        def separate_2gram_t2(entry):
            if len(re.findall(r'(\w+(?=\s)|(?<=\s)\w+)', entry)) == 2:
                term1, term2 = re.findall(r'(\w+(?=\s)|(?<=\s)\w+)', entry)
            return term2
        
        #checando se a coluna está no DF de reultados
        if DF_column not in self.results_DF_processed.columns:
            print(f'Erro! A coluna "{DF_column}" não existe na DF de resultados.')
            return
        
        #coletando todos os termos da coluna
        series_t1 = self.results_DF_processed[DF_column].str.lower().map(separate_2gram_t1)
        series_t2 = self.results_DF_processed[DF_column].str.lower().map(separate_2gram_t2)
        series_t1.name = f'{DF_column}_0'
        series_t2.name = f'{DF_column}_1'

        #testando se foi processado algum valor
        if len(series_t1.values) == 0:
            print(f'Erro! Não existem 2grams na coluna "{DF_column}".')
            return
                
        DF_2grams_splitted = pd.DataFrame([], index=series_t1.index)
        DF_2grams_splitted = DF_2grams_splitted.join(series_t1)
        DF_2grams_splitted = DF_2grams_splitted.join(series_t2)
        self.results_DF_processed = self.results_DF_processed.join(DF_2grams_splitted)



    def add_split_index_names_to_column(self, DF):

        import regex as re

        #resetando o index
        DF.reset_index(inplace=True)

        #nome da coluna com os termos concatenados
        concat_terms_column_name = DF.columns.values[0][0]
        
        new_index_names = [ concat_terms_column_name ] + [multi_index[1] for multi_index in DF.columns.values[1:]]
        DF.columns = new_index_names
        
        def separate_2gram_t1(entry):
            if len(re.findall(r'([\w\s]+(?=\+)|(?<=\+)[\w\s]+)', entry)) == 2:
                term1, term2 = re.findall(r'([\w\s]+(?=\+)|(?<=\+)[\w\s]+)', entry)
            return term1
        
        def separate_2gram_t2(entry):
            if len(re.findall(r'([\w\s]+(?=\+)|(?<=\+)[\w\s]+)', entry)) == 2:
                term1, term2 = re.findall(r'([\w\s]+(?=\+)|(?<=\+)[\w\s]+)', entry)
            return term2

        #adicionando as novas colunas
        column1 = separate_2gram_t1(concat_terms_column_name)
        column2 = separate_2gram_t2(concat_terms_column_name)
        DF[column1] = DF[concat_terms_column_name].apply(separate_2gram_t1)
        DF[column2] = DF[concat_terms_column_name].apply(separate_2gram_t2)
        
        return DF
    

    def filter_groups_by_min_occurrences(self, DF = None, grouped = None, min_occurrences = 10):

        #import time
    
        for group in grouped.groups:
            #mínima ocrrência por grupo
            if len(grouped.get_group(group).values) < min_occurrences:
                #print('group: ', group)
                #print(grouped.get_group(group))
                #print('len: ', len(grouped.get_group(group).values))
                #print(f'group "{group}" excluido! ( < {min_occurrences})')
                #time.sleep(3)
                DF = DF.drop( grouped.get_group(group).index )
            else:
                #print(f'group "{group}" coletado! (len = {len(grouped.get_group(group).values)})')
                #time.sleep(3)
                pass

        return DF



    def filter_indexes_by_category_inputs(self, DF = None,  DF_column_to_filter = None, DF_indexes_names = ['doc', 'counter']):

        from FUNCTIONS import load_dic_from_json
        from pandas.core.series import Series    
        
        if type(DF) == Series:
            initial_columns_names = DF.name
            DF_type = 'series'
            print('filtering series on:', DF_column_to_filter)
        else:
            initial_columns_names = list(DF.columns)
            DF_type = 'df'
            print('filtering DF on:', DF_column_to_filter)
            
        #carregando o dic com valores de grupo para remover
        dic_to_remove = load_dic_from_json(self.diretorio + '/Inputs/cats_to_remove.json')
        dic_to_replace = load_dic_from_json(self.diretorio + '/Inputs/cats_to_replace.json')
        
        cat_found_to_remove = False
        cat_found_to_replace = False

        try:
            #tentando encontrar a DF_column no dicionário
            inputs_to_replace = dic_to_replace[DF_column_to_filter]
        
            def replace_inputs(entry):
                try:
                    #print(entry, '>', inputs_to_replace[entry])
                    return inputs_to_replace[entry]
                except KeyError:
                    return entry

            #caso haja mais que um index
            if len(DF_indexes_names) > 1:
                DF = DF.reset_index()            

            DF[DF_column_to_filter] = DF[DF_column_to_filter].apply(replace_inputs)

            #caso haja mais que um index
            if len(DF_indexes_names) > 1:
                DF = DF.reset_index().set_index(DF_indexes_names)
            cat_found_to_replace = True
        
        except KeyError:
            pass        

        try:
            #tentando encontrar a DF_column no dicionário
            cats_to_remove = dic_to_remove[DF_column_to_filter]    

            #mudando o index
            DF = DF.reset_index().set_index(DF_column_to_filter)
            
            #varrendo as categorias
            for cat in cats_to_remove:
                try:
                    #print('removing:', cat)
                    DF = DF.drop(cat)
                except KeyError:
                    pass
            
            #resetando o index
            DF = DF.reset_index().set_index(DF_indexes_names)
            cat_found_to_remove = True
        
        except KeyError:
            pass

        
        if cat_found_to_replace is False:
            print(f'Coluna "{DF_column_to_filter}" não teve indexes substituidos (categoria não foi encontrada em "~/Inputs/cats_to_replace.json").')
        
        #caso a coluna não tenha index para serem eliminados no arquivo: /Inputs/cats_to_remove.json
        if cat_found_to_remove is False:
            print(f'A coluna "{DF_column_to_filter}" não teve indexes removidos (categoria não foi encontrada em "~/Inputs/cats_to_replace.json").')

        #restaurando as colunas do DF
        if DF_type == 'series':
            DF = DF[initial_columns_names]
        elif DF_type == 'df':
            DF = DF[initial_columns_names].copy()

        return DF



    def group_cat_columns_with_input_classes(self, DF_column = 'column1', groups_name = 'classes'):

        #import time
        import numpy as np
        from FUNCTIONS import load_dic_from_json
        
        group_dic = load_dic_from_json(self.diretorio + f'/Inputs/{groups_name}.json')

        def add_group(entry):
            #print(entry)
            #print(type(entry))
            #print('entry == nan', entry == 'nan')
            #print('entry == np.nan', entry == np.nan)
            #time.sleep(1)
            group_name_found = False
            for group_name in group_dic.keys():
                group_members = group_dic[group_name]
                for member in group_members:
                    try:
                        if member in entry:
                            group_name_found = True
                            group = group_name
                    except TypeError:
                        pass
            
            if group_name_found is True:                
                return group
            elif type(entry) == float:
                return np.nan
            else:
                return 'other'            
        
        #criando a DF de grupo
        series_group = self.results_DF_processed[DF_column].str.lower().map(add_group)
        series_group.name = groups_name
        
        #concatenando com a DF de resultados
        self.results_DF_processed = self.results_DF_processed.join(series_group)



    def plot_cat_cat_stacked_barplots(self, DF_columns = ['0', '1'], axes_labels = ['a', 'b'], 
                                      min_occurrences = 10, cat_to_filter_min_occurrences = ''):
    
        print('\n\n> function: plot_cat_cat_stacked_barplots')    
        print('Columns used: ', DF_columns)

        #checando se as colunas estão no DF de reultados
        for column in DF_columns:
            if column not in self.results_DF_processed.columns:
                print(f'Erro! A coluna "{column}" não existe na DF de resultados.')
                return
        
        import numpy as np
        np.seterr(invalid='ignore')
        import holoviews as hv
        hv.extension('matplotlib')
        from holoviews import opts
        import pandas as pd
        #import time

        DF_copy = self.results_DF_processed[[DF_columns[0], DF_columns[1]]].copy()
        #definir a coluna que será usada para o agrupamento        
        column_to_groupby = DF_columns[0]
        
        #filtrando indexes que estão no arquivo /Inputs/cats_to_remove.json e /Inputs/cats_to_replace.json
        filtered_DF = self.filter_indexes_by_category_inputs(DF = DF_copy,  DF_column_to_filter = DF_columns[0], DF_indexes_names = ['doc', 'counter'])
        filtered_DF = self.filter_indexes_by_category_inputs(DF = filtered_DF,  DF_column_to_filter = DF_columns[1], DF_indexes_names = ['doc', 'counter'])
        
        #agrupando a DF (gera um objeto GROUPBY com multiindex)
        grouped = filtered_DF.groupby(cat_to_filter_min_occurrences)
        #filtrando por ocorrência
        filtered_DF = self.filter_groups_by_min_occurrences(DF = filtered_DF, grouped = grouped, min_occurrences = min_occurrences)

        #criando uma DF para colocar os resultados modificados
        modified_DF = pd.DataFrame([], columns=[ DF_columns[0], DF_columns[1], 'count', 'perc' ])
        #agrupando a DF (gera um objeto GROUPBY com multiindex)
        grouped = filtered_DF.groupby(column_to_groupby) 
        counter = 0
        max_counts = 0
        for group in grouped.groups:
            if len(grouped.get_group(group).values) > max_counts:
                max_counts = len(grouped.get_group(group).values)
            #agrupando novamente
            grouped_inner = grouped.get_group(group).groupby(DF_columns[1])
            #coletando a serie agrupada
            series = grouped_inner.describe().loc[ : , ('Publication Year', 'count') ]
            #normalizando os valores
            series_perc = series / series.cumsum().max() * 100 
            for index in series.index:
                modified_DF.loc[counter] = group, index , series.loc[index], series_perc.loc[index]
                counter += 1
        
        key_dimensions   = [(DF_columns[0], axes_labels[0]), (DF_columns[1], axes_labels[1])]
        value_dimensions = [('count', 'Counts'), ('perc', 'Percentage (%)')]
        
        table = hv.Table(modified_DF, key_dimensions, value_dimensions)
        
        #plotand em counts e porcentagens
        for mode in ('count', 'perc'):
            if mode == 'count':
                ylim = [0 - (0.09*max_counts), max_counts + (0.09*max_counts)]
                ybounds = [0, max_counts]
            elif mode == 'perc':
                ylim = [-9, 109]
                ybounds = [0, 100]
            
            #plotando
            stacked_bars = table.to.bars([DF_columns[0], DF_columns[1]], mode)
            
            #customizando
            stacked_bars.opts(opts.Bars(color=hv.Cycle('tab20')))
            stacked_bars.opts(xrotation=45,
                              show_legend=True, 
                              legend_position='right', 
                              legend_opts={'title_fontsize': 35,
                                         'fontsize': 35,
                                         'frameon':False},
                              stacked=True, 
                              aspect=2,
                              fig_size=1200)
            
            
            fig = hv.render(stacked_bars)
            fig.axes[0].set_xlabel(axes_labels[0], fontsize=45)
            fig.axes[0].set_ylabel(axes_labels[1], fontsize=45)
            fig.axes[0].set_ylim(ylim)
            fig.axes[0].tick_params(axis="x",colors="k",width=4,length=7, labelsize=45)
            fig.axes[0].tick_params(axis="y",colors="k",width=4,length=7, labelsize=45)
            fig.axes[0].spines["left"].set_visible(True)
            fig.axes[0].spines["left"].set_color("k")
            fig.axes[0].spines["left"].set_linewidth(4)
            fig.axes[0].spines["left"].set_bounds(ybounds)
            fig.axes[0].spines["bottom"].set_visible(True)
            fig.axes[0].spines["bottom"].set_color("k")
            fig.axes[0].spines["bottom"].set_linewidth(4)
            fig.axes[0].spines["top"].set_visible(False)
            fig.axes[0].spines["right"].set_visible(False)
            fig.tight_layout(pad=1.4)
        
            print(f'Salvando a figura ~/Outputs/DataFrames/Plots/Plot{self.last_fig_filename_index}.png ...')
            fig.savefig(self.diretorio + f'/Outputs/DataFrames/Plots/Plot{self.last_fig_filename_index}.png')
            
            #atualizando o número do plot_index
            self.last_fig_filename_index = ( ( len('0000') - len(str(int(self.last_fig_filename_index)+1)) ) * '0' ) + str(int(self.last_fig_filename_index) + 1)



    def plot_1column_network_graph(self, DF_column = '', max_circle_size = 100, min_circle_size = 10, min_occurrences = 0):

        import itertools as itt
        import pandas as pd
        import time
        
        print('\n\n> function: plot_1column_network_graph')
        print('Column used: ', DF_column)
        
        if DF_column not in self.results_DF_processed.columns:
            print(f'Erro! A coluna "{DF_column}" não existe na DF de resultados.')
            return
        
        #copiando a DF
        DF_copy = self.results_DF_processed[DF_column].copy().dropna()

        #filtrando indexes que estão no arquivo /Inputs/cats_to_remove.json e /Inputs/cats_to_replace.json
        filtered_series = self.filter_indexes_by_category_inputs(DF = DF_copy,  DF_column_to_filter = DF_column, DF_indexes_names = ['doc', 'counter'])

        #agrupando a DF (gera um objeto GROUPBY com multiindex)
        grouped = filtered_series.groupby(filtered_series)
        filtered_series = self.filter_groups_by_min_occurrences(DF = filtered_series, grouped = grouped, min_occurrences = min_occurrences)

        #criando uma df para colocar as relações da rede
        concat_network_relat = pd.Series([], name='weight', dtype=object)
        
        #criando uma lista para colocar todos os nodes
        all_nodes = []
        for file in filtered_series.index.levels[0]:
            try:
                vals = filtered_series.loc[ file , ].values
            except KeyError:
                continue
            
            vals = list(set([val for val in vals]))
            if len(vals) > 1:
                for comb in itt.combinations(vals, 2):                    
                    sorted_comb = tuple(sorted(comb))
                    index = sorted_comb[0] + '_' + sorted_comb[1]
                    try:
                        concat_network_relat.loc[index] = concat_network_relat.loc[index] + 1
                    except KeyError:
                        concat_network_relat.loc[index] = 1
                    
                    for node in sorted_comb:
                        if node not in all_nodes:
                            all_nodes.append(node)

        #print(concat_network_relat)
        concat_network_relat.index.name = 'index'
        concat_network_relat = concat_network_relat.reset_index()

        nodes_cats = dict(zip(sorted(all_nodes), [DF_column] * len(all_nodes)))

        self.graph_plot(concat_network_relat, nodes_cats, color_cats = False, max_circle_size = max_circle_size, min_circle_size = min_circle_size)
            
        

    def plot_multicolumn_network_graph(self, DF_columns = ['0', '1', '2'], max_circle_size = 100, min_circle_size = 10, min_occurrences = 10):
        
        import numpy as np
        import pandas as pd

        print('\n\n> function: plot_multicolumn_network_graph')
        print('Columns used: ', DF_columns)

        #checando se as colunas estão no DF de reultados
        for column in DF_columns:
            if column not in self.results_DF_processed.columns:
                print(f'Erro! A coluna "{column}" não existe na DF de resultados.')
                return

        DF_column_name_join_list = []

        #copiando a DF
        DF_copy = self.results_DF_processed[DF_columns].copy().dropna()

        #filtrando indexes que estão no arquivo /Inputs/cats_to_remove.json e /Inputs/cats_to_replace.json
        DF_copy = self.filter_indexes_by_category_inputs(DF = DF_copy,  DF_column_to_filter = DF_columns[0], DF_indexes_names = ['doc', 'counter'])
        DF_copy = self.filter_indexes_by_category_inputs(DF = DF_copy,  DF_column_to_filter = DF_columns[1], DF_indexes_names = ['doc', 'counter'])

        #varrendo as colunas para encontrar as conexões
        for column_i in range(len(DF_columns)-1):
            
            name1 = DF_columns[column_i]
            name2 = DF_columns[column_i+1]
                        
            #fazendo o merge das duas colunas para grouping
            DF_column_name_join = name1 + '_' + name2
            DF_column_name_join_list.append(DF_column_name_join)
            DF_copy[DF_column_name_join] = pd.Series(DF_copy[name1].values + '_' + DF_copy[name2].values, index=DF_copy.index)
    
        #definindo uma tag para cada coluna coletada no DataFrame
        column_group_tag_list = []

        #varrendo as colunas para definir as tags de cada entrada
        for column in DF_columns:
            #definindo uma tag para cada intem (node) de coluna coletada no DataFrame
            tags = list( zip( list(np.unique(DF_copy[column].values)) , [column] * len(np.unique(DF_copy[column].values)) ) )
            column_group_tag_list.extend(tags)
        
        #definindo um dicionário com o tag de cada termo usado (node)
        nodes_cats = dict(column_group_tag_list)

        
        #criando o DF para concatenar
        concat_network_relat = pd.DataFrame([])
        #agrupando os resultados
        for join_name in DF_column_name_join_list:

            #agrupando a DF (gera um objeto GROUPBY com multiindex)
            grouped = DF_copy.groupby(join_name)
            DF_copy = self.filter_groups_by_min_occurrences(DF = DF_copy, grouped = grouped, min_occurrences = min_occurrences)

            #agrupando para fazer a contagem de aparição das correlações
            grouped_DF = DF_copy.groupby(join_name).describe()
            concat_network_relat = pd.concat([ concat_network_relat , grouped_DF.loc[ : , ( DF_columns[0] , 'count' ) ]])            
        
        #resetando o index e renomeando a coluna
        concat_network_relat = concat_network_relat.reset_index()
        concat_network_relat.rename(columns={0:'weight'}, inplace=True)
        
        self.graph_plot(concat_network_relat, nodes_cats, color_cats = True, max_circle_size = max_circle_size, min_circle_size = min_circle_size)



    def plot_2column_network_chord(self, DF_columns = ['0', '1'], min_occurrences = 10):
        
        import pandas as pd


        print('\n\n> function: plot_2column_network_chord')    
        print('Columns used: ', DF_columns)

        #checando se as colunas estão no DF de reultados
        for column in DF_columns:
            if column not in self.results_DF_processed.columns:
                print(f'Erro! A coluna "{column}" não existe na DF de resultados.')
                return

        #copiando a DF
        DF_copy = self.results_DF_processed[DF_columns].copy().dropna()
                        
        #filtrando indexes que estão no arquivo /Inputs/cats_to_remove.json e /Inputs/cats_to_replace.json
        filtered_DF = self.filter_indexes_by_category_inputs(DF = DF_copy,  DF_column_to_filter = DF_columns[0], DF_indexes_names = ['doc', 'counter'])
        filtered_DF = self.filter_indexes_by_category_inputs(DF = filtered_DF,  DF_column_to_filter = DF_columns[1], DF_indexes_names = ['doc', 'counter'])

        #fazendo o merge das duas colunas para grouping
        DF_column_name_join = DF_columns[0] + '_' + DF_columns[1]
        filtered_DF[DF_column_name_join] = pd.Series(filtered_DF[DF_columns[0]].values + '_' + filtered_DF[DF_columns[1]].values, index=filtered_DF.index)

        #agrupando a DF (gera um objeto GROUPBY com multiindex)
        grouped = filtered_DF.groupby(DF_column_name_join)
        filtered_DF = self.filter_groups_by_min_occurrences(DF = filtered_DF, grouped = grouped, min_occurrences = min_occurrences)

        #agrupando para fazer a contagem de aparição das correlações
        grouped_DF = filtered_DF.groupby(DF_column_name_join).describe().loc[ : , ( DF_columns[0] , 'count' ) ]

        #plotando
        self.chord_plot(grouped_DF)
        


    def plot_cat_cat_gridplot_bins(self, DF_columns = ['0', '1'],
                                   min_occurrences = 10, size_factor = 0.4, colobar_nticks = 20,
                                   palette = 'Viridis256',
                                   background_fill_color = 'blue',
                                   plot_width=1000, plot_height=1000):

        import pandas as pd
        
        print('\n\n> function: plot_cat_cat_gridplot_bins')    
        print('DF_columns: ', DF_columns)
        
        #checando se as colunas estão no DF de reultados
        for column in DF_columns:
            if column not in self.results_DF_processed.columns:
                print(f'Erro! A coluna "{column}" não existe na DF de resultados.')
                return
        
        #copiando a DF        
        DF_copy = self.results_DF_processed[[DF_columns[0], DF_columns[1]]].copy().dropna()

        #filtrando indexes que estão no arquivo /Inputs/cats_to_remove.json e /Inputs/cats_to_replace.json
        DF_copy = self.filter_indexes_by_category_inputs(DF = DF_copy,  DF_column_to_filter = DF_columns[0], DF_indexes_names = ['doc', 'counter'])
        DF_copy = self.filter_indexes_by_category_inputs(DF = DF_copy,  DF_column_to_filter = DF_columns[1], DF_indexes_names = ['doc', 'counter'])

        #fazendo o merge das duas colunas para grouping
        DF_column_name_join = DF_columns[0] + '+' + DF_columns[1]
        DF_copy[DF_column_name_join] = pd.Series(DF_copy[DF_columns[0]].values + '+' + DF_copy[DF_columns[1]].values, index=DF_copy.index)
        
        #agrupando a DF (gera um objeto GROUPBY com multiindex)
        grouped = DF_copy.groupby(DF_column_name_join)
        #limpando os grupos que possuem poucos valores (menores que min_values_for_column) e que estão no ~/Inputs
        DF_copy = self.filter_groups_by_min_occurrences(DF = DF_copy, grouped = grouped, min_occurrences = min_occurrences)

        #agrupando a DF (gera um objeto GROUPBY com multiindex)
        DF_grouped = DF_copy.groupby(DF_column_name_join).describe()        
        DF_grouped = self.add_split_index_names_to_column(DF_grouped)
        
        #coletando apenas a coluna com o número de ocorrências
        DF_grouped = DF_grouped[[DF_columns[0], DF_columns[1], 'count']].copy()

        #plotando
        self.cat_cat_gridplot(DF = DF_grouped,
                              DF_x_column = DF_columns[0], DF_y_column = DF_columns[1], DF_val_column = 'count',
                              min_occurrences = min_occurrences,
                              size_factor=size_factor, 
                              colobar_nticks = colobar_nticks,
                              palette = palette,
                              background_fill_color = background_fill_color,
                              plot_width=1000, plot_height=1000)



    def plot_cat_loc_gridplot_bins(self, DF_column = '', axes_labels = ['a', 'b'], min_occurrences = 10, size_factor = 0.4, 
                                   colobar_nticks = 20,
                                   palette = 'Viridis256',
                                   background_fill_color = 'blue',
                                   plot_width=1000, plot_height=1000):

        import pandas as pd
        import numpy as np

        print('\n\n> function: plot_cat_loc_gridplot_bins')
        print('DF_column: ', DF_column)
        
        #checando se as colunas estão no DF de reultados
        if DF_column not in self.results_DF_processed.columns:
            print(f'Erro! A coluna "{DF_column}" não existe na DF de resultados.')
            return
            
        def cluster_location(line_vals):
            return [val for val in line_vals if val is not np.nan]
        
        temp_series = self.doc_location_DF.apply(cluster_location, axis=1)
        temp_DF1 = pd.DataFrame([])
        temp_DF1['doc'] = temp_series.index.values
        temp_DF1['location'] = temp_series.values

        temp_DF2 = pd.merge(self.results_DF_processed.reset_index(), temp_DF1, on='doc')
        temp_DF2.set_index(['doc', 'counter'], inplace=True)
        temp_DF2 = temp_DF2[[DF_column, 'location']].copy()
        
        #unstacking o location
        loc_unstacked = []
        loc_multipler = []
        for loc_list in temp_DF2['location'].values:
            loc_unstacked.extend(loc_list)
            loc_multipler.append(len(loc_list))
        #criando a lista associada com os valores da coluna com a categoria
        cat_unstacked = []
        for i in range(len(temp_DF2[DF_column].values)):
            cat_unstacked.extend( [temp_DF2[DF_column].values[i]] * loc_multipler[i] )
        DF_copy = pd.DataFrame(list(zip(cat_unstacked , loc_unstacked)), columns=[DF_column, 'location'])
        DF_copy.index.name = 'index'

        #filtrando indexes que estão no arquivo /Inputs/cats_to_remove.json e /Inputs/cats_to_replace.json
        DF_copy = self.filter_indexes_by_category_inputs(DF = DF_copy,  DF_column_to_filter = DF_column, DF_indexes_names = ['index'])

        #fazendo o merge das duas colunas para grouping
        DF_column_name_join = DF_column + '+location'
        DF_copy[DF_column_name_join] = pd.Series(DF_copy[DF_column].values + '+' + DF_copy['location'].values, index=DF_copy.index)
        
        #agrupando a DF (gera um objeto GROUPBY com multiindex)
        grouped = DF_copy.groupby(DF_column_name_join)        
        #limpando os grupos que possuem poucos valores (menores que min_values_for_column) e que estão no ~/Inputs
        DF_copy = self.filter_groups_by_min_occurrences(DF = DF_copy, grouped = grouped, min_occurrences = min_occurrences)
        
        #agrupando a DF (gera um objeto GROUPBY com multiindex)
        DF_grouped = DF_copy.groupby(DF_column_name_join).describe()        
        DF_grouped = self.add_split_index_names_to_column(DF_grouped)
        #coletando apenas a coluna com o número de ocorrências
        DF_grouped = DF_grouped[[DF_column, 'location', 'count']].copy()

        #plotando
        self.cat_cat_gridplot(DF = DF_grouped,
                              DF_x_column = DF_column, DF_y_column = 'location', DF_val_column = 'count',
                              min_occurrences = min_occurrences,
                              size_factor = size_factor,
                              colobar_nticks = colobar_nticks,
                              palette = palette,
                              background_fill_color = background_fill_color,
                              plot_width=plot_width, plot_height=plot_height)


    
    def plot_group_num_boxplot_correlation(self, DF_column_with_cat_vals = 'a', DF_column_with_num_vals = 'b', 
                                           axes_labels=['', ''],
                                           y_quantiles = [0, 1], 
                                           min_values_for_column = 10,
                                           size_factor_boxplot = 1,
                                           size_factor_anova_grid = 1,
                                           grouplabel_x_offset = 0,
                                           colobar_nticks = 5,
                                           palette = 'Viridis256',
                                           background_fill_color='blue',
                                           box_plot_plot_width=1000, box_plot_plot_height=1000,
                                           grid_plot_width=1000, grid_plot_height=1000):
        
        import pandas as pd
        import statsmodels.stats.multicomp as mc

        print('\n\n> function: plot_group_num_boxplot_correlation')

        #checando se as colunas estão no DF de reultados
        for column in (DF_column_with_cat_vals, DF_column_with_num_vals):
            if column not in self.results_DF_processed.columns:
                print(f'Erro! A coluna "{column}" não existe na DF de resultados.')
                return

        #eliminando os outliers
        input1 , input2 = DF_column_with_cat_vals , DF_column_with_num_vals
        DF_copy = pd.DataFrame([], index=self.results_DF_processed.index)
        DF_copy = DF_copy.join(self.results_DF_processed[input1])
        y_q1 , y_q2 = y_quantiles
        DF_copy = DF_copy.join( self.cutting_outliers_in_DF(column = input2, quantiles = [y_q1, y_q2]) )
        DF_copy.dropna(axis=0, how='any', inplace=True)       
    
        #filtrando indexes que estão no arquivo /Inputs/cats_to_remove.json e /Inputs/cats_to_replace.json
        DF_copy = self.filter_indexes_by_category_inputs(DF = DF_copy,  DF_column_to_filter = DF_column_with_cat_vals, DF_indexes_names = ['doc', 'counter'])

        #agrupando a DF (gera um objeto GROUPBY com multiindex)
        grouped = DF_copy.groupby(DF_column_with_cat_vals)
        #limpando os grupos que possuem poucos valores (menores que min_values_for_column) e que estão no ~/Inputs
        DF_copy = self.filter_groups_by_min_occurrences(DF = DF_copy, grouped = grouped, min_occurrences = min_values_for_column)        
        
        #agrupando a serie
        grouped_filtered = DF_copy[input2].groupby(DF_copy[input1])
        
        #valores mínimo e máximo do input numérico
        ymin , ymax = DF_copy[input2].values.min() , DF_copy[input2].values.max()
        
        #plotando
        self.group_num_boxplot(grouped_DF = grouped_filtered, 
                               input1 = input1, 
                               input2 = input2, 
                               axes_label1 = axes_labels[0],
                               axes_label2 = axes_labels[1],
                               ymin = ymin, ymax = ymax,
                               size_factor = size_factor_boxplot,
                               grouplabel_x_offset = grouplabel_x_offset,
                               colobar_nticks = colobar_nticks,
                               plot_width=box_plot_plot_width, plot_height=box_plot_plot_height)
        
        
        #plotando o anova entre os grupo        
        stacked_data = pd.DataFrame([])
        #varrendo os grupos
        for group_name in grouped_filtered.groups:
            temp_df = grouped_filtered.get_group(group_name).reset_index()
            series = temp_df[input2]
            temp_df['values'] = series
            temp_df['group'] = [group_name] * len(temp_df['values'].index)
            stacked_data = pd.concat([stacked_data, temp_df[['values', 'group']].copy()])

        #fazendo o TUKEY HONESTLY SIGNIFICANT DIFFERENCE
        comp = mc.MultiComparison(stacked_data['values'], stacked_data['group'])
        post_hoc_res = comp.tukeyhsd()        
        post_hoc_df = pd.DataFrame(data=post_hoc_res._results_table.data[1:], columns=post_hoc_res._results_table.data[0])

        self.cat_cat_gridplot(DF = post_hoc_df, DF_x_column = 'group1', DF_y_column = 'group2', DF_val_column = 'p-adj', 
                              min_occurrences = 0, size_factor=size_factor_anova_grid, 
                              colobar_nticks = colobar_nticks, palette = palette, background_fill_color = background_fill_color,
                              plot_width=grid_plot_width, plot_height=grid_plot_height)



    def plot_num_num_hexbin_correlation(self, DF_columns_with_num_vals = ['a', 'b'], PUs = ['d', 'e'], x_quantiles = [0, 1], y_quantiles = [0, 1],
                                        x_min = None, x_max = None, y_min = None, y_max = None,
                                        hex_size = 10, plot_width=1000, plot_height=1000):
            
        import pandas as pd

        print('\n\n> function: plot_num_num_hexbin_correlation')

        #checando se as colunas estão no DF de reultados        
        for column in DF_columns_with_num_vals:
            if column not in self.results_DF_processed.columns:
                print(f'Erro! A coluna "{column}" não existe na DF de resultados.')
                return
        
        #eliminando os outliers
        input1 , input2 = DF_columns_with_num_vals
        DF_copy = pd.DataFrame([], index=self.results_DF_processed.index)
        x_q1 , x_q2 = x_quantiles
        DF_copy = DF_copy.join( self.cutting_outliers_in_DF(column = input1, quantiles = [x_q1, x_q2]) )
        y_q1 , y_q2 = y_quantiles
        DF_copy = DF_copy.join( self.cutting_outliers_in_DF(column = input2, quantiles = [y_q1, y_q2]) )
        DF_copy.dropna(axis=0, how='any', inplace=True)
         
        #plotando
        self.num_num_hexbinplot(DF = DF_copy, PUs = PUs, input1 = input1, input2 = input2, 
                                x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max, 
                                hex_size = hex_size, plot_width=plot_width, plot_height=plot_height)



    def plot_group_num_num_hexbin_correlation(self, DF_columns_with_num_vals = ['a', 'b'], PUs = ['d', 'e'], 
                                              x_quantiles = [0, 1], y_quantiles = [0, 1],
                                              DF_column_with_cat_vals_to_group = None,
                                              x_min = None, x_max = None, y_min = None, y_max = None,
                                              min_values_for_column = 10,
                                              hex_size = 10, plot_width=1000, plot_height=1000):
            
        import pandas as pd

        print('\n\n> function: plot_cat_num_num_hexbin_correlation')

        #checando se as colunas estão no DF de reultados        
        for column in DF_columns_with_num_vals:
            if column not in self.results_DF_processed.columns:
                print(f'Erro! A coluna "{column}" não existe na DF de resultados.')
                return            
        if DF_column_with_cat_vals_to_group not in self.results_DF_processed.columns:
            print(f'Erro! A coluna "{DF_column_with_cat_vals_to_group}" não existe na DF de resultados.')
            return

        DF_copy = pd.DataFrame([], index=self.results_DF_processed.index)
        
        #adicionando a coluna com o parametro categorico a agrupar
        cat_to_group = DF_column_with_cat_vals_to_group
        DF_copy = DF_copy.join(self.results_DF_processed[cat_to_group])
       
        #eliminando os outliers
        input1 , input2 = DF_columns_with_num_vals
        x_q1 , x_q2 = x_quantiles
        DF_copy = DF_copy.join( self.cutting_outliers_in_DF(column = input1, quantiles = [x_q1, x_q2]) )
        y_q1 , y_q2 = y_quantiles
        DF_copy = DF_copy.join( self.cutting_outliers_in_DF(column = input2, quantiles = [y_q1, y_q2]) )
        DF_copy.dropna(axis=0, how='any', inplace=True)
    
        #filtrando indexes que estão no arquivo /Inputs/cats_to_remove.json e /Inputs/cats_to_replace.json
        DF_copy = self.filter_indexes_by_category_inputs(DF = DF_copy,  DF_column_to_filter = DF_column_with_cat_vals_to_group, 
                                                         DF_indexes_names = ['doc', 'counter'])

        #agrupando a DF (gera um objeto GROUPBY com multiindex)
        grouped = DF_copy[[input1,input2, cat_to_group]].groupby(cat_to_group)
        #limpando os grupos que possuem poucos valores (menores que min_values_for_column) e que estão no ~/Inputs
        DF_copy = self.filter_groups_by_min_occurrences(DF = DF_copy, grouped = grouped, min_occurrences = min_values_for_column)
        
        if len(DF_copy.index) == 0:
            print('ERRO!')
            print('Não há entradas no DF filtrado.')
            print('Diminuir o valor de "min_values_for_column".')
            return
        
        #agrupando a DF (gera um objeto GROUPBY com multiindex)
        grouped_filtered = DF_copy[[input1, input2, cat_to_group]].groupby(cat_to_group)
        print('Número de grupos encontrados: ', len(grouped_filtered))
        
        if len(grouped_filtered) > len(self.palette_bins):
            print('ERRO! O número de grupos é maior que o número de paletas de cores para plotagem do group_num_num_hexbin.')
            print('Número de grupos: ', len(grouped_filtered))
            print('Número de paletas: ', len(self.palette_bins))
        else:
            pass
            #plotando
            self.group_num_num_hexbinplot(DF = DF_copy, grouped = grouped_filtered, PUs = PUs, input1 = input1, input2 = input2, 
                                          x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max, 
                                          hex_size = hex_size, plot_width=plot_width, plot_height=plot_height)



    #funções de plotagem            
    def graph_plot(self, concat_network_relat, nodes_cats, color_cats = True, max_circle_size = 100, min_circle_size = 20):
        
        import numpy as np
        import pandas as pd
        import networkx as nx
        from bokeh.io import export_png
        from bokeh.models import Range1d, Circle, ColumnDataSource, MultiLine, Label
        from bokeh.plotting import figure
        from bokeh.plotting import from_networkx
        from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges
        from bokeh.palettes import inferno, viridis, cividis, turbo
        from networkx.algorithms import community        
        
        def get_first_column_names(term):            
            import regex as re
            split_term = re.findall(r'[\w\s]+(?=_)', term)
            return split_term[0]

        def get_second_column_names(term):            
            import regex as re
            split_term = re.findall(r'(?<=_)[\w\s]+', term)
            return split_term[0]  
        
        #mudando o dtype para a coluna weight para plotar
        edges_weights_DF = pd.DataFrame([])
        edges_weights_DF['weight'] = concat_network_relat['weight'].values.astype('int32')

        #coletando os sources e targets
        edges_weights_DF['source'] = concat_network_relat['index'].apply(get_first_column_names)
        edges_weights_DF['target'] = concat_network_relat['index'].apply(get_second_column_names)

        #obtendo a rede no networkx        
        G = nx.from_pandas_edgelist(edges_weights_DF, source='source', target='target', edge_attr='weight')
        nodes_position_dic = nx.drawing.layout.random_layout(G)                   

        #calculando o tamanho dos círculos
        degrees = dict(nx.degree(G))
        max_degree_val = np.array(list(degrees.values())).max()
        min_degree_val = np.array(list(degrees.values())).min()        
        for node_name in degrees.keys():
            degrees[node_name] = ( ( (degrees[node_name] - min_degree_val) / (max_degree_val - min_degree_val) ) * (max_circle_size - min_circle_size) ) + min_circle_size
            
        #calculando a espessura das edges
        edges_weights = nx.get_edge_attributes(G,'weight')
        max_degree_val = np.array(list(degrees.values())).max()
        min_degree_val = np.array(list(degrees.values())).min()  
        max_edge_line_width = 2
        min_edge_line_width = 1
        for edge_name in edges_weights.keys():
            edges_weights[edge_name] = round( (((edges_weights[edge_name] - min_degree_val) / (max_degree_val - min_degree_val) ) * (max_edge_line_width - min_edge_line_width)), 3) + min_edge_line_width
        
        #atribuindo um cor para cada node
        nodes_color = {}
        #definindo uma cor para cada categoria de nodes (cada categoria é uma coluna da DF)
        if color_cats is True:
            palette_nodes = ["#f25f5c", "#247ba0", "#70c1b3", "#50514f"]
            #definindo um dicionário de número para categoria
            cats_number_dic = dict( zip( list(set(nodes_cats.values())) , list(range(len(set(nodes_cats.values())))) ))
            for node in nodes_cats.keys():
                nodes_cat = nodes_cats[node]
                nodes_color[node] = palette_nodes[ cats_number_dic[nodes_cat] ]
        
        #definindo uma cor para cada node
        elif color_cats is False:
            palette_nodes = turbo(len(nodes_cats.keys()))
            #definindo um dicionário de número para node
            cats_number_dic = dict( zip( list(set(nodes_cats.keys())) , list(range(len(set(nodes_cats.keys())))) ))
            for node in nodes_cats.keys():
                nodes_color[node] = palette_nodes[ cats_number_dic[node] ]    

        #fazendo o color map dos edges weights
        #encontrando os valores únicos de weight
        unique_weights_list = list(set(edges_weights.values()))
        unique_weights_list.sort()
        #coletando uma paleta com o número de cores correspondente ao número de edge weights
        palette_colors = cividis(len(unique_weights_list))
        unique_weight_colors_dic = dict(zip(unique_weights_list, palette_colors))
        
        #dicionário com as cores
        edge_colors = {}
        for edge_name in edges_weights:
            weight = edges_weights[edge_name]            
            edge_colors[edge_name] = unique_weight_colors_dic[weight]
        
        #plotando o boxplot
        p = figure(toolbar_location=None, 
                   #title='.',
                   match_aspect=False,
                   x_range=Range1d(-0.2, 1.5), y_range=Range1d(-0.2, 1.2),
                   background_fill_color='white', 
                   plot_width=1500, plot_height=1500,
                   min_border=10)

        p.axis.visible = False
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
                
        #adicionando as edges
        for edge_name in edges_weights.keys():
            source = edge_name[0]
            target = edge_name[1]
            #print('source_name: ', source, nodes_position_dic[source], '; target: ', target, nodes_position_dic[target])
            p.line([nodes_position_dic[source][0], nodes_position_dic[target][0]], [nodes_position_dic[source][1], nodes_position_dic[target][1]], 
                   line_width=edges_weights[edge_name], color='gray', alpha=0.4)
            
        #adicionando o círculo (node)
        for node_name in nodes_position_dic.keys():
            p.circle(nodes_position_dic[node_name][0], nodes_position_dic[node_name][1], size=degrees[node_name], 
                     color=nodes_color[node_name], alpha=1)

        #adicionando os textos
        text_list = []
        x_pos_list = []
        y_pos_list = []
        for node in nodes_position_dic.keys():
            text_list.append(node)
            x_pos_list.append(nodes_position_dic[node][0] + 0.005)
            y_pos_list.append(nodes_position_dic[node][1] + 0.005)        
        xytext_dic = dict(x=x_pos_list, y=y_pos_list, text=text_list)
        fontsize_scale_factor = 0.5
        offset_scale_factor=0.2
        for i in range(len(text_list)):
            label = xytext_dic['text'][i]
            x_pos = xytext_dic['x'][i]
            y_pos = xytext_dic['y'][i]
            
            citation = Label(x=x_pos, y=y_pos, angle=0, text=label,
                             x_offset=degrees[label]*offset_scale_factor, y_offset=degrees[label]*offset_scale_factor,
                             text_font_size = str(int(degrees[label]*fontsize_scale_factor)) + 'pt', 
                             text_font='helvetica',
                             text_font_style='bold',
                             text_color=nodes_color[label],
                             background_fill_alpha=1.0)
            p.add_layout(citation)

        print(f'Salvando a figura ~/Outputs/DataFrames/Plots/Plot{self.last_fig_filename_index}.png ...')
        export_png(p, filename=self.diretorio + f'/Outputs/DataFrames/Plots/Plot{self.last_fig_filename_index}.png')
        
        
        #atualizando o número do plot_index
        self.last_fig_filename_index = ( ( len('0000') - len(str(int(self.last_fig_filename_index)+1)) ) * '0' ) + str(int(self.last_fig_filename_index) + 1)

    
    
    def chord_plot(self, grouped_DF):
        
        import numpy as np
        import pandas as pd
        import holoviews as hv
        from holoviews import opts, dim
        
        def get_first_column_names(term):            
            import regex as re
            split_term = re.findall(r'[\w\s]+(?=_)', term)
            return split_term[0]

        def get_second_column_names(term):            
            import regex as re
            split_term = re.findall(r'(?<=_)[\w\s]+', term)
            return split_term[0] 
        
        #resetando o index        
        grouped_DF = grouped_DF.reset_index()
        grouped_DF.columns = grouped_DF.columns.get_level_values(1)
        
        #coletando em uma DF somente os valores conexão (weights)
        weights_DF = grouped_DF.copy()
        
        #mudando o dtype para a coluna weight para plotar
        weights_DF['count'] = weights_DF['count'].values.astype('int32')
        
        #associando um valor inteiro para cada label para usar o hollowviews
        sources_names_series = grouped_DF[''].apply(get_first_column_names)
        taget_names_series = grouped_DF[''].apply(get_second_column_names)
        all_names = list(np.unique(np.hstack([sources_names_series.values, taget_names_series.values])))

        #função para substiuição        
        names_dic = {}
        for i in range(len(all_names)):
            names_dic[all_names[i]] = i

        def replace_name_by_numbers(name):
            return names_dic[name]
        
        weights_DF['source'] = sources_names_series.apply(replace_name_by_numbers)
        weights_DF['target'] = taget_names_series.apply(replace_name_by_numbers)

        #adicionando os nodes labels a um DF
        nodes_labels_df = pd.DataFrame([])
        nodes_labels_df['index'] = names_dic.values()
        nodes_labels_df['name'] = names_dic.keys()
        
        #plotting with holoviews        
        hv.extension('bokeh')
        hv.output(size=500)
        #chord = hv.Chord(concat_network_relat)
        
        nodes = hv.Dataset(nodes_labels_df, 'index')
        chord = hv.Chord((weights_DF[['source', 'target', 'count']].copy(), nodes)).select(value=(5, None))
        chord.opts(opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('source').str(), 
                              labels='name', node_color=dim('index').str()))
        chord.opts(label_text_font_size='15pt')
        
        print(f'Salvando a figura ~/Outputs/DataFrames/Plots/Plot{self.last_fig_filename_index}.png ...')
        hv.save(chord, self.diretorio + f'/Outputs/DataFrames/Plots/Plot{self.last_fig_filename_index}.png', backend='bokeh')
        
        #atualizando o número do plot_index
        self.last_fig_filename_index = ( ( len('0000') - len(str(int(self.last_fig_filename_index)+1)) ) * '0' ) + str(int(self.last_fig_filename_index) + 1)



    def group_num_boxplot(self, grouped_DF = None, input1 = '', input2 = '', axes_label1 = '', axes_label2 = '', ymin = None, ymax = None, 
                          size_factor = 1, grouplabel_x_offset = 0, colobar_nticks = 20, plot_width=800, plot_height=1200):
        
        import numpy as np
        from bokeh.models import Label
        from bokeh.models.ranges import Range1d
        from bokeh.plotting import figure
        from bokeh.palettes import viridis, turbo, plasma
        from bokeh.io import export_png
        
        print('input1: ', input1)
        print('input2: ', input2)
                
        cats = grouped_DF.describe().index.values
        cats_counter = grouped_DF.describe()['count'].values
        cats_number = len(cats)
        max_cat_allowed = 50
        if cats_number > max_cat_allowed:
            print(f'ERRO! Reduzir para {max_cat_allowed} o número de categorias para o group_num_plot.')
            print('Número atual: ', cats_number)
            return
        
        #encontrando os valores de percentiles para cada grupo
        qmin = grouped_DF.quantile(q=0.00)
        q1 = grouped_DF.quantile(q=0.25)
        q2 = grouped_DF.quantile(q=0.5)
        q3 = grouped_DF.quantile(q=0.75)
        qmax = grouped_DF.quantile(q=1.00)
        iqr = q3 - q1
        upper = q3 + 1.5*iqr
        lower = q1 - 1.5*iqr
        
        # find the outliers for each category
        def outliers(group):
            cat = group.name
            result = group[ ( group > upper[cat] ) | ( group < lower[cat] ) ]
            return result
                    
        out = grouped_DF.apply(outliers).dropna()
                
        # prepare outlier data for plotting, we need coordinates for every outlier.
        if not out.empty:
            outx = []
            outy = []
            for keys in out.index:
                outx.append(keys[0])
                outy.append(out.loc[keys[0]].loc[keys[1]].loc[keys[2]])
            
        #plotando o boxplot
        p = figure(toolbar_location=None, 
                   #title='.',
                   match_aspect=False,
                   x_range=cats,                  
                   background_fill_color='white', 
                   plot_width=plot_width, plot_height=plot_height,
                   min_border=200)

        #general settings
        p.grid.visible = True
        font = 'helvetica'
        p.title.text_font_size = str(int(cats_number*size_factor) + 2) + 'pt'
        p.outline_line_color = "white"
        p.outline_line_width = 0
        p.axis.axis_line_width = 2
        p.axis.axis_line_color = 'red'
        p.axis.major_tick_line_color = 'red'
        p.axis.major_tick_line_width = 3
        p.axis.minor_tick_line_color = 'red'
        p.axis.minor_tick_line_width = 2
        p.axis.minor_tick_in = -2
        p.axis.minor_tick_out = 6
        
        #x
        p.xaxis.axis_label = axes_label1
        p.xaxis.axis_label_text_font = font
        p.xaxis.axis_label_text_color = 'black'
        p.xaxis.axis_label_text_font_size = str(int(cats_number*size_factor) + 0) + 'pt'
        p.xaxis.axis_label_standoff = 10
        p.xaxis.major_label_text_color = "black"
        p.xaxis.major_label_orientation = np.pi/3
        p.xaxis.major_label_text_font_size = str(int(cats_number*size_factor) + 0) + 'pt'
        p.xaxis.major_label_standoff = 10
        
        #y
        p.yaxis.axis_label = axes_label2
        p.yaxis.axis_label_text_font = font
        p.yaxis.axis_label_text_color = 'black'
        p.yaxis.axis_label_text_font_size = str(int(cats_number*size_factor) + 0) + 'pt'
        p.yaxis.axis_label_standoff = 10
        p.yaxis.major_label_text_color = "black"
        p.yaxis.major_label_orientation = "horizontal"
        p.yaxis.major_label_text_font_size = str(int(cats_number*size_factor) + 0) + 'pt'
        p.yaxis.major_label_standoff = 10
        p.yaxis.bounds = (ymin, ymax)
        p.ygrid.grid_line_color = 'gray'
        p.ygrid.grid_line_alpha = 0.1
        p.ygrid.minor_grid_line_color = 'gray'
        p.ygrid.minor_grid_line_alpha = 0.1    
        p.y_range = Range1d(ymin, ymax+(ymax*0.4))
        p.yaxis.ticker = [round(val, 0) for val in np.arange(ymin, ymax + (ymax*0.05), (ymax-ymin+1)/6) ]
        

        # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
        upper_values = [min([x,y]) for (x,y) in zip(list(qmax.values), upper.values)]
        lower_values = [max([x,y]) for (x,y) in zip(list(qmin.values), lower.values)]
                
        # stems
        p.segment(cats, upper_values, cats, q3.values, line_width=3, line_color="black")
        p.segment(cats, lower_values, cats, q1.values, line_width=3, line_color="black")
        
        #palettes
        #paleta de cores
        colors1 = plasma(len(q1.values))
        colors2 = turbo(len(q1.values))
        
        # boxes
        p.vbar(cats, 0.7, q2.values, q3.values, line_width=2, fill_color=colors1, line_color="white")
        p.vbar(cats, 0.7, q1.values, q2.values, line_width=2, fill_color=colors2, line_color="white")
        
        # whiskers (almost-0 height rects simpler than segments)
        p.rect(cats, lower_values, width=0.3, height=0.1, line_width=3, line_color="black")
        p.rect(cats, upper_values, width=0.3, height=0.1, line_width=3, line_color="black")
        
        # outliers
        if not out.empty:
            p.circle(outx, outy, size=11, color="red", fill_alpha=0.6)
            
        #adicionando o count sobre as barras
        citation_yval = max(outy) + (max(outy)*0.1)
        for i in range(len(cats_counter)):
            citation = Label(x=i+cats_number*size_factor*0.025, x_offset=grouplabel_x_offset, 
                             y=citation_yval, angle=np.pi/2, text=str(int(cats_counter[i])),
                             text_font_size = str(int(cats_number*size_factor) + 0) + 'pt', background_fill_alpha=1.0)
            p.add_layout(citation)
            
        
        print(f'Salvando a figura ~/Outputs/DataFrames/Plots/Plot{self.last_fig_filename_index}.png ...')
        export_png(p, filename=self.diretorio + f'/Outputs/DataFrames/Plots/Plot{self.last_fig_filename_index}.png')
        
        #atualizando o número do plot_index
        self.last_fig_filename_index = ( ( len('0000') - len(str(int(self.last_fig_filename_index)+1)) ) * '0' ) + str(int(self.last_fig_filename_index) + 1)
        


    def cat_cat_gridplot(self, DF = None, DF_x_column = '', DF_y_column = '', DF_val_column = '', 
                         min_occurrences = 0, size_factor = 0.4, colobar_nticks = 20, palette = 'Viridis256', 
                         background_fill_color = 'blue', plot_width=1000, plot_height=1000):
        
        from bokeh.models import BasicTicker, ColorBar, LinearColorMapper
        from bokeh.plotting import figure
        from bokeh.io import export_png
        import numpy as np
        import time

        proc_vals = len(DF.index)
        print('vals_number: ', proc_vals)
        
        #determinando os valores mínimos e máximos de ocorrências
        lowest_val , highest_val = DF[DF_val_column].values.min(), DF[DF_val_column].values.max()

        #valores de x e y                
        x_vals = sorted(list(np.unique(DF[DF_x_column].values)))
        y_vals = sorted(list(np.unique(DF[DF_y_column].values)))[::-1]
        delta_x = len(x_vals)
        delta_y = len(y_vals)
        max_axis_vals = max(len(x_vals),len(y_vals))

        #determinando os fatores para padronizar o tamanho do gráfico
        if delta_x / delta_y  > 1:
            multp_fac_x = 1
            multp_fac_y = delta_x / delta_y
        elif delta_x / delta_y < 1:
            multp_fac_x = delta_y / delta_x
            multp_fac_y = 1            
        else:
            multp_fac_x = 1
            multp_fac_y = 1

        p = figure(toolbar_location=None,
                   title='vals_number = ' + str(proc_vals),
                   background_fill_color = background_fill_color,
                   min_border = 200,
                   x_range=x_vals, y_range=y_vals,
                   x_axis_location="above", 
                   plot_width=plot_width, plot_height=plot_height)
        
        #definições gerais
        p.title.text_font_size = str(int(max_axis_vals*size_factor) + 2) + 'pt'
        p.outline_line_color = "black"
        p.outline_line_width = 1
        p.grid.grid_line_color = 'gray'
        p.grid.grid_line_width = 1
        p.grid.grid_line_alpha = 0.0
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = str(int(max_axis_vals*size_factor) + 0) + 'pt'
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = np.pi/3

        # color mapping
        mapper = LinearColorMapper(palette=palette,
                                   low=lowest_val, 
                                   high=highest_val)
        
        p.rect(x=DF_x_column, y=DF_y_column, width=1, height=1,
               source=DF,
               fill_color={'field': DF_val_column, 'transform': mapper},
               line_color=None)
        
        #colorbar
        color_bar = ColorBar(color_mapper=mapper,
                             ticker=BasicTicker(desired_num_ticks=colobar_nticks),
                             orientation = 'horizontal',
                             label_standoff=6, border_line_color=None, location=(0, 0))
        color_bar.title_text_font_size = str(int(max_axis_vals*size_factor) + 0) + 'pt'
        color_bar.major_label_text_font_size = str(int(max_axis_vals*size_factor) + 0) + 'pt'
        color_bar.bar_line_color = 'black'
        color_bar.bar_line_width = 0
        color_bar.major_tick_line_alpha = 0
        color_bar.major_tick_out = int(max_axis_vals*size_factor*0.8)
        color_bar.title = f'occurrences (min = {min_occurrences})'
        color_bar.title_text_font_size = str(int(max_axis_vals*size_factor) + 0) + 'pt'
        
        p.add_layout(color_bar, 'above')
        
        print(f'Salvando a figura ~/Outputs/DataFrames/Plots/Plot{self.last_fig_filename_index}.png ...')
        export_png(p, filename=self.diretorio + f'/Outputs/DataFrames/Plots/Plot{self.last_fig_filename_index}.png')
        
        #atualizando o número do plot_index
        self.last_fig_filename_index = ( ( len('0000') - len(str(int(self.last_fig_filename_index)+1)) ) * '0' ) + str(int(self.last_fig_filename_index) + 1)
                
    
    
    def num_num_hexbinplot(self, DF = None, PUs = ['', ''], input1 = '', input2 = '', 
                           x_min = None, x_max = None, y_min = None, y_max = None, hex_size = 10,
                           plot_width=1000, plot_height=1000):

        import regex as re
        import numpy as np
        from bokeh.layouts import gridplot
        from bokeh.models import BasicTicker, ColorBar, LinearColorMapper
        from bokeh.plotting import figure
        from bokeh.io import export_png
        
        print('input1: ', input1)
        print('input2: ', input2)
        proc_vals = len(DF.index)
        print('vals_number: ', proc_vals)
        
        #coletando os resultados de interesse
        x = DF[input1].values
        y = DF[input2].values

        print('xmin found (aut) = ', DF[input1].values.min())
        print('xmax found (aut) = ', DF[input1].values.max())
        print('ymin found (aut) = ', DF[input2].values.min())
        print('ymax found (aut) = ', DF[input2].values.max())
        
        #xmin
        if x_min is None:
            xmin = DF[input1].values.min()
        else:
            xmin = x_min
        #xmax
        if x_max is None:
            xmax = DF[input1].values.max()
        else:
            xmax = x_max        
        #ymin
        if y_min is None:
            ymin = DF[input2].values.min()
        else:
            ymin = y_min
        #ymax
        if y_max is None:
            ymax = DF[input2].values.max()
        else:
            ymax = y_max
                    
        print('xmin inserted = ', xmin)
        print('xmax inserted = ', xmax)
        print('ymin inserted = ', ymin)
        print('ymax inserted = ', ymax)
        
        delta_limits = 100
        input1_PU , input2_PU = PUs
        input1_replaced = re.sub(r'_', ' ', input1)
        input2_replaced = re.sub(r'_', ' ', input2)

        #plotando o scatter plot
        p = figure(toolbar_location=None, 
                   title='vals_number = ' + str(proc_vals),
                   match_aspect=False,
                   background_fill_color='white', 
                   x_range=(xmin - delta_limits, xmax + delta_limits), y_range=(ymin - delta_limits, ymax + delta_limits),
                   plot_width=plot_width, plot_height=plot_height,
                   min_border=50)

        #general settings
        p.grid.visible = False
        font = 'helvetica'
        p.title.text_font_size = '22pt'
        p.outline_line_color = "white"
        p.outline_line_width = 0
        p.axis.axis_line_width = 2
        p.axis.axis_line_color = 'red'
        p.axis.major_tick_line_color = 'red'
        p.axis.major_tick_line_width = 3
        p.axis.minor_tick_line_color = 'red'
        p.axis.minor_tick_line_width = 2
        p.axis.minor_tick_in = -2
        p.axis.minor_tick_out = 6
        
        #x
        p.xaxis.axis_label = input1_replaced + ' (' + input1_PU + ')'
        p.xaxis.axis_label_text_font = font
        p.xaxis.axis_label_text_color = 'black'
        p.xaxis.axis_label_text_font_size = '25pt'
        p.xaxis.axis_label_standoff = 10
        p.xaxis.major_label_text_color = "black"
        p.xaxis.major_label_orientation = "horizontal"
        p.xaxis.major_label_text_font_size = '22pt'
        p.xaxis.major_label_standoff = 10
        p.xaxis.bounds = (xmin, xmax)
        p.xgrid.grid_line_color = 'gray'
        p.xgrid.grid_line_alpha = 0.1
        p.xgrid.minor_grid_line_color = 'gray'
        p.xgrid.minor_grid_line_alpha = 0.1
        p.xaxis.ticker = [int(val) for val in np.arange(xmin, xmax + (xmax*0.05), (xmax-xmin+1)/6) ]
        
        #y
        p.yaxis.axis_label = input2_replaced + ' (' + input2_PU + ')'
        p.yaxis.axis_label_text_font = font
        p.yaxis.axis_label_text_color = 'black'
        p.yaxis.axis_label_text_font_size = '25pt'
        p.yaxis.axis_label_standoff = 10
        p.yaxis.major_label_text_color = "black"
        p.yaxis.major_label_orientation = "vertical"
        p.yaxis.major_label_text_font_size = '22pt'
        p.yaxis.major_label_standoff = 10
        p.yaxis.bounds = (ymin, ymax)
        p.ygrid.grid_line_color = 'gray'
        p.ygrid.grid_line_alpha = 0.1
        p.ygrid.band_fill_color = "gray"
        p.ygrid.band_fill_alpha = 0.1
        p.ygrid.minor_grid_line_color = 'gray'
        p.ygrid.minor_grid_line_alpha = 0.1
        p.yaxis.ticker = [int(val) for val in np.arange(ymin, ymax + (ymax*0.05), (ymax-ymin+1)/6) ]

        #plotando
        r, bins = p.hexbin(x, y, size=hex_size, palette=self.palette_bins[0])
        p.circle(x, y, color="gray", size=6, alpha=0)

        #adicionando o colormap
        mapper = LinearColorMapper(palette=self.palette_bins[0], low=bins['counts'].values.min(), high=bins['counts'].values.max())
        color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="7px",
                             ticker=BasicTicker(desired_num_ticks=12),
                             orientation = 'horizontal',
                             label_standoff=12, border_line_color=None, location=(0, 0))
        color_bar.major_label_text_font_size = '22pt'
        color_bar.bar_line_color = 'black'
        color_bar.bar_line_width = 2
        color_bar.major_tick_line_alpha = 0
        color_bar.major_tick_out = 5
        color_bar.title = 'Occurrences'
        color_bar.title_text_font_size = '19pt'
        
        p.add_layout(color_bar, 'above')    
    
        #plotando o histograma de cima
        hhist, hedges = np.histogram(x, bins=20, range=(xmin,xmax))        
        hmax = max(hhist)*1.1

        ph = figure(toolbar_location=None, plot_width=p.plot_width, plot_height=200, x_range=p.x_range,
                    y_range=( 0 - (0.1*hmax), hmax), min_border=50)
        ph.outline_line_color = "white"
        ph.outline_line_width = 0
        ph.grid.visible = False
        ph.yaxis.major_label_orientation = 'horizontal'
        ph.background_fill_color = "white"        
        ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="white", line_color="red")
        ph.xaxis.visible = False
        ph.yaxis.major_label_text_color = "black"
        ph.yaxis.major_label_text_font_size = '21pt'
        ph.yaxis.ticker = (0, int(hmax))
        ph.yaxis.bounds = (0, int(hmax))
        ph.yaxis.axis_label_text_font_size = '21pt'
        ph.yaxis.axis_label = 'freq'
        
        #plotando o histograma do lado
        vhist, vedges = np.histogram(y, bins=50, range=(ymin, ymax))
        vmax = max(vhist)*1.1
        
        pv = figure(toolbar_location=None, plot_width=200, plot_height=p.plot_height, x_range=( 0 - (0.1*vmax), vmax),
                    y_range=p.y_range, min_border=50)
        pv.outline_line_color = "white"
        pv.outline_line_width = 0
        pv.grid.visible = False
        pv.xaxis.major_label_orientation = np.pi/4
        pv.background_fill_color = "white"        
        pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, color="white", line_color="red")
        pv.yaxis.visible = False
        pv.xaxis.major_label_text_color = "black"
        pv.xaxis.major_label_text_font_size = '21pt'
        pv.xaxis.ticker = (0, int(vmax))
        pv.xaxis.bounds = (0, int(vmax))
        pv.xaxis.axis_label_text_font_size = '21pt'
        pv.xaxis.axis_label = 'freq'
        
        layout = gridplot([[p, pv], [ph, None]], merge_tools=False)
        
        print(f'Salvando a figura ~/Outputs/DataFrames/Plots/Plot{self.last_fig_filename_index}.png ...')
        export_png(layout, filename=self.diretorio + f'/Outputs/DataFrames/Plots/Plot{self.last_fig_filename_index}.png')
        
        #atualizando o número do plot_index
        self.last_fig_filename_index = ( ( len('0000') - len(str(int(self.last_fig_filename_index)+1)) ) * '0' ) + str(int(self.last_fig_filename_index) + 1)


    
    def group_num_num_hexbinplot(self, DF = None, grouped = None, PUs = ['', ''], input1 = '', input2 = '', 
                                 x_min = None, x_max = None, y_min = None, y_max = None,
                                 hex_size = 10, plot_width=1000, plot_height=1000):

        import regex as re
        import numpy as np
        from bokeh.layouts import gridplot
        from bokeh.models.ranges import Range1d
        from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, Label
        from bokeh.plotting import figure
        from bokeh.io import export_png
        
        print('input1: ', input1)
        print('input2: ', input2)
        proc_vals = len(DF.index)
        print('vals_number: ', proc_vals)

        print('xmin found (aut) = ', DF[input1].values.min())
        print('xmax found (aut) = ', DF[input1].values.max())
        print('ymin found (aut) = ', DF[input2].values.min())
        print('ymax found (aut) = ', DF[input2].values.max())        
        
        #xmin
        if x_min is None:
            xmin = DF[input1].values.min()
        else:
            xmin = x_min
        #xmax
        if x_max is None:
            xmax = DF[input1].values.max()
        else:
            xmax = x_max        
        #ymin
        if y_min is None:
            ymin = DF[input2].values.min()
        else:
            ymin = y_min
        #ymax
        if y_max is None:
            ymax = DF[input2].values.max()
        else:
            ymax = y_max
        
        print('xmin inserted = ', xmin)
        print('xmax inserted = ', xmax)
        print('ymin inserted = ', ymin)
        print('ymax inserted = ', ymax)
                
        delta_limits = 100
        input1_PU , input2_PU = PUs
        input1_replaced = re.sub(r'_', ' ', input1)
        input2_replaced = re.sub(r'_', ' ', input2)

        #plotando o scatter plot
        p = figure(toolbar_location=None, 
                   #title='vals_number = ' + str(proc_vals),
                   match_aspect=False,
                   background_fill_color='white', 
                   x_range=(xmin - delta_limits, xmax + delta_limits), y_range=(ymin - delta_limits, ymax + delta_limits),
                   plot_width=plot_width, plot_height=plot_height,
                   min_border=50)

        #general settings
        p.grid.visible = False
        font = 'helvetica'
        p.title.text_font_size = '22pt'
        p.outline_line_color = "white"
        p.outline_line_width = 0
        p.axis.axis_line_width = 2
        p.axis.axis_line_color = 'red'
        p.axis.major_tick_line_color = 'red'
        p.axis.major_tick_line_width = 3
        p.axis.minor_tick_line_color = 'red'
        p.axis.minor_tick_line_width = 2
        p.axis.minor_tick_in = -2
        p.axis.minor_tick_out = 6
        
        #x
        p.xaxis.axis_label = input1_replaced + ' (' + input1_PU + ')'
        p.xaxis.axis_label_text_font = font
        p.xaxis.axis_label_text_color = 'black'
        p.xaxis.axis_label_text_font_size = '25pt'
        p.xaxis.axis_label_standoff = 10
        p.xaxis.major_label_text_color = "black"
        p.xaxis.major_label_orientation = "horizontal"
        p.xaxis.major_label_text_font_size = '22pt'
        p.xaxis.major_label_standoff = 10
        p.xaxis.bounds = (xmin, xmax)
        p.xgrid.grid_line_color = 'gray'
        p.xgrid.grid_line_alpha = 0.1
        p.xgrid.minor_grid_line_color = 'gray'
        p.xgrid.minor_grid_line_alpha = 0.1
        p.xaxis.ticker = [int(val) for val in np.arange(xmin, xmax + (xmax*0.05), (xmax-xmin+1)/6) ]
        
        #y
        p.yaxis.axis_label = input2_replaced + ' (' + input2_PU + ')'
        p.yaxis.axis_label_text_font = font
        p.yaxis.axis_label_text_color = 'black'
        p.yaxis.axis_label_text_font_size = '25pt'
        p.yaxis.axis_label_standoff = 10
        p.yaxis.major_label_text_color = "black"
        p.yaxis.major_label_orientation = "vertical"
        p.yaxis.major_label_text_font_size = '22pt'
        p.yaxis.major_label_standoff = 10
        p.yaxis.bounds = (ymin, ymax)
        p.ygrid.grid_line_color = 'gray'
        p.ygrid.grid_line_alpha = 0.1
        p.ygrid.band_fill_color = "gray"
        p.ygrid.band_fill_alpha = 0.1
        p.ygrid.minor_grid_line_color = 'gray'
        p.ygrid.minor_grid_line_alpha = 0.1
        p.yaxis.ticker = [int(val) for val in np.arange(ymin, ymax + (ymax*0.05), (ymax-ymin+1)/6) ]

        #plotando o histograma de cima
        ph = figure(toolbar_location=None, plot_width=p.plot_width, plot_height=200,
                    min_border=50)
        ph.outline_line_color = "white"
        ph.outline_line_width = 0
        ph.grid.visible = False
        ph.yaxis.major_label_orientation = 'horizontal'
        ph.background_fill_color = "white"        
        ph.xaxis.visible = False
        ph.yaxis.major_label_text_color = "black"
        ph.yaxis.major_label_text_font_size = '21pt'
        ph.yaxis.axis_label_text_font_size = '21pt'
        ph.yaxis.axis_label = 'freq'

        #plotando o histograma de lado
        pv = figure(toolbar_location=None, plot_width=200, plot_height=p.plot_height,
                    min_border=50)
        pv.outline_line_color = "white"
        pv.outline_line_width = 0
        pv.grid.visible = False
        pv.xaxis.major_label_orientation = np.pi/4
        pv.background_fill_color = "white"        
        pv.yaxis.visible = False
        pv.xaxis.major_label_text_color = "black"
        pv.xaxis.major_label_text_font_size = '21pt'
        pv.xaxis.axis_label_text_font_size = '21pt'
        pv.xaxis.axis_label = 'freq'

        #color bars        
        c_bars = figure(toolbar_location=None, plot_width=p.plot_width, plot_height=200,
                    x_range=(0, (len(self.palette_bins[0])*10) + 60), min_border=50)
        c_bars.outline_line_color = "white"
        c_bars.outline_line_width = 0
        c_bars.grid.visible = False
        c_bars.background_fill_color = "white"
        c_bars.yaxis.visible = False
        c_bars.xaxis.visible = False

        #plotando
        counter = 0
        hmax = 0
        vmax = 0
        for group in grouped.groups:
            grouped_vals = grouped.get_group(group)
            #coletando os resultados de interesse
            x = grouped_vals[input1].values
            y = grouped_vals[input2].values
            n_vals = len(grouped_vals[input2].values)
        
            r, bins = p.hexbin(x, y, size=hex_size, palette=self.palette_bins[counter])
            p.circle(x, y, color="gray", size=6, alpha=0)
    
            '''
            #adicionando o colormap
            mapper = LinearColorMapper(palette=self.palette_bins[counter], low=bins['counts'].values.min(), high=bins['counts'].values.max())
            color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="7px",
                                 ticker=BasicTicker(desired_num_ticks=12),
                                 orientation = 'horizontal',
                                 label_standoff=12, border_line_color=None, location=(0, 0))
            color_bar.major_label_text_font_size = '22pt'
            color_bar.bar_line_color = 'black'
            color_bar.bar_line_width = 2
            color_bar.major_tick_line_alpha = 0
            color_bar.major_tick_out = 5
            color_bar.title = 'Occurrences'
            color_bar.title_text_font_size = '19pt'
            
            p.add_layout(color_bar, 'above')'''
        
            #plotando o histograma de cima
            hhist, hedges = np.histogram(x, bins=20, range=(xmin,xmax))        
            if max(hhist)*1.1 > hmax:
                hmax = max(hhist)*1.1
            ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color=self.palette_bins[counter][-1], alpha=0.5)
            
            #plotando o histograma do lado
            vhist, vedges = np.histogram(y, bins=50, range=(ymin, ymax))
            if max(vhist)*1.1 > vmax:
                vmax = max(vhist)*1.1
            pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, color=self.palette_bins[counter][-1], alpha=0.5)

            #adicionando a legenda no gráfico acima
            for color_i in range(len(self.palette_bins[counter])):
                rect_color = c_bars.rect(10+(color_i*10), 0-(counter*10), 10, 5, line_color = 'white', fill_color=self.palette_bins[counter][color_i])            
                c_bars.add_layout(rect_color)

            y_base_pos = -7.5
            leg_label1 = Label(x=0, y=y_base_pos-(counter*10), text='0',
                              text_font_size = '16pt', text_color = 'gray', background_fill_alpha=0.8)            
            leg_label2 = Label(x=7+(len(self.palette_bins[0])*10), y=y_base_pos-(counter*10), text=str(n_vals),
                              text_font_size = '16pt', text_color = 'gray', background_fill_alpha=0.8)
            leg_label3 = Label(x=30+(len(self.palette_bins[0])*10), y=y_base_pos-(counter*10), text=re.sub(r'_', ' ', group),
                               text_font_size = '16pt', text_color = self.palette_bins[counter][-1], background_fill_alpha=0.8)
            c_bars.add_layout(leg_label1)
            c_bars.add_layout(leg_label2)
            c_bars.add_layout(leg_label3)
            
            counter += 1
            
        
        #ajustando a escala dos histogramas após a plotagem
        ph.x_range = p.x_range
        ph.y_range = Range1d( 0 - (0.1*hmax), hmax)
        ph.yaxis.ticker = (0, int(hmax))
        ph.yaxis.bounds = (0, int(hmax))
        pv.x_range = Range1d( 0 - (0.1*vmax), vmax)
        pv.y_range = p.y_range
        pv.xaxis.ticker = (0, int(vmax))
        pv.xaxis.bounds = (0, int(vmax))
        
        layout = gridplot([[c_bars,None], [p, pv], [ph, None]], merge_tools=False)
        
        print(f'Salvando a figura ~/Outputs/DataFrames/Plots/Plot{self.last_fig_filename_index}.png ...')
        export_png(layout, filename=self.diretorio + f'/Outputs/DataFrames/Plots/Plot{self.last_fig_filename_index}.png')
        
        #atualizando o número do plot_index
        self.last_fig_filename_index = ( ( len('0000') - len(str(int(self.last_fig_filename_index)+1)) ) * '0' ) + str(int(self.last_fig_filename_index) + 1)