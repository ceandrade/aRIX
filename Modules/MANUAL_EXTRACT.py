#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from DFs import DataFrames
diretorio = '/media/ext4_storage2/Biochar/NLP'
filename = 'surface_area_4'
search_mode = 'search_with_combined_models'
parameter = 'surface_area'

DF = DataFrames(mode = 'collect_parameters_automatic', 
                consolidated_DataFrame_name = f'{filename}_FULL_DF', 
                save_to_GoogleDrive = False, 
                diretorio = diretorio)
DF.set_settings(input_DF_name = filename,
                search_mode = search_mode,
                output_DF_name = filename,
                parameters = [parameter],
                hold_samples = False,
                hold_sample_number = False,
                numbers_extraction_mode = 'all',
                ngram_for_textual_search = 2,
                min_ngram_appearence = 5,
                match_all_sent_findings = False,
                get_avg_num_results = True)
#use o regex para pegar parâmetros numéricos dentro das sentenças
DF.get_data(max_token_in_sent = 100)