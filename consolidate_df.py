#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def main():
    
    import argparse
      
    #Função principal   
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--diretorio', default = 'None', help ='Introduzir o Master Folder do programa (manipular arquivo em ~/Settings/search_inputs.csv)..', type = str)
    
    args = parser.parse_args()

    process(args.diretorio)


    
def process(diretorio):

    import sys
    sys.path.append('/media/ext4_storage2/Biochar/NLP/Modules')
    program_folder = '/media/ext4_storage2/Biochar/NLP'
    
    from DFs import DataFrames    
    from FUNCTIONS import extract_inputs_from_csv
    inputs_to_consolidate = extract_inputs_from_csv(csv_filename = 'DFs_to_consolidate', diretorio = program_folder, mode = 'consolidate_df')
    
    for i in inputs_to_consolidate.keys():        

        filename = inputs_to_consolidate[i]['filename']
        parameter = inputs_to_consolidate[i]['parameter_to_extract']
        
        DF = DataFrames(mode = 'collect_parameters_automatic', consolidated_DataFrame_name = '_CONSOLIDATED_DF', save_to_GoogleDrive = False, diretorio = diretorio)
        DF.set_settings(input_DF_name = filename,
                        search_mode = inputs_to_consolidate[i]['search_mode'].lower(),
                        output_DF_name = 'consolidated_' + filename,
                        parameters = [parameter],
                        hold_samples = inputs_to_consolidate[i]['hold_samples'],
                        hold_sample_number = inputs_to_consolidate[i]['hold_sample_number'],
                        numbers_extraction_mode = inputs_to_consolidate[i]['numbers_extraction_mode'],
                        filter_unique_results = inputs_to_consolidate[i]['filter_unique_results'],
                        ngram_for_textual_search = inputs_to_consolidate[i]['ngram_for_textual_search'],
                        min_ngram_appearence = inputs_to_consolidate[i]['min_ngram_appearence'],
                        match_all_sent_findings = inputs_to_consolidate[i]['match_all_sent_findings'],
                        get_avg_num_results = inputs_to_consolidate[i]['get_avg_num_results'])
        #use o regex para pegar parâmetros numéricos dentro das sentenças
        DF.get_data(max_token_in_sent = 100)
    


###############################################################################################
#executando a função
main()
