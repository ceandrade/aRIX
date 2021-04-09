#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def main():
    
    import argparse
      
    #Função principal   
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-s', '--search_input_index', default = 'None', help = 'Introduzir o index com os inputs de SEARCH/EXTRACT (manipular arquivo em ~/Settings/search_inputs.csv).', type = str)
    parser.add_argument('-d', '--diretorio', default = 'None', help ='Introduzir o Master Folder do programa (manipular arquivo em ~/Settings/search_inputs.csv)..', type = str)
    
    args = parser.parse_args()

    process(args.search_input_index, args.diretorio)


    
def process(search_input_index, diretorio):

    print('=============================')
    print('SEARCH_EXTRACT MODE')
    print('=============================')
    

    from FUNCTIONS import extract_inputs_from_csv
    SE_inputs = extract_inputs_from_csv(csv_filename = 'SE_inputs', diretorio = diretorio, mode = 'search_extract')
    search_input_index = int(search_input_index)
    print()
    print('> SE_input line index: ', search_input_index)
    
    #definindo as variáveis
    print('> filename: ', SE_inputs[search_input_index]['filename'])
    print('> parameter: ', SE_inputs[search_input_index]['parameter_to_extract'])
    print('> search_mode: ', SE_inputs[search_input_index]['search_mode'].lower())
    for key in SE_inputs[search_input_index]['search_inputs'].keys():
        print('> ', key, ': ' , SE_inputs[search_input_index]['search_inputs'][key])
    
    go_to_extract = False
    
    from SCHENG import search_engine
    
    #inserir a combinação de procura no arquivo /Settings/SE_inputs.csv
    if SE_inputs[search_input_index]['search_mode'].lower() == 'search_with_combined_models':
        go_to_extract = True
        if SE_inputs[search_input_index]['search_status'].lower() != 'finished':
            print('> Go to search engine...')
            se = search_engine(diretorio = diretorio)
            se.set_search_conditions(search_input_dic = SE_inputs[search_input_index]['search_inputs'],
                                     search_mode = SE_inputs[search_input_index]['search_mode'].lower(),
                                     output_DF_name = SE_inputs[search_input_index]['filename'],
                                     filter_model = SE_inputs[search_input_index]['search_inputs']['filter_section'],
                                     nGrams_to_get_semantic = SE_inputs[search_input_index]['search_inputs']['semantic_ngrams'],
                                     min_ngram_semantic_appearence = SE_inputs[search_input_index]['search_inputs']['semantic_ngrams_min_app'],
                                     min_topic_corr_threshold = SE_inputs[search_input_index]['search_inputs']['topic_thr_corr_val']
                                     )
            se.search_with_combined_models()
    
    
    elif SE_inputs[search_input_index]['search_mode'].lower() == 'search_with_topic_match':
        go_to_extract = True
        if SE_inputs[search_input_index]['search_inputs']['topic'] != '()':
            if SE_inputs[search_input_index]['search_status'].lower() != 'finished':                
                print('> Go to search engine...')    
                se = search_engine(diretorio = diretorio)        
                se.set_search_conditions(search_input_dic = SE_inputs[search_input_index]['search_inputs'],
                                         search_mode = SE_inputs[search_input_index]['search_mode'].lower(),
                                         output_DF_name = SE_inputs[search_input_index]['filename'],
                                         export_random_sent = True,
                                         filter_model = SE_inputs[search_input_index]['search_inputs']['filter_section'],
                                         nGrams_to_get_semantic = SE_inputs[search_input_index]['search_inputs']['semantic_ngrams'],
                                         min_ngram_semantic_appearence = SE_inputs[search_input_index]['search_inputs']['semantic_ngrams_min_app'],
                                         topic_match_mode = SE_inputs[search_input_index]['search_inputs']['topic_match_mode'],
                                         min_topic_corr_threshold = SE_inputs[search_input_index]['search_inputs']['topic_thr_corr_val']
                                         )
                se.search_with_topic_match()
        
        else:
            print('Erro! O modo "search_with_topic_match" precisa conter termos para procura em tópico.')


    if go_to_extract is True and SE_inputs[search_input_index]['export_status'].lower() != 'finished':
        
        from DFs import DataFrames
        
        DF = DataFrames(mode = 'collect_parameters_automatic', 
                        consolidated_DataFrame_name = SE_inputs[search_input_index]['filename'] + '_FULL_DF', 
                        save_to_GoogleDrive = False, 
                        diretorio = diretorio)
        DF.set_settings(input_DF_name = SE_inputs[search_input_index]['filename'],
                        search_mode = SE_inputs[search_input_index]['search_mode'].lower(),
                        output_DF_name = SE_inputs[search_input_index]['filename'],
                        parameters = [SE_inputs[search_input_index]['parameter_to_extract']],
                        ngram_for_textual_search = SE_inputs[search_input_index]['ngrams_to_extract'],
                        min_ngram_appearence = SE_inputs[search_input_index]['ngrams_min_app'],
                        hold_samples = False,
                        hold_sample_number = False,
                        numbers_extraction_mode = 'all',
                        match_all_sent_findings = False,
                        get_avg_num_results = False)
        #use o regex para pegar parâmetros numéricos dentro das sentenças
        DF.get_data(max_token_in_sent = 100)



###############################################################################################
#executando a função
main()
