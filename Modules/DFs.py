#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class DataFrames(object):
    
    def __init__(self, mode = 'select_parameters_manual', consolidated_DataFrame_name = 'ConsolDF' , save_to_GoogleDrive = False, diretorio = None):
        
        print('\n=============================\nDATA FRAMES...\n=============================\n')
        print('( Class: DataFrames )')
        

        import pandas as pd
        from GoogleDriveAPI import gDrive

        self.diretorio = diretorio
        self.class_name = 'DataFrames'        
        self.mode = mode.lower()
        self.consolidated_DataFrame_name = consolidated_DataFrame_name
        self.save_to_GDrive = save_to_GoogleDrive
        if self.save_to_GDrive == True:
            self.drive = gDrive()

        #testando os erros de inputs
        from FUNCTIONS import error_incompatible_strings_input        
        self.abort_class = error_incompatible_strings_input('mode', mode, ('collect_parameters_automatic', 'collect_parameters_manual', 'select_sentences'), class_name = self.class_name)
                
        print(f'Mode chose: {mode}')            


    def set_settings(self, 
                     input_DF_name = 'frags_extracted', 
                     search_mode = '',
                     output_DF_name = 'Paramaters', 
                     parameters = ['temperature_pyrolysis'], 
                     hold_samples = False, 
                     hold_sample_number = False,
                     ngram_for_textual_search = 2,
                     min_ngram_appearence = 5,
                     numbers_extraction_mode = 'first',
                     match_all_sent_findings = True,
                     get_avg_num_results = False):
        
        print('( Function: set_files )')
        print('Setting Files...')
        
        self.input_DF_name = input_DF_name
        self.search_mode = search_mode
        self.output_DF_name = output_DF_name
        self.input_parameters_list = parameters
        self.hold_samples = hold_samples
        self.hold_sample_number = hold_sample_number
        self.min_ngram_appearence = min_ngram_appearence
        self.ngram_for_textual_search = ngram_for_textual_search
        self.numbers_extraction_mode = numbers_extraction_mode
        self.match_all_sent_findings = match_all_sent_findings
        self.get_avg_num_results = get_avg_num_results

    
    def get_data(self, max_token_in_sent = 100):

        print('( Function: get_data )')
        
        #checando erros de instanciação/inputs
        from FUNCTIONS import error_print_abort_class
        if self.abort_class is True:        
            error_print_abort_class(self.class_name)
            return
        from FUNCTIONS import create_PDFFILE_list
        from FUNCTIONS import error_incompatible_strings_input
        from FUNCTIONS import load_dic_from_json
        from FUNCTIONS import create_DF_columns_names_with_index
        from functions_PARAMETERS import list_numerical_parameter
        from functions_PARAMETERS import list_textual_parameter
        import regex as re
        import time
        import pandas as pd
        import webbrowser as wb
        import os

        #checando se existe um Data Frame de fragmentos
        print(f'Procurando... {self.diretorio}/Outputs/Extracted/' + f'{self.input_DF_name}_extracted_mode_{self.search_mode}.csv')
        if os.path.exists(self.diretorio + '/Outputs/Extracted/' + f'{self.input_DF_name}_extracted_mode_{self.search_mode}.csv'):
            self.extracted_fragsDF = pd.read_csv(self.diretorio + '/Outputs/Extracted/' + f'{self.input_DF_name}_extracted_mode_{self.search_mode}.csv', index_col=[0,1], dtype=object)
            print(f'Carregando o DataFrame de INPUT com os frags/sents extraidos (~/Outputs/Extracted/{self.input_DF_name}_extracted_mode_{self.search_mode}.csv)')
            #se multiplica por 2 o valor de self.input_parameters_list pois uma coluna é dos indexes das sentenças
            if len(self.extracted_fragsDF.columns) / 2 == len(self.input_parameters_list):
                
                #dic para salvar as SI PUs padronizadas para cada features (ou parâmetros)
                if not os.path.exists(self.diretorio + '/Outputs/DataFrames/SI_PUs.json'):
                    self.SI_PUs_dic_to_record = {}
                else:
                    self.SI_PUs_dic_to_record = load_dic_from_json(self.diretorio + '/Outputs/DataFrames/SI_PUs.json')
                
                #definindo os inputs de parâmetros
                for filename in self.extracted_fragsDF.index.levels[0]:
                    
                    print('--------------------------------------------')
                    self.filename = filename
                    
                    #montando os nomes das colunas do DF Temporário (isso pois os nomes das colunas do DF que será exportado sairá das função de extração de dados)
                    Temp_DF_columns_name = create_DF_columns_names_with_index(self.input_parameters_list, index_columns = [], index_name = '_index')
                                        
                    #checando se já existe um Data Frame para esses parâmetros
                    if os.path.exists(self.diretorio + f'/Outputs/DataFrames/{self.output_DF_name}_mode_{self.numbers_extraction_mode}_match_{self.match_all_sent_findings}.csv'):
                        self.output_DF = pd.read_csv(self.diretorio + f'/Outputs/DataFrames/{self.output_DF_name}_mode_{self.numbers_extraction_mode}_match_{self.match_all_sent_findings}.csv', index_col=[0,1], dtype=object)
                        self.output_DF.index.names = ['Filename', 'Counter']
                        self.create_output_DF = False
                        #print(f'Carregando o DataFrame de OUTPUT (~/Outputs/Extracted/{self.output_DF_name}.csv)')
                        try:
                            last_pdf_processed = self.output_DF.index.levels[0][-1]
                        except IndexError:
                            last_pdf_processed = 'PDF00000'
                    else:
                        print(f'Output DF {self.output_DF_name}_mode_{self.numbers_extraction_mode}_match_{self.match_all_sent_findings}.csv não encontrado.')
                        print(f'Criando o output_DF data frame: {self.output_DF_name}_mode_{self.numbers_extraction_mode}_match_{self.match_all_sent_findings}.csv')
                        #caso tenha que ser gerada a output_DF
                        self.create_output_DF = True
                        last_pdf_processed = 'PDF00000'
                    
                    pdf_extracted_list = create_PDFFILE_list(last_pdf_processed)

                    #abrindo o search-extract report
                    if os.path.exists(self.diretorio + f'/Outputs/LOG/counter_SE_report_{self.input_DF_name}_extracted_mode_{self.search_mode}.json'):
                        #carregando o dicionário
                        self.search_report_dic = load_dic_from_json(self.diretorio + f'/Outputs/LOG/counter_SE_report_{self.input_DF_name}_extracted_mode_{self.search_mode}.json')
                        if self.search_report_dic['search']['searching_status'] != 'finished':
                            print(f'Erro! O processo de extração para o search_input {self.input_DF_name} ainda não terminou.' )
                            return                        
                        
                        elif last_pdf_processed == 'PDF00000':
                            self.search_report_dic['export'] = {}
                            self.search_report_dic['export']['last_PDF_processed'] = None
                            self.search_report_dic['export']['total_finds'] = 0
                            self.search_report_dic['export']['pdf_finds'] = 0
                            
                    else:
                        print('Erro! LOG counter_SE_report não encontrado em ~/outputs/LOG' )
                        print(f'Erro! O processo de extração para o search_input {self.input_DF_name} não foi feito.' )
                        return

                    
                    if self.filename not in pdf_extracted_list:
                        print('# Processando ', self.filename)                
                        
                        #dicionário para coletar os parâmetros numéricos extraídos
                        self.parameters_extracted = {}
                        self.parameters_extracted['filename'] = self.filename                    
                                            
                        #varrendo as colunas do input_DF (coluna = feature)
                        for parameter_N in range(len(self.extracted_fragsDF.columns)):
                            
                            #nome da coluna do input DF
                            input_DF_column_name = self.extracted_fragsDF.columns[parameter_N]
                            #nome do parâmetro de entrada (os parâmetros de fato sairão das sentenças)
                            input_parameter = Temp_DF_columns_name[parameter_N]
                            
                            #ignora-se as colunas dos indexes dos parâmetros
                            if input_DF_column_name[ -len('_index') : ] != '_index':                                

                                print('\nParameter column: ( ', input_parameter, ' )') 
                                print('Fragments extracted from: ', self.filename)                                

                                #criando uma key de coluna, para o index de sentença coletada e para confirmação dos dados extraídos
                                self.parameters_extracted['column' + str(parameter_N)] = {}                                                                                            
                                self.parameters_extracted['column' + str(parameter_N)]['selected_sent_index'] = None
                                self.parameters_extracted['column' + str(parameter_N)]['extracted_paramaters_from_column'] = False

                                #varrendo as linhas com as sentenças para esse PDF (input_DF)
                                for i in self.extracted_fragsDF.loc[ (self.filename , ) , ].index:                                    
                                    
                                    #checando se o número de tokens é maior que o desejado                                    
                                    sent_len = len( str(self.extracted_fragsDF.loc[ (self.filename , i ) , input_DF_column_name ]).split() )

                                    #sentença
                                    sent = self.extracted_fragsDF.loc[ (self.filename , i ) , input_DF_column_name ]                                        
                                    #index de sentença
                                    sent_index = self.extracted_fragsDF.loc[ (self.filename , i ) , input_DF_column_name + '_index' ]

                                    #coletando a sentença e o sent_index                                        
                                    self.parameters_extracted['column' + str(parameter_N)][i] = {}                                        
                                    self.parameters_extracted['column' + str(parameter_N)][i]['sent'] = sent
                                    self.parameters_extracted['column' + str(parameter_N)][i]['sent_index'] = sent_index                                        
                                                                            
                                    #Mostrando as sentenças a serem processadas para esse PDF
                                    print(f'\nIndex {i} (sent_index {sent_index}):', sent, '\n')
                                    
                                    #só entramos na sequência abaixo para coletar os parâmetros
                                    if self.mode != 'select_sentences':                                            

                                        #essa key definida aqui coleta somente os valores extraídos (serve para comparação/matching)
                                        self.parameters_extracted['column' + str(parameter_N)][i]['extracted_num_str_vals'] = None
                                        
                                        #essa key definida aqui coleta o index_counter, o sent_index e o valores numéricos extraídos                                        
                                        for output_type in ('num_output', 'str_output'):
                                            self.parameters_extracted['column' + str(parameter_N)][i][output_type] = {}
                                        
                                        #checando se o parâmetro irá para a extração numérica
                                        if input_parameter in list_numerical_parameter():
                                            #extraindo os parâmetros numéricos com as unidades físicas
                                            numerical_params_extracted_from_sent = self.extract_numerical_parameters(sent, sent_index, input_parameter, extract_mode = self.numbers_extraction_mode)
                                            #número de parâmetros extraidos
                                            n_num_outputs_extracted = numerical_params_extracted_from_sent['n_num_outputs_extracted']
                                            #unidade SI extraída
                                            SI_units = numerical_params_extracted_from_sent['SI_units']
                                            #caso tenha sido extraído algum output numérico corretamente
                                            print('extracted numerical outputs - n_num_outputs: ', n_num_outputs_extracted, ' ; SI_units: ', SI_units, ' ; sent_len: ', sent_len, '( max_sent_allowed : ', max_token_in_sent, ')')
                                            if n_num_outputs_extracted > 0 and SI_units is not None and sent_len < max_token_in_sent:
                                                #adicionando as SI PUs no nome da coluna que vai ser exportada para a consolidated_DF
                                                self.SI_PUs_dic_to_record[input_parameter] = SI_units
                                                print('> Paramêtros numéricos extraídos para: ', input_parameter + ' (' + SI_units + ')' )
                                                
                                                #coletando os parâmetros extraidos da sentença
                                                self.parameters_extracted['column' + str(parameter_N)][i]['num_output'] = numerical_params_extracted_from_sent['num_output']
                                                self.parameters_extracted['column' + str(parameter_N)][i]['extracted_num_str_vals'] = numerical_params_extracted_from_sent['extracted_num_str_vals']
                                                
                                                print('> ', numerical_params_extracted_from_sent['num_output'] )
                                                #time.sleep(2)
                                            
                                            #caso nenhum output numérico tenha sido exportado
                                            else:
                                                print(f'> Nenhum parâmetro numérico foi extraído para o parameter: {input_parameter}')

                                        #checando se o parâmetro irá para a extração textual                                                
                                        elif input_parameter in list_textual_parameter():
                                            textual_params_extracted_from_sent = self.extract_textual_parameters(sent.lower(), sent_index, input_parameter, ngram = self.ngram_for_textual_search)
                                            #número de parâmetros extraidos
                                            n_str_outputs_extracted = textual_params_extracted_from_sent['n_str_outputs_extracted']
                                            #caso tenha sido extraído algum output numérico
                                            if n_str_outputs_extracted > 0 and sent_len < max_token_in_sent:
                                                print('> Paramêtros textuais extraídos: ', input_parameter)
                                                                                                                                                
                                                #coletando os parâmetros extraidos da sentença                                                                                                        
                                                self.parameters_extracted['column' + str(parameter_N)][i]['str_output'] = textual_params_extracted_from_sent['str_output']
                                                self.parameters_extracted['column' + str(parameter_N)][i]['extracted_num_str_vals'] = textual_params_extracted_from_sent['extracted_num_str_vals']
                                                
                                                print('> ', textual_params_extracted_from_sent['str_output'] )
                                                #time.sleep(2)
                                                
                                            else:
                                                print(f'> Nenhum parâmetro textual foi extraído para o parameter: {input_parameter}')
                                            
                                        #caso os parâmetros introduzidos (input_parameters) não estejam lista para extração nas FUNÇÕES DE EXTRAÇÃO
                                        else:
                                            #listar os parâmetros disponíveis para extração
                                            available_inputs = list_numerical_parameter() + list_textual_parameter()
                                            abort_class = error_incompatible_strings_input('column', input_parameter, available_inputs, class_name = self.class_name)
                                            if abort_class is True:
                                                return                                        
                                                                        
                                while True: 
                                    try:  
                                        if self.mode == 'select_sentences':
                                            print('\nDigite o(s) index(es) das sentenças de interesse (digite valores inteiros)')
                                            print('Outros comandos: "+" para ignorar esse PDF e ir para o próximo; "open" para abrir o PDF; e "exit" para sair.')
                                            self.param_val = str(input('Index: '))
                                        
                                        elif self.mode == 'collect_parameters_manual':
                                            print('\nConfirme o(s) valor(es) extraidos ( digite * para confirmar )')
                                            print('Outros comandos: "+" para ignorar esse PDF e ir para o próximo; "open" para abrir o PDF; e "exit" para sair.')
                                            self.param_val = str(input('Confirma: '))
                                            
                                        elif self.mode == 'collect_parameters_automatic':
                                            self.param_val = '*'
                                        
                                        #processando o input                                        
                                        if self.param_val.lower() == 'open':
                                            wb.open_new(self.diretorio + '/DB/' + self.filename + '.pdf')
                                            continue
    
                                        elif self.param_val.lower() == '*':
                                            break
                                        
                                        elif self.param_val.lower() == '+':
                                            break
                                        
                                        elif self.param_val.lower() == 'exit':
                                            print('> Abortando função: DataFrames.get_data')
                                            return                                    
                                        
                                        else:
                                            #caso o modo seja para coletar as sentenças para treinamento ML                                                                                
                                            if self.mode == 'select_sentences':
                                                try:
                                                    selected_sent_key = int(self.param_val)
                                                    #verificando se esse parametro inserido é relativo ao valor de key das sentenças coletadas
                                                    self.parameters_extracted['column' + str(parameter_N)][selected_sent_key]
                                                    self.parameters_extracted['column' + str(parameter_N)]['selected_sent_index'] = selected_sent_key
                                                    break
                                                except (TypeError, KeyError):
                                                    print('Erro! Inserir um index válido para coletar a sentença.')
                                                    continue                                                                                            
                                
                                    except ValueError:
                                        print('--------------------------------------------')
                                        print('Erro!')
                                        print('Inserir valor válido')
                                        break                                                                                        
                        
                        #se o último input introduzido não foi "+" (que é usado para passar para o proximo PDF)   
                        if self.param_val != '+':
                                            
                            #caso seja modo de coleta de dados
                            convert_to_DF = True
                            if self.mode != 'select_sentences':                                
                                
                                #print(self.parameters_extracted)
                                
                                #coletando somente os keys numéricos do dicionário (inteiros) respectivos à posição da coluna
                                self.columns_numeric_keys = [int( key[ len('column') : ] ) for key in self.parameters_extracted.keys() if ( key[ : len('column') ] == 'column')]

                                #varrendo as colunas
                                for N in self.columns_numeric_keys:
                                    
                                    #concatenando os valores extraídos
                                    #OBS: o formato do dicionário nesse processo passa de:
                                    #       self.parameters_extracted[column_number][sentence_number][output_type][param]
                                    #   para:
                                    #       self.parameters_extracted[column_number][output_type][param] (é uma lista)                                    
                                    for output_type in ('num_output', 'str_output'):
                                        #definindo uma lista para colocar todos os parâmetros coletados
                                        #serve para impedir duplicatas quando o modo self.match_all_sent_findings está ligado
                                        all_values_collected_dic = {}
                                        #definindo uma lista para ser usada no modo de clusterização dos resultados numéricos
                                        num_results_cluster_list = []
                                        #definindo o key para o concatenado
                                        self.parameters_extracted['column' + str(N)][output_type] = {}
                                        #coletando o primeiro valor dos outputs extraidos
                                        first_output = self.parameters_extracted['column' + str(N)][0][output_type]
                                        #checador caso tudo seja igual
                                        all_same = True
                                        #olhando somente os valores extraidos das sentenças (sem o sent_index e o index_counter)
                                        vals_extracted_line_0 = self.parameters_extracted['column' + str(N)][0]['extracted_num_str_vals']
                                        #varrendo todas as sentenças que tiveram dados extraidos
                                        index_counter = 0
                                        #varrendo os indexes de sentença
                                        for i in [ i for i in self.parameters_extracted['column' + str(N)].keys() if type(i) == int ]:
                                            vals_extracted_next_line = self.parameters_extracted['column' + str(N)][i]['extracted_num_str_vals']
                                            #checando se os valores extraídos para diferentes linhas batem
                                            if self.match_all_sent_findings is True:
                                                if vals_extracted_next_line == vals_extracted_line_0:
                                                    continue
                                                else:
                                                    all_same = False
                                                    break
                                            else:
                                                #varrendo os parâmetros de cada output_type
                                                for param in self.parameters_extracted['column' + str(N)][i][output_type].keys():
                                                    #criando uma lista para coletar todos os valores extraidos para cada parâmetro
                                                    all_values_collected_dic[param] = []
                                                    #caso os resultados numéricos sejam clusterizados (só entra o menor e o maior valor encontrado no artigo)
                                                    if (output_type == 'num_output') and (self.get_avg_num_results is True):
                                                        index_counter = 0
                                                        #varrendo os outputs no formato: [ counter , sent_index , val ]
                                                        for res in self.parameters_extracted['column' + str(N)][i][output_type][param]:
                                                            sent_index = res[1]
                                                            num_results_cluster_list.append( round(float( res[2] ), 10) )
                                                            min_val = round(min(num_results_cluster_list), 10)
                                                            max_val = round(max(num_results_cluster_list), 10)
                                                            self.parameters_extracted['column' + str(N)][output_type][param] = [[index_counter , 
                                                                                                                                sent_index , 
                                                                                                                                str(min_val) + ' ' + str(max_val)]]
                                                        #confirmando que foram extraidos parâmetros dessa coluna
                                                        self.parameters_extracted['column' + str(N)]['extracted_paramaters_from_column'] = True
                                                        
                                                    #caso todos os resultados entrem
                                                    else:
                                                        try:
                                                            self.parameters_extracted['column' + str(N)][output_type][param]
                                                        except KeyError:
                                                            index_counter = 0
                                                            self.parameters_extracted['column' + str(N)][output_type][param] = []
                                                        #varrendo os outputs no formato: [ counter , sent_index , val ]
                                                        for res in self.parameters_extracted['column' + str(N)][i][output_type][param]:
                                                            sent_index = res[1]
                                                            val = res[2]
                                                            if val not in all_values_collected_dic[param]:
                                                                self.parameters_extracted['column' + str(N)][output_type][param].append([ index_counter , sent_index , val])
                                                                all_values_collected_dic[param].append(val)
                                                                index_counter += 1
                                                        #confirmando que foram extraidos parâmetros dessa coluna
                                                        self.parameters_extracted['column' + str(N)]['extracted_paramaters_from_column'] = True
                                        
                                        #modo que seleciona os parâmetros somente quanto todos os valores extraidos de todas as sentenças sejam iguais
                                        if self.match_all_sent_findings is True:
                                            #print(self.parameters_extracted)
                                            #concatenando os parâmetros extraidos em uma key
                                            if all_same is True:
                                                self.parameters_extracted['column' + str(N)][output_type] = first_output
                                                #confirmando que foram extraidos parâmetros dessa coluna
                                                self.parameters_extracted['column' + str(N)]['extracted_paramaters_from_column'] = True
                                            else:
                                                convert_to_DF = False                                                
                                        
                                    #apagando as keys de cada sentença
                                    for i in [ i for i in self.parameters_extracted['column' + str(N)].keys() if type(i) == int ]:
                                        del(self.parameters_extracted['column' + str(N)][i])

                            if convert_to_DF is True:
                                #if len(self.parameters_extracted['column0']['str_output']) > 0:
                                #    print(self.parameters_extracted)
                                #    time.sleep(2)
                                #gerando a DF com os dados colletados
                                #print(self.parameters_extracted)
                                #time.sleep(20)
                                self.convert_parameters_extracted_to_DF()
                            else:
                                print('Erro! Os parâmetros extraídos de multiplas linhas não são idênticos.')
                                                                                
                    else:
                        print(f'O PDF {self.filename} já foi processado (checar no arquivo ~/Outputs/DataFrames/{self.output_DF_name}.csv)')
                        continue
                    
                #consolidando o report na DF caso seja o ultimo arquivo de procura        
                if self.filename == self.extracted_fragsDF.index.levels[0][-1]:
                    self.generate_search_report()
                    
            else:
                print('Erro! O número de parâmetros introduzidos é diferente do número de termos coletados (colunas do extracted_DF)')
                print('Número de parâmetros introduzido: ',  len(self.input_parameters_list) )
                print('Número de colunas da extracted_DF: ',  len(self.extracted_fragsDF.columns) / 2)
                print('> Abortando a classe: DataFrames')
                return

        else:
            print('Erro! Não foi encontrado um DF de fragamentos de artigos.')
            print('> Abortando a classe: DataFrames')
            return


    def convert_parameters_extracted_to_DF(self):
                
        import pandas as pd
        import os
        import time

        if not os.path.exists(self.diretorio + '/Outputs/DataFrames'):
            os.makedirs(self.diretorio + '/Outputs/DataFrames')
        
        #caso o modo seja de selecionar sentenças
        if self.mode == 'select_sentences':
            #gerando e salvando as DF com as sentenças
            self.analyze_output_results()
            if self.output_results_checked is True:
                self.consolidate_DF()
        
        else:                        
            
            #Caso os ouputs de todos os parâmetros (colunas) sejam coletados
            #if False not in [ self.parameters_extracted[key]['extracted_paramaters_from_column'] for key in self.parameters_extracted.keys() if key[ : len('column')] == 'column' ]:                
                    
            #caso a output_DF tenha que ser criada
            if self.create_output_DF is True:
                                                        
                self.output_DF = pd.DataFrame(columns=['Filename', 'Counter'], dtype=object)
                self.output_DF.set_index(['Filename', 'Counter'], inplace=True)
                            
            print(f'Procurando... {self.diretorio}/Outputs/DataFrames/{self.consolidated_DataFrame_name}.csv')
            if not os.path.exists(self.diretorio + f'/Outputs/DataFrames/{self.consolidated_DataFrame_name}.csv'):
                print(f'Criando a DF consolidada... (~/Outputs/DataFrames/{self.consolidated_DataFrame_name}.csv)')                
                self.consolidated_DF = pd.DataFrame(index=[[],[]], dtype=object)
                self.consolidated_DF.index.names = ['Filename', 'Counter']
                self.consolidated_DF.to_csv(self.diretorio + f'/Outputs/DataFrames/{self.consolidated_DataFrame_name}.csv')
            
            #carregando a DF consolidada caso o modo seja de coleção de parâmetros
            else:
                print(f'Abrindo a consolidated DF: {self.diretorio}/Outputs/DataFrames/{self.consolidated_DataFrame_name}.csv')
                self.consolidated_DF = pd.read_csv(self.diretorio + f'/Outputs/DataFrames/{self.consolidated_DataFrame_name}.csv', index_col=[0,1], dtype=object)
                            
            #criando a coluna do Sample_counter no DF consolidado
            if 'Sample_counter' not in self.consolidated_DF.columns:
                self.consolidated_DF['Sample_counter'] = ''                                
            
            #gerando e salvando as DF com os parâmetros
            self.analyze_output_results()
            if self.output_results_checked is True:
                self.consolidate_DF()
            
            #else:
            #    #print([ self.parameters_extracted[key]['extracted_paramaters_from_column'] for key in self.parameters_extracted.keys() if key[ : len('column')] == 'column' ])
            #    print('Erro! Não foram coletados dados para todos os parâmetros inseridos.')
            #    print('Passando para o próximo PDF')
            #    #time.sleep(3)
            #    return


    def analyze_output_results(self):

        #determinando todos os parâmetros que foram extraidos de todas as colunas
        self.extracted_parameters_list = []
        for N in self.columns_numeric_keys:
            #parametros numéricos
            if len(self.parameters_extracted['column' + str(N)]['num_output']) > 0:
                num_parameter_extracted_list = self.parameters_extracted['column' + str(N)]['num_output'].keys()
                for parameter in num_parameter_extracted_list:
                    if parameter not in self.extracted_parameters_list:
                        self.extracted_parameters_list.append(parameter)
            
            #parametros textuais
            if len(self.parameters_extracted['column' + str(N)]['str_output']) > 0:
                str_parameter_extracted_list = self.parameters_extracted['column' + str(N)]['str_output'].keys()
                for parameter in str_parameter_extracted_list:
                    if parameter not in self.extracted_parameters_list:
                        self.extracted_parameters_list.append(parameter)

        if self.mode == 'select_sentences':        

            #testa se indexes foram coletados para todas as colunas (parâmetros)               
            if None not in [self.parameters_extracted['column' + str(j)]['selected_sent_index'] for j in self.columns_numeric_keys]:                               
                #uma linha pois só uma sentença é coletada por PDF
                self.output_results_checked = True
            else:
                print('Erro de extração! Nenhum parâmetro foi extraído (max_output_number = 0).')
                self.output_results_checked = False
            
        #modo de coleta de dados
        else:
            #caso tenha tido pelo menos um parametro extraido
            if len( self.extracted_parameters_list ) > 0:
                self.output_results_checked = True
                #fazendo o match para os parâmetros        
                self.match_parameters_extracted()
            else:
                print('Erro de extração! Nenhum parâmetro foi extraído (max_output_number = 0).')
                self.output_results_checked = False


    def match_parameters_extracted(self):                
        
        import time
        
        #setando as condições para fazer o fill dos outputs
        #o dicionário output number carrega o número de outputs por coluna (feature ou parâmetro) a ser colocado na DF consolidada
        self.output_number = {}

        #varrendo os parâmetros
        for N in self.columns_numeric_keys:                    
            #somente um output é extraído por coluna

            #varrendo as colunas (parâmetros)
            for output_parameter in self.extracted_parameters_list:
                #ou é numerico
                try:
                    #caso essa coluna tenha tido parâmetro numérico extraído
                    self.output_number[output_parameter] = len(self.parameters_extracted['column' + str(N)]['num_output'][output_parameter] )
                    #print('column' + str(N), ' ; ', output_parameter, ' : ' , len(self.parameters_extracted['column' + str(N)]['num_output'][output_parameter]) )
                except KeyError:
                    continue                        
                
            #varrendo as colunas (parâmetros)
            for output_parameter in self.extracted_parameters_list:    
                #ou é textual
                try:                
                    #caso essa coluna tenha tido parâmetro numérico extraído
                    self.output_number[output_parameter] = len(self.parameters_extracted['column' + str(N)]['str_output'][output_parameter] )
                    #print('column' + str(N), ' ; ', output_parameter, ' : ' , len(self.parameters_extracted['column' + str(N)]['num_output'][output_parameter]) )
                except KeyError:
                    continue

        self.len_set_output_number = len(set(self.output_number.values()))
        self.min_output_number = min(list(set(self.output_number.values())))
        self.max_output_number = max(list(set(self.output_number.values())))
        
        #checando os resultados
        print('* code analysis begin *')
        print('output_number: ', self.output_number)
        print('len_set_output_number = ', self.len_set_output_number)
        print('min_output_number = ', self.min_output_number)
        print('max_output_number = ', self.max_output_number)
        #time.sleep(1)


    def check_output_conditions(self):
        
        import time
        
        checked = True

        #procurando se o index de PDF já tem sample number
        file_tag = self.parameters_extracted['filename']
        try:        
            #caso seja possível converter para inteiro
            self.current_sample_number = int(float(self.consolidated_DF.loc[ ( file_tag , 0 ), 'Sample_counter' ]))             
        #exceção caso o index não exista na DF ou caso o valor de sample_counter seja None
        except KeyError:
            self.consolidated_DF.loc[ ( file_tag , 0 ), 'Sample_counter' ] = 0
            self.current_sample_number = 0
            
        #o número de amostra atualizado é definido ou pelo número que já está na DF consolidada ou pelo número de outputs que está sendo atualizado
        self.updated_sample_number = max(self.max_output_number, self.current_sample_number)
        
        print('min_output_number = ', self.min_output_number)
        print('max_output_number = ', self.max_output_number)
        print('current_sample_number = ', self.current_sample_number)
        print('updated_sample_number = ', self.updated_sample_number)
        
        #condições para fazer a consolidação do FULL DF
        
        cond_output_number =  False
        #quando o número de outputs seja tudo igual (ex: temperature_pyrolysis = 2 ; raw_materials = 2)
        if (self.len_set_output_number == 1) and (self.min_output_number > 0):
            cond_output_number = True
        #quando o número de outputs não é igual (ex: temperature_pyrolysis = 3 ; time_pyrolysis = 1 ; raw_materials = 1)
        #o número de itens do set deve ser 2 
        #o maior valor de output_number deve ser maior que 1
        #o menor valor de output_number deve ser 1 ou 0 
        elif (self.len_set_output_number == 2) and ((self.max_output_number >= 1)) and (self.min_output_number in (0, 1)) :
            cond_output_number = True

        #para atualização no número de amostras
        cond_match_sample_number = False
        #só se atualiza quando o número de amostra atualizao for igual àquele que está na DF consolidada
        if self.hold_sample_number is True:
            if self.updated_sample_number == self.current_sample_number:
                cond_match_sample_number = True        
        else:        
            #quando o número de amostra atualizao for igual àquele que está na DF consolidada
            if self.updated_sample_number == self.current_sample_number:
                cond_match_sample_number = True
            #ou quando um valor seja 0 ou 1 e outro maior que 1
            elif self.updated_sample_number in (0, 1) and  self.current_sample_number >= 1:
                cond_match_sample_number = True
            elif self.updated_sample_number >= 1 and  self.current_sample_number in (0, 1):
                cond_match_sample_number = True

        #quando o atributo de hold_samples estive ligado, o algoritmo só adiciona parâmetros nas amostras que já tem sample_counter > 0        
        cond_hold_samples = True                    
        if self.hold_samples is True:
            #só atualizará se o já houver um número de amostra na DF consolidada
            if (self.current_sample_number == 0):
                cond_hold_samples = False
        else:
            pass                
                
        #checando os resultados
        print('cond_output_number: ', cond_output_number)
        print('cond_match_sample_number: ', cond_match_sample_number)
        print('cond_hold_samples: ', cond_hold_samples)
        #time.sleep(2)

        #se alguma das condições falhou
        if False in (cond_output_number, cond_match_sample_number, cond_hold_samples):
            checked = False
                            
        return checked


    def consolidate_DF(self):

        import time
        import pandas as pd
        from FUNCTIONS import save_dic_to_json        
        
        file_tag = self.parameters_extracted['filename']            
        if self.check_output_conditions() is True:
            
            #checando os outputs que serão preenchidos de forma repetida
            self.check_outputs_to_fill_in_consolidated_DF()            

            #varrendo os parâmetros que serão extraidos
            for output_parameter in self.extracted_parameters_list:
                
                #caso já exista alguma entrada na consolidated DF para esse file_tag e para esse output_parameter
                try:
                    self.consolidated_DF.loc[ ( file_tag , ) , output_parameter ]
                    if str(self.consolidated_DF.loc[ ( file_tag , 0 ) , output_parameter ]) == 'nan':
                        output_already_in_consolidated_df = False
                    else:
                        output_already_in_consolidated_df = True
                    
                #caso não exista                    
                except KeyError:
                    output_already_in_consolidated_df = False
                    
                #varrendo as colunas (parâmetros)
                for N in self.columns_numeric_keys:
                    #varrendo os dois parâmetros
                    for output_type in ('num_output','str_output'):                    
                        #a instância possui a quantidade de outputs a ser preenchida na consolidated DF            
                        for counter in range(self.updated_sample_number):
                            try:
                                self.parameters_extracted['column' + str(N)][output_type][output_parameter]
                            except KeyError:
                                continue
                            
                            if self.parameters_extracted['column' + str(N)][output_type][output_parameter] is None:
                                output_val_index = ''
                                output_val = ''
                            
                            elif self.filloutputs[output_parameter] is True and self.parameters_extracted['column' + str(N)][output_type][output_parameter] is not None:
                                output_val_index = self.parameters_extracted['column' + str(N)][output_type][output_parameter][0][1]
                                output_val = self.parameters_extracted['column' + str(N)][output_type][output_parameter][0][2]
                            
                            elif self.filloutputs[output_parameter] is False and self.parameters_extracted['column' + str(N)][output_type][output_parameter] is not None:
                                #caso o parâmetro a ser trabalhado esteja no dicionário self.parameters_extracted
                                output_val_index = self.parameters_extracted['column' + str(N)][output_type][output_parameter][counter][1]
                                output_val = self.parameters_extracted['column' + str(N)][output_type][output_parameter][counter][2]                            
                        
                            #salvando no output_DF
                            self.output_DF.loc[ ( file_tag , counter ) , output_parameter ] = output_val
                            self.output_DF.loc[ ( file_tag , counter ) , output_parameter + '_index' ] = output_val_index
            
                            if self.mode != 'select_sentences':
                                #salvando na consolidated
                                self.consolidated_DF.loc[ ( file_tag , counter ) , output_parameter ] = output_val
                                self.consolidated_DF.loc[ ( file_tag , counter ) , output_parameter + '_index' ] = output_val_index                          
                                                                
                            #salvando o número atualizado de amostras por PDF na DF consolidada
                            self.consolidated_DF.loc[ ( file_tag , counter ) , 'Sample_counter' ] = self.updated_sample_number                        
            
            
                #salvando a consolidated DF caso não haja entrada para esse file_tag e output_parameter
                if output_already_in_consolidated_df is False and self.mode != 'select_sentences':
                    
                    #varrendo os parâmetros que que já foram extraidos para a consolidated_DF e que não estão na self.extracted_parameters_list
                    for output_parameter in self.parameters_in_consolidated_DF_list:
                        if self.filloutputs[output_parameter] == True:                    
                            output_val = self.consolidated_DF.loc[ ( file_tag , 0 ) , output_parameter ]
                            output_val_index = self.consolidated_DF.loc[ ( file_tag , 0 ) , output_parameter + '_index' ]
                            for counter in range(self.updated_sample_number):
                                self.consolidated_DF.loc[ ( file_tag , counter ) , output_parameter ] = output_val
                                self.consolidated_DF.loc[ ( file_tag , counter ) , output_parameter + '_index' ] = output_val_index
                    
                    #salvando a DF consolidada
                    save_dic_to_json(self.diretorio + '/Outputs/DataFrames/SI_PUs.json', self.SI_PUs_dic_to_record)
                    self.consolidated_DF.sort_index(level=[0,1], inplace=True)
                    self.consolidated_DF.to_csv(self.diretorio + f'/Outputs/DataFrames/{self.consolidated_DataFrame_name}.csv')
                    print(f'\nAtualizando DataFrame consolidado para {file_tag}...')

            #salvando a output_DF            
            self.output_DF.to_csv(self.diretorio + f'/Outputs/DataFrames/{self.output_DF_name}_mode_{self.numbers_extraction_mode}_match_{self.match_all_sent_findings}.csv')
            print(f'\nSalvando output_DF para {file_tag}...')

            #contadores para exportar nas DFs de report
            self.search_report_dic['export']['last_PDF_processed'] = self.parameters_extracted['filename']
            self.search_report_dic['export']['total_finds'] += self.max_output_number
            self.search_report_dic['export']['pdf_finds'] += 1
            
            #salvando o DF report
            save_dic_to_json(self.diretorio + f'/Outputs/LOG/counter_SE_report_{self.input_DF_name}_extracted_mode_{self.search_mode}.json', self.search_report_dic)
                    
        else:
            print('\nHá incompatibilidade entre os outputs (função: check_output_conditions).')
            print(f'DataFrame para o {file_tag} não foi consolidada.')
            return

    
    
    def check_outputs_to_fill_in_consolidated_DF(self):
        
        import time

        #resgatando os parâmetros que já estão na consolidated_DF
        #usa-se o slica [ 1 : ] na consolidated_DF porque a primeira coluna é de sample_counter
        self.parameters_in_consolidated_DF_list = [parameter for parameter in self.consolidated_DF.columns[ 1: ] if ( (parameter[ -len('_index') : ] != '_index') and (parameter not in self.extracted_parameters_list) ) ]

        full_parameters_list = self.parameters_in_consolidated_DF_list + self.extracted_parameters_list
        print('full_parameters_list:', full_parameters_list)

        #checando se haverá preenchimento automático de outputs
        self.filloutputs = {}        
        
        #varrendo as colunas (parâmetros) tanto para a DF consolidada quanto para a output DF    
        for output_parameter in full_parameters_list:
            
            fillout = False
            
            try:
                #máximo número de outputs numéricos coletados para cada parametro (coluna)
                #print(f'output_number[{output_parameter}]: ', self.output_number[output_parameter], ' ; updated_sample_number: ' , self.updated_sample_number)
                if self.output_number[output_parameter] < self.updated_sample_number:
                    fillout = True
            
            #caso o parametro já tenha sido extraido anteriormente e já esteja na consolidated_DF
            except KeyError:
                #caso o Sample_Counter seja 1 o Updated_sample_counter seja maior
                if (self.current_sample_number == 1) and self.current_sample_number < self.updated_sample_number:
                    fillout = True
            
            #determinando se terá o fillout ou não
            if fillout is True:
                self.filloutputs[output_parameter] = True
            else:
                self.filloutputs[output_parameter] = False

        print('filloutputs: ', self.filloutputs)
        print('* code analysis end *')
        #time.sleep(1)
    

    def extract_textual_parameters(self, sent, sent_index, parameter, ngram = 1):
        
        from functions_TOKENS import get_nGrams_list
        import regex as re
        #import time
        
        extracted_dic = {}
        
        str_list = []
        term_list_modified, prob_term_list = get_nGrams_list(parameter, ngrams = self.ngram_for_textual_search, min_ngram_appearence = self.min_ngram_appearence, diretorio=self.diretorio)
        #varrendo os termos
        for term in term_list_modified:
            if re.search(term, sent):
                str_list.append(term)
    
        str_list.sort()
        extracted_dic['extracted_num_str_vals'] = str_list
    
        #organizando os dados que serão exportados
        extracted_dic['str_output'] = {}
        extracted_dic['str_output'][parameter] = []
        counter = 0
        for str_val in str_list:
            extracted_dic['str_output'][parameter].append([counter, sent_index, str_val])            
            counter += 1
        
        #exportando o número de resultados
        extracted_dic['n_str_outputs_extracted'] = len(str_list)
        
        #print(extracted_dic)
        #time.sleep(10)
        return extracted_dic


    def extract_numerical_parameters(self, text, text_index, parameter, extract_mode = 'first'):
        
        import time
        from functions_PARAMETERS import regex_patt_from_parameter
        from functions_PARAMETERS import get_physical_units_converted_to_SI
        import regex as re

        extracted_dic = {}
        extracted_dic['n_num_outputs_extracted'] = 0
        extracted_dic['SI_units'] = None
        extracted_dic['num_output'] = None
                
        #obtendo os padrões regex para encontrar e para não encontrar
        parameters_patterns = regex_patt_from_parameter(parameter)

        #definindo a lista com as PUs encontradas na sentença
        PUs_found_list = []            
        #definindo a lista com os valores numéricos encontrados na sentença
        num_list_extracted_raw = []            
        #definindo os parâmetros que podem ser extraídos da sentença
        parameters_found = []


        #função de extração de números e PUs
        def extract_PU_and_numbers_from_text(match_text):
            num_list_extracted_raw = []
            PUs_list = []
            #texto encontrado
            #print('numerical 1.', match_text)
            #extraindo os números
            #eliminando os números associados às PUs inversas (ex: min–1)
            numbers_found_in_text_p1 = re.sub( r'(?<=[a-z]–?)[0-9]+', '', match_text )
            #print('numerical 2.', numbers_found_in_text_p1)
            #substituindo o sinal de menos (–) por um caracter que o python reconheça como negativo (-)
            numbers_found_in_text_p2 = re.sub( r'\–(?=[0-9])', '-', numbers_found_in_text_p1 )
            #print('numerical 3.', numbers_found_in_text_p2)
            #substituindo o "to" por "&"
            numbers_found_in_text_p3 = re.sub( r'\s+to\s+', '&', numbers_found_in_text_p2 )
            #print('numerical 4.', numbers_found_in_text_p3)
            #encontrando somente os números
            numbers_find_list = re.findall( r'-?[0-9]+\.?[0-9]*&-?[0-9]+\.?[0-9]*(?!\s*[0-9]*\s*\(\s*[0-9]+)|-?[0-9]+\.?[0-9]*(?!\s*[0-9]*\s*\(\s*[0-9]+)' , numbers_found_in_text_p3 )
            #print('numerical 5.', numbers_find_list)
            
            #seperando as unidades físicas dos números (ex: 2h -> 2 h)
            num_list_extracted_raw = []
            for result_found in numbers_find_list:
                result_found_filtered1 = re.sub(r'[A-Za-z]', '', result_found)
                result_found_filtered2 = re.sub(r'\s', '', result_found_filtered1)
                num_list_extracted_raw.append(result_found_filtered2)
            
            num_list_extracted_raw_filtered = []
            for raw_num in num_list_extracted_raw:
                try:
                    if round(float(raw_num), 1) != 0.0:
                        num_list_extracted_raw_filtered.append(raw_num)
                    else:
                        continue
                except ValueError:
                    num_list_extracted_raw_filtered.append(raw_num)
            
            #print('Números: ', num_list_extracted_raw_filtered)
        
            #procurando a unidade física encontrada para o parâmetro introduzido
            #eliminando todos os números com exceção daqueles associados às PUs inversas (ex: 250)
            PUs_found_in_text_p1 = re.sub( r'(?<![a-z]–?)[0-9]+', '', match_text )
            #print('numerical 6.', PUs_found_in_text_p1)
            #eliminando todos os sinais de menos (–) com exceção daqueles associados às PUs inversas (ex: –)
            PUs_found_in_text_p2 = re.sub( r'(?<![a-z])–', '', PUs_found_in_text_p1 )
            #print('numerical 7.', PUs_found_in_text_p2)
            #eliminando os separadores de unidades numéricas "," ".", "&", "to" ou "and"
            PUs_found_in_text_p3 = re.sub( r'(or|to|and|\,|\.|;|&|\(|\))', '', PUs_found_in_text_p2 )
            #print('numerical 8.', PUs_found_in_text_p3)
            for pu in PUs_found_in_text_p3.split():
                if pu not in PUs_list:
                    PUs_list.append(pu)
            #print('PUs: ', PUs_list)
            #time.sleep(2)
            
            return PUs_list , num_list_extracted_raw_filtered


        #função de extração de parâmetros
        def extract_parameters_from_text(match_text):
            #print('match_parameters_to_find: ', match_parameters_to_find)
            #eliminando os separadores de unidades numéricas "," ".", ou "and"
            parameter_found_in_text_p1 = re.sub( r'(\s|or|and|\,|\.|;)', '', match_text )
            #print('parameter 1. ', parameter_found_in_text_p1)
            parameters_ = re.findall(parameters_patterns['parameter_to_find_in_sent'], parameter_found_in_text_p1)
            #print('Parameters: ', parameters_)
            
            return parameters_

        
        #função de extração de PUs
        def extract_PU_to_SI(PU_list):
            #caso o parâmetro seja adimensional
            if len(PU_list) == 0:
                PU_units_found = ['adimensional']
            #caso o parâmetro tenha unidade
            else:
                PU_units_found =[]
                for unit in PU_list:
                    if unit not in PU_units_found:
                        PU_units_found.append(unit)
                
            print('Raw PUs encontradas: ', PU_units_found)
            #time.sleep(1)
            
            #adquirindo o factor numérico de conversão para colocar as unidades no SI
            factor_to_multiply , factor_to_add , SI_units = get_physical_units_converted_to_SI(PU_units_found)
        
            return factor_to_multiply , factor_to_add , SI_units

        #listas com os indices da sentença onde foram extraidos os números e os parâmetros
        parameters_cut_index_list = []
        numbers_cut_index_list = []
        #caso só seja coletado a primeira aparição do padrão regex
        if extract_mode.lower() == 'first':            

            #condições de match
            match_numbers_to_find = None
            match_numbers_further_in_sent = None
            match_numbers_not_to_find = None
            match_parameters_to_find = None
            
            #caso seja feita a procura por parâmetros no texto
            if parameters_patterns['parameter_to_find_in_sent'] is not None:
                parameters_pattern = r'(([\s\,]*and\s|[\,\s]+|\s+)({text_to_find}))+[\s\.\,;:\)\]]'.format(text_to_find = parameters_patterns['parameter_to_find_in_sent'])
                match_parameters_to_find = re.search(parameters_pattern , text)
                if match_parameters_to_find:
                    match_text = match_parameters_to_find.captures()[0]
                    parameters_found = extract_parameters_from_text(match_text)
                print('Parâmetros extraídos: ', parameters_found)

            #encontrando os parâmetros com as unidades físicas que são desejadas
            match_numbers_to_find = re.search( parameters_patterns['PU_to_find_regex'] , text)
            
            #caso o match de parâmetros tenha sido encontrado
            if match_numbers_to_find:
                #pegando os índices
                matches_start, matches_end = match_numbers_to_find.span()
    
                #encontrando os parâmetros com as unidades físicas que não são desejadas
                #caso seja None, é porque a unidade física de interesse é combinada (ex: concentração - mol L-1, taxa de aumento de temperatura C s-1)
                if parameters_patterns['PU_not_to_find_regex'] is None:
                    pass
                #caso não seja None, é porque a unidade física de interesse é single (ex: temperatura - C, energia - J)
                else:
                    #soma-se 7 no matches_end para checar o restante do texto (ex: C min-1)
                    match_numbers_not_to_find = re.search( parameters_patterns['PU_not_to_find_regex'] , text[ matches_start : matches_end + 7] )
                    if match_numbers_not_to_find:
                        print('match_numbers_not_to_find: ', text[ : matches_end + 7], ' ; result: ', match_numbers_not_to_find)
                        time.sleep(2)
                    
                #checando se há mais uma parte que bate com o padrão na mesma sentença
                match_numbers_further_in_sent = re.search( parameters_patterns['PU_to_find_regex'] , text[ matches_end : ])
                if match_numbers_further_in_sent:
                    print('match_numbers_further_in_sent: ', match_numbers_further_in_sent)
                    #time.sleep(2)
                
            #caso todas as condições tenham sido preenchidas
            if ( match_numbers_to_find is not None ) and ( match_numbers_further_in_sent is None ) and (match_numbers_not_to_find is None):
                #texto encontrado
                found_in_text = text[ matches_start : matches_end ]                
                PUs_found_list , num_list_extracted_raw = extract_PU_and_numbers_from_text(found_in_text)


        #caso sejam coletados todos os findings na sentença
        elif extract_mode.lower() == 'all':            

            cumulative_cut_for_parameters = 0
            cumulative_cut_for_numbers = 0
            PUs_found_list_temp1 = []
            PUs_found_list_temp2 = []
            while True:
                cut_text = text[ cumulative_cut_for_parameters : ]                
                #caso seja feita a procura por parâmetros no texto            
                if parameters_patterns['parameter_to_find_in_sent'] is not None:
                    parameters_pattern = r'(([\s\,]*and\s|[\,\s]+|\s+|)({text_to_find}))+[\s\.\,;:\)\]]'.format(text_to_find = parameters_patterns['parameter_to_find_in_sent'])
                    match_parameters_to_find = re.search(parameters_pattern, cut_text)
                    if match_parameters_to_find:
                        cumulative_cut_for_parameters +=  match_parameters_to_find.span()[1]
                        for param in extract_parameters_from_text( match_parameters_to_find.captures()[0] ):
                            if param not in parameters_found:
                                parameters_found.append( param )
                                parameters_cut_index_list.append(cumulative_cut_for_parameters)
                    print('Parâmetros extraídos: ', parameters_found)
                
                cut_text = text[ cumulative_cut_for_numbers : ]                
                #print(parameters_patterns['PU_to_find_regex'])
                match_numbers_to_find = re.search( parameters_patterns['PU_to_find_regex'] , cut_text )
                #print(match_numbers_to_find)
                
                #caso tenha sido encontrado o padrão
                if match_numbers_to_find:
                    matches_start, matches_end = match_numbers_to_find.span()
                    
                    #encontrando os parâmetros com as unidades físicas que não são desejadas
                    #caso seja None, é porque a unidade física de interesse é combinada (ex: concentração - mol L-1, taxa de aumento de temperatura C s-1)
                    if parameters_patterns['PU_not_to_find_regex'] is None:
                        pass
                    #caso não seja None, é porque a unidade física de interesse é single (ex: temperatura - C, energia - J)
                    else:
                        #soma-se 7 no matches_end para checar o restante do texto (ex: C min-1)
                        match_numbers_not_to_find = re.search( parameters_patterns['PU_not_to_find_regex'] , text[ matches_start + cumulative_cut_for_numbers : matches_end + cumulative_cut_for_numbers + 7 ] )
                        #print(parameters_patterns['PU_not_to_find_regex'])
                        if match_numbers_not_to_find:
                            print('match_numbers_not_to_find: ', text[ matches_start + cumulative_cut_for_numbers : matches_end + cumulative_cut_for_numbers + 7 ], ' ; result: ', match_numbers_not_to_find)
                            cumulative_cut_for_numbers +=  match_numbers_to_find.span()[1]
                            #time.sleep(2)
                            continue
                    
                    #print('all 1. match captures: ', match_numbers_to_find)
                    match_text = match_numbers_to_find.captures()[0]
                    #print('all 2. match_text: ', match_text)
                    PUs_found_list_temp1 , num_list_extracted_raw_temp = extract_PU_and_numbers_from_text( match_text )
                    num_list_extracted_raw.extend( num_list_extracted_raw_temp )
                    PUs_found_list_temp2.extend( PUs_found_list_temp1 )
                    #print('all 3. num_list ; PUs_list: ', num_list_extracted_raw , ' ; ', PUs_found_list)
                    cumulative_cut_for_numbers +=  match_numbers_to_find.span()[1]
                    numbers_cut_index_list.append(cumulative_cut_for_numbers)
                    #print('all 4. index:', cumulative_cut_for_numbers)
                    #print('all 5. cut text: ', text[ cumulative_cut_for_numbers : ])
                    #time.sleep(2)
                else:
                    for pu in PUs_found_list_temp2:
                        if pu not in PUs_found_list:
                            PUs_found_list.append(pu)
                    break

        #print('Números extraídos: ', num_list_extracted_raw)
        #print('PUs extraídos: ', PUs_found_list)
        #print('parameters_cut_index_list: ', parameters_cut_index_list)
        #print('numbers_cut_index_list: ', numbers_cut_index_list)
        #time.sleep(2)
        
        #após ter feito a extração no mode "all" ou "first"
        if len(num_list_extracted_raw) > 0:
            
            #convertendo as PUs para SI        
            factor_to_multiply , factor_to_add , SI_units = extract_PU_to_SI(PUs_found_list)
                    
            #caso não tenha sido achada a unidade fisica
            if None in (factor_to_multiply , factor_to_add , SI_units):            
                print('Erro! Os Parâmetros numéricos foram encontrados, mas houve erro no processamento das unidades físicas PUs.')
                return extracted_dic
            
            else:
                
                #quantos dados numéricos foram extraidos
                extracted_dic['n_num_outputs_extracted'] = len(num_list_extracted_raw)
                
                #coletando os dados numéricos
                num_list_extracted_converted = []
                #varrendo os valores numéricos capturados
                for num in num_list_extracted_raw:
                    if '&' in num:
                        numbers_interval = re.findall(r'-?[0-9]+\.?[0-9]*', num)
                        first_number = round( float(numbers_interval[0]) * factor_to_multiply + factor_to_add , 8)
                        last_number = round( float(numbers_interval[1]) * factor_to_multiply + factor_to_add , 8)
                        num_list_extracted_converted.append( str(first_number) )
                        num_list_extracted_converted.append( str(last_number) )
                    else:
                        num_list_extracted_converted.append( str( round( float(num) * factor_to_multiply + factor_to_add, 9) ) )

                #caso haja procura de parâmetro no texto (ex: C / O, C / H)
                if parameters_patterns['parameter_to_find_in_sent']:
                    try:
                        #o número de parâmetros encontrados deve ser igual à quantidade de números extraídos
                        if len(parameters_found) > 0 and ( len(parameters_found) == len(num_list_extracted_converted) or len(num_list_extracted_converted) == 2*len(parameters_found) ):
                            extracted_dic['SI_units'] = SI_units
                            #definindo a lista com os parâmetros diferentes
                            parameters_list = []
                            for param in parameters_found:
                                if param not in parameters_list:
                                    parameters_list.append(param)                                        
                            
                    except UnboundLocalError:
                        pass
    
                else:                                        
                    extracted_dic['SI_units'] = SI_units
                    #definindo a lista com o parâmetro repetido (ex: [ mV , mV , mV ])
                    parameters_list = len(num_list_extracted_converted) * [parameter]
            
            #caso terha sido extraídos os parâmetros corretamente
            if extracted_dic['SI_units']:
                            
                #definindo o dic para coletar os parâmetros numéricos
                extracted_dic['num_output'] = {}
                
                #caso os parâmetros e o valores numéricos estejam organizados da seguinte forma: "param1 and param2 fdsfds number1 and number2 dfdsf number3 and number 4"
                cond1, cond2, cond3, cond4 = False, False, False, False
                if len(parameters_found) > 1:
                    cond1  = True    
                    cond4 = len(num_list_extracted_converted) == 2*len(parameters_found)
                if len(parameters_cut_index_list) > 0:
                    cond3 = True                    
                    cond4 = max(parameters_cut_index_list) < min(numbers_cut_index_list)
                    
                if False not in (cond1, cond2, cond3, cond4):
                    doubled_parameters_list = parameters_list + parameters_list
                    #varrendo os resultados extraídos
                    for i in range( len(doubled_parameters_list) ):                    
                        index_counter = 0
                        #caso seja a primeira instância do key
                        try:
                            extracted_dic['num_output'][doubled_parameters_list[i]]
                            index_counter += 1
                        except KeyError:
                            extracted_dic['num_output'][doubled_parameters_list[i]] = []

                        extracted_dic['num_output'][doubled_parameters_list[i]].append( [ index_counter , text_index, num_list_extracted_converted[i]] )
                                    
                else:                    
                    #caso o número de valores numéricos seja o dobro do número de parâmetros
                    double_num_outputs = False
                    if len(num_list_extracted_converted) == 2*len(parameters_found) and (parameters_patterns['parameter_to_find_in_sent'] is not None) :
                        double_num_outputs = True
                    
                    #varrendo os resultados extraídos
                    index_counter = 0
                    for i in range( len(parameters_list) ):
                        #caso seja a primeira instância do key
                        try:
                            extracted_dic['num_output'][parameters_list[i]]
                            index_counter += 1
                        except KeyError:
                            index_counter = 0
                            extracted_dic['num_output'][parameters_list[i]] = []

                        if double_num_outputs is True:
                            extracted_dic['num_output'][parameters_list[i]].append( [ index_counter , text_index, num_list_extracted_converted[i]] )
                            extracted_dic['num_output'][parameters_list[i]].append( [ index_counter + 1 , text_index, num_list_extracted_converted[i + 1]] )
                        else:
                            extracted_dic['num_output'][parameters_list[i]].append( [ index_counter , text_index, num_list_extracted_converted[i]] )
                                        
    
                #fazendo o sorting da lista (se faz o sort pois esses valores servirão para comparação)
                num_list_extracted_converted.sort()
                #exportando só os valores numéricos
                extracted_dic['extracted_num_str_vals'] = num_list_extracted_converted
    
    
        print('extracted_dic from sent: ', extracted_dic)
        #if double_num_outputs is True:
        #time.sleep(10)
        return extracted_dic
    

    
    def generate_search_report(self):
        
        import os
        import pandas as pd
        
        #abrindo o SE report            
        if os.path.exists(self.diretorio + '/Settings/SE_inputs.csv'):
            search_report_DF = pd.read_csv(self.diretorio + '/Settings/SE_inputs.csv', index_col = 0)

            search_report_DF.loc[self.input_DF_name, 'Total Extracted'] = self.search_report_dic['export']['total_finds']
            search_report_DF.loc[self.input_DF_name, 'PDF Extracted'] = self.search_report_dic['export']['pdf_finds']
            search_report_DF.loc[self.input_DF_name , 'export_status' ] = 'finished'

            search_report_DF.sort_index(inplace=True)
            search_report_DF.to_csv(self.diretorio + '/Settings/SE_inputs.csv')
            print('Salvando o SE report em ~/Settings/SE_inputs.csv')
                
