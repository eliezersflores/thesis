#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 17:39:14 2021

@author: eliezer
"""

import numpy as np
import os

#txt_path = '/home/eliezer/Área de Trabalho/Experimentos Tese/Metrics Evaluation/Results_Hyperparameter_Tuning'
#fnames_list = ['dictK3_dermis_results', 'dictK4_dermis_results', 'dictK5_dermis_results', 'dictK6_dermis_results', 'dictK7_dermis_results', 'dictK8_dermis_results', 'dictK9_dermis_results']

txt_path = '/home/eliezer/Área de Trabalho/Experimentos Tese/Metrics Evaluation/Results_Model_Selection'
fnames_list = ['dictK3_dermis_results', 'dictK3A1_dermis_results', 'dictK3A2_dermis_results', 'dictK3A3_dermis_results', 'dictK3A4_dermis_results', 'dictK3A5_dermis_results']

dst = open('table.txt', 'w')

for fname in fnames_list:
        
    with open(os.path.join(txt_path, fname + '.txt')) as src:
        
        strings = src.readlines()
        
        for string in strings:
                                    
            if string.startswith('xor') or string.startswith('bf1') or string.startswith('time'):
                
                string_without_newline = string[:-1]
                string_value = string_without_newline.split(' ')[-1]
                float_value = np.float(string_value)
                
                if string.startswith('time'):
                    formatted_value = '{:.2f}'.format(float_value).replace('.', ',')
                else:
                    formatted_value = '{:.2f}'.format(100*float_value).replace('.', ',')    
                    
                if string.startswith('xor mean'):
                    dst.write(fname.split('_')[0] + ' & & ')
                    dst.write(formatted_value + ' & ')
                elif string.startswith('xor std') or string.startswith('bf1 std'):
                    dst.write(formatted_value + ' & & ') 
                elif string.startswith('time std'):
                    if fname == fnames_list[-1]:
                        dst.write(formatted_value + ' \\\\ \\hline')
                    else:
                        dst.write(formatted_value + ' \\\\\n')
                elif string.startswith('xor') or string.startswith('bf1') or string.startswith('time'):
                    dst.write(formatted_value + ' & ')
                                 
    src.close()

dst.close()

"""



"""