#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:51:18 2020

@author: eliezer
"""
import matplotlib.pyplot as plt

#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 10})

import numpy as np
import os
import Orange

from hypothesis_tests import wilcoxon, friedman

# <= EDIT HERE (1 of 4)
#results_dir = '/home/eliezer/Área de Trabalho/thesis/seg/Metrics Evaluation Segmentation/Results_Hyperparameter_Tuning'
#results_dir = '/home/eliezer/Área de Trabalho/thesis/seg/Metrics Evaluation Segmentation/Results_Model_Selection'
results_dir = '/home/eliezer/Área de Trabalho/thesis/seg/Metrics Evaluation Segmentation/Results_SOTA'

# <= EDIT HERE (2 of 4)
dataset_name = 'dermquest' # dermnet, dermquest or dermis

# <= EDIT HERE (3 of 4)
#metric_name = 'xor'
metric_name = 'bf1'
#metric_name = 'time'
        
# Loading 'mat', which is a n x p matrix, 
# n is the number of datasets,
# p is the number of algorithms.
      
fnames_tmp = sorted(os.listdir(results_dir))
fnames = [fname for fname in fnames_tmp if dataset_name in fname and metric_name in fname and fname.endswith('.npy')]
        
for i, fname in enumerate(fnames):
    vec = np.load(os.path.join(results_dir, fname))
    if i == 0:
        mat = vec
    else:
        mat = np.vstack((mat, vec))
  
# Standardizing in a way that greater results are better ('mat' => 'T').     
if metric_name == 'iou' or metric_name == 'bf1' or metric_name == 'acc' or metric_name == 'sens' or metric_name == 'spec':    
    T = mat.T
elif metric_name == 'xor' or metric_name == 'time':
    T = 1 - mat.T
        
n = T.shape[0]
p = T.shape[1]    
 
# Setting the methods names and image size to be displayed.
methods_names = []

if results_dir.split('/')[-1] == 'Results_Hyperparameter_Tuning':
    figname = 'postest_seg_' + 'ht' + '_' + metric_name + '.pdf'
    for i, fname in enumerate(fnames):  
        method_name = fname.split('_')[0]
        methods_names.append('$K =$ ' + method_name[-1])
        if method_name[-1] == '3':
            cdmethod = i 
elif results_dir.split('/')[-1] == 'Results_Model_Selection':
    figname = 'postest_seg_' + 'ms' + '_' + metric_name + '.pdf'
    for i, fname in enumerate(fnames):
        method_name = fname.split('_')[0]
        if method_name == 'dictK3':
            methods_names.append('método proposto')
            cdmethod = i
        elif method_name == 'dictK3A1':
            methods_names.append('esquema alternativo 1')
        elif method_name == 'dictK3A2':
            methods_names.append('esquema alternativo 2')
        elif method_name == 'dictK3A3':
            methods_names.append('esquema alternativo 3')
        elif method_name == 'dictK3A4':
            methods_names.append('esquema alternativo 4')
        elif method_name == 'dictK3A5':
            methods_names.append('esquema alternativo 5')       
elif results_dir.split('/')[-1] == 'Results_SOTA':
    figname = 'postest_seg_' + 'sota' + '_' + metric_name + '_' + dataset_name +'.pdf'        
    for i, fname in enumerate(fnames):
        method_name = fname.split('_')[0]
        if method_name == 'dictK3':
            methods_names.append('método proposto')
            cdmethod = i
        elif method_name == 'dsnet':
            methods_names.append('DSNet')
        elif method_name == 'fcn':
            methods_names.append('FCNet')
        elif method_name == 'ftu':
            methods_names.append('FrCNet')
        elif method_name == 'jf':
            methods_names.append('LGPNet')
        elif method_name == 'pspnet':
            methods_names.append('PSPNet')
        elif method_name == 'segnet':
            methods_names.append('SegNet')
        elif method_name == 'unet':
            methods_names.append('U-Net')
        else:
            methods_names.append(method_name.upper())
        
# Significance level for the hypothesis tests.  
alpha = 0.05

# Comparing the proposed method with the alternatives. 
# Comparisons based on the Wilcoxon's test.
print('***************************************************************')
print('********************** WILCOXON\'S TESTS ***********************')
print('***************************************************************')
print('\n')

for i in range(len(fnames)):
    fnamei = fnames[i].split('_')[0]
    for j in range(len(fnames)):
        fnamej = fnames[j].split('_')[0]
        if fnamei == 'dictK3' and i != j:
            print('***************************************************************')
            print('{:s} vs {:s}'.format(fnamei, fnamej))
            print('***************************************************************')
            wilcoxon(T[:,i], T[:,j], alpha)
            print('***************************************************************')
            print('\n')

# Comparing all methods with the Friedman's test. 
print('***************************************************************')
print('*********************** FRIEDMAN\'S TEST ***********************')
print('***************************************************************')

avg_ranks, perc_wins = friedman(T, alpha)

for i, fname in enumerate(fnames):
    
    method_name = fname.split('_')[0]

    print('. {:s} method: average rank = {:.2f}, wins = {:.2f}%'.format(method_name, avg_ranks[i], 100*perc_wins[i]))

print('***************************************************************')
print('\n')


# Pos-hoc analysis for the Friedman's test. 

# <= EDIT HERE (4 of 4)
postest = 'bonferroni' #  1 vs all
#postest = 'nemenyi' # <= all vs all  

if postest == 'bonferroni':
    cd = Orange.evaluation.compute_CD(avg_ranks, n, alpha=str(alpha), test='bonferroni-dunn')
else:
    cd = Orange.evaluation.compute_CD(avg_ranks, n, alpha=str(alpha), test='nemenyi')
    
print('Critical difference = {:.2f}'.format(cd))

if postest == 'bonferroni':
    Orange.evaluation.graph_ranks(avg_ranks, methods_names, cd=cd, width=len(methods_names)/1.25, textspace=1.6, cdmethod=cdmethod, filename=figname)
else:
    Orange.evaluation.graph_ranks(avg_ranks, methods_names, cd=cd, width=len(methods_names)/1.25, textspace=1.6, filename=figname)
