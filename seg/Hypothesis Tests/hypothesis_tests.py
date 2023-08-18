#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 10:38:47 2021

@author: eliezer
"""

import cv2
import numpy as np
    
def wilcoxon(x, y, alpha):    
    
    """
    
    This function was implemented by Eliezer Soares Flores 
    (https://scholar.google.com.br/citations?user=3jNjADcAAAAJ&hl=pt-BR&oi=ao).
    It is a Python implementation of the Wilcoxon's non parametric hypothesis 
    test and it is based on the Demsar's paper "Statistical Comparision of
    Classifiers over Multiple Data Sets".
        
    INPUTS: 
        
        x: first set of measurements (i.e., results of the first algorithm
                                      over a given dataset). 
        
        y: second set of measurements (i.e., results of the second algorithms
                                       over the same dataset).
            
        alpha: significance level. 
     
    USAGE EXAMPLE:    
     
    x = np.array([0.763, 0.599, 0.954, 0.628, 0.882, 0.936, 0.661, 0.583, 0.775, 1.000, 0.940, 0.619, 0.972, 0.957])
    y = np.array([0.768, 0.591, 0.971, 0.661, 0.888, 0.931, 0.668, 0.583, 0.838, 1.000, 0.962, 0.666, 0.981, 0.978])    
    
    wilcoxon(x, y, alpha)
            
    Ps.: this is the same example presented in page 8 of the Demsar's paper and
    it is based on data from Table 2 of such paper.
     
    Ps.: this function should be practically equivalent to:
        
        stat, p = st.wilcoxon(x, y)
        
        if p < alpha:
            print('Reject the null-hypothesis')
        else:
            print('Fail to reject the null-hypothesis')
            
        see https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/ 
        for details.
    
    """

    import numpy as np
    import scipy.stats as st    

    n = len(x)
    
    diff = y - x
    diff_abs = abs(y - x)
    ranks = st.rankdata(diff_abs)
    
    array_rzeros = ranks[diff == 0]
    if len(array_rzeros) % 2 == 1:
        array_rzeros = array_rzeros[:-1]
        
    rzeros = sum(array_rzeros)
      
    rplus = sum(ranks[diff > 0]) + 0.5*rzeros
    rminus = sum(ranks[diff < 0]) + 0.5*rzeros
    
    print('r+ = {:.0f}'.format(rplus))
    print('r- = {:.0f}'.format(rminus))
    
    s = min(rplus, rminus) 
    
    z = (s - 0.25*n*(n + 1)) / np.sqrt((1/24)*n*(n + 1)*(2*n + 1))
    print('z-score = {:.2f}'.format(z))
     
    cv = st.norm.ppf(alpha/2)
    print('critical value = {:.2f}'.format(cv))
    
    if z < cv:
        print('z-score < critical value => reject the null hypothesis')
    else:
        print('z-score >= critical value => fail to reject the null hypothesis')
    
    # p = st.norm.cdf(z)*2
    # print('p-value = {:.2f}'.format(p))
    
def friedman(T, alpha):

    """
    
    This function was implemented by Eliezer Soares Flores 
    (https://scholar.google.com.br/citations?user=3jNjADcAAAAJ&hl=pt-BR&oi=ao).
    It is a Python implementation of the Friedman's non parametric hypothesis 
    test and it is based on the Demsar's paper "Statistical Comparision of
    Classifiers over Multiple Data Sets".

    INPUTS: 
        
        T: n x p table, where n is the number of datasets and p is the number 
        of algorithms.
        
        alpha: significance level. 
        
    USAGE EXAMPLE:
        
    T = np.array([
        [0.763, 0.768, 0.771, 0.798],
        [0.599, 0.591, 0.590, 0.569],
        [0.954, 0.971, 0.968, 0.967],
        [0.628, 0.661, 0.654, 0.657],
        [0.882, 0.888, 0.886, 0.898],
        [0.936, 0.931, 0.916, 0.931],
        [0.661, 0.668, 0.609, 0.685],
        [0.583, 0.583, 0.563, 0.625],
        [0.775, 0.838, 0.866, 0.875],
        [1.000, 1.000, 1.000, 1.000],
        [0.940, 0.962, 0.965, 0.962],
        [0.619, 0.666, 0.614, 0.669],
        [0.972, 0.981, 0.97501, 0.975],
        [0.957, 0.978, 0.946, 0.970]
    ])
        
    friedman(T, 0.05)

    Ps.: this is the same example presented in page 13 of the Demsar's paper 
    and it is based on data from Table 6 of such paper. 

    Ps.: this function should be practically equivalent to:
        
        stat, p = st.friedmanchisquare(T[:,0], T[:,1], ..., T[:,p-1])
        
        if p < alpha:
            print('Reject the null-hypothesis')
        else:
            print('Fail to reject the null-hypothesis')
    
    see https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/ 
    for details.        

    """

    import numpy as np
    import scipy.stats as st   

    n = T.shape[0] 
    p = T.shape[1]
    
    for i in range(n):
        ranks_tmp = st.rankdata(T[i]) # Algorithms ranked for the i-th dataset i (1 = lowest, p = highest). 
        ranks = p - ranks_tmp + 1 # Algorithms ranked for the i-th dataset i (1 = highest, p = lowest). 
        if i == 0:
            R = ranks
        else:
            R = np.vstack((R, ranks))
        
    #print(R)
        
    #print('n = {}, p = {}'.format(n, p))
        
    avg_ranks = R.mean(axis=0)
    Xs = (12*n)/(p*(p+1))*(sum(avg_ranks**2) - p*(p+1)**2/4)
    
    F = ((n-1)*Xs)/(n*(p-1) - Xs)
    print('Chi-square = {:.2f} => F = {:.2f}'.format(Xs, F))
    
    cv = st.f.ppf(q=1-alpha, dfn=p-1, dfd=(p-1)*(n-1))
    print('Critical value = {:.2f}'.format(cv))
    
    if F > cv:
        print('=> F > critical value => reject the null-hypothesis')
    else:
        print('=> F <= critical value => fail to reject the null-hypothesis')
       
    perc_wins = np.sum(R==1,axis=0)/n
    
    return avg_ranks, perc_wins 
